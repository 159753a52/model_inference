"""
文本生成模块

实现基于自回归的文本生成，支持KV Cache加速。
"""

from __future__ import annotations

import logging
import time
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

from my_llm_engine.config import ModelConfig, EngineConfig, GenerationConfig
from my_llm_engine.tensor_types import TokenIds, Logits, Tensor
from my_llm_engine.logging_utils import get_logger
from my_llm_engine.engine.kv_cache import KVCache

if TYPE_CHECKING:
    from my_llm_engine.models.transformer import DecoderOnlyModel


def generate(
    model: "DecoderOnlyModel",
    input_ids: TokenIds,
    model_config: ModelConfig,
    engine_config: EngineConfig,
    gen_config: Optional[GenerationConfig] = None,
    eos_token_id: Optional[int] = None,
    use_kv_cache: bool = True,
    logger: Optional[logging.Logger] = None,
) -> TokenIds:
    """
    自回归文本生成
    
    支持两种模式:
    - use_kv_cache=True: prefill + decode模式，高效增量推理
    - use_kv_cache=False: 每步重算整个序列（兼容模式）
    
    Args:
        model: DecoderOnlyModel实例
        input_ids: 初始token序列, shape [batch, seq_len]
        model_config: 模型配置
        engine_config: 引擎配置
        gen_config: 生成配置，None时使用默认值
        eos_token_id: EOS token ID，遇到则停止生成
        use_kv_cache: 是否使用KV Cache
        logger: 可选的logger
        
    Returns:
        生成的完整序列, shape [batch, seq_len + generated_len]
    """
    if logger is None:
        logger = get_logger(__name__)
    
    if gen_config is None:
        gen_config = GenerationConfig()
    
    if eos_token_id is None:
        eos_token_id = model_config.vocab_size - 1
    
    if use_kv_cache:
        return _generate_with_kv_cache(
            model, input_ids, model_config, engine_config,
            gen_config, eos_token_id, logger
        )
    else:
        return _generate_without_kv_cache(
            model, input_ids, model_config, engine_config,
            gen_config, eos_token_id, logger
        )


def _generate_with_kv_cache(
    model: "DecoderOnlyModel",
    input_ids: TokenIds,
    model_config: ModelConfig,
    engine_config: EngineConfig,
    gen_config: GenerationConfig,
    eos_token_id: int,
    logger: logging.Logger,
) -> TokenIds:
    """使用KV Cache的生成（prefill + decode模式）"""
    
    device = input_ids.device
    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    max_new_tokens = gen_config.max_new_tokens
    max_seq_len = engine_config.max_seq_len
    
    # 检查序列长度
    if prompt_len + max_new_tokens > max_seq_len:
        logger.warning(
            f"prompt_len({prompt_len}) + max_new_tokens({max_new_tokens}) > "
            f"max_seq_len({max_seq_len}), 将截断生成"
        )
        max_new_tokens = max_seq_len - prompt_len
    
    # 创建KV Cache
    kv_cache = KVCache.empty(model_config, engine_config, batch_size)
    
    # 跟踪哪些序列已完成
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    logger.debug(f"开始生成(KV Cache): batch={batch_size}, prompt_len={prompt_len}, "
                 f"max_new_tokens={max_new_tokens}")
    
    with torch.no_grad():
        # === Prefill阶段 ===
        logits = model.prefill(input_ids, kv_cache)
        # logits: [batch, prompt_len, vocab_size]
        
        # 取最后一个token的logits用于第一次采样
        next_token_logits = logits[:, -1, :]
        
        current_seq_len = prompt_len
        generated_tokens = []
        
        # === Decode循环 ===
        for step in range(max_new_tokens):
            # 采样下一个token
            next_tokens = sample_next_token(
                next_token_logits,
                temperature=gen_config.temperature,
                top_k=gen_config.top_k,
                top_p=gen_config.top_p,
                do_sample=gen_config.do_sample,
            )  # [batch]
            
            generated_tokens.append(next_tokens)
            
            # 检查EOS
            finished = finished | (next_tokens == eos_token_id)
            if finished.all():
                logger.debug(f"所有序列在第 {step + 1} 步完成")
                break
            
            # Decode: 只处理新token
            new_input_ids = next_tokens.unsqueeze(-1)  # [batch, 1]
            
            logits = model.decode(
                new_input_ids,
                kv_cache,
                past_seq_len=current_seq_len,
            )
            # logits: [batch, 1, vocab_size]
            
            next_token_logits = logits[:, -1, :]
            current_seq_len += 1
    
    # 拼接所有生成的token
    if generated_tokens:
        generated = torch.stack(generated_tokens, dim=1)  # [batch, num_generated]
        result = torch.cat([input_ids, generated], dim=1)
    else:
        result = input_ids
    
    logger.debug(f"生成完成: 最终序列长度={result.shape[1]}")
    return result


def _generate_without_kv_cache(
    model: "DecoderOnlyModel",
    input_ids: TokenIds,
    model_config: ModelConfig,
    engine_config: EngineConfig,
    gen_config: GenerationConfig,
    eos_token_id: int,
    logger: logging.Logger,
) -> TokenIds:
    """不使用KV Cache的生成（每步重算整个序列）"""
    
    device = input_ids.device
    batch_size = input_ids.shape[0]
    max_new_tokens = gen_config.max_new_tokens
    max_seq_len = engine_config.max_seq_len
    
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    logger.debug(f"开始生成(无KV Cache): batch={batch_size}, max_new_tokens={max_new_tokens}")
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            if input_ids.shape[1] >= max_seq_len:
                logger.warning(f"达到最大序列长度 {max_seq_len}，停止生成")
                break
            
            # 前向传播（每次重算整个序列）
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]
            
            # 采样
            next_tokens = sample_next_token(
                next_token_logits,
                temperature=gen_config.temperature,
                top_k=gen_config.top_k,
                top_p=gen_config.top_p,
                do_sample=gen_config.do_sample,
            )
            
            # 拼接
            next_tokens = next_tokens.unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
            # 检查EOS
            finished = finished | (next_tokens.squeeze(-1) == eos_token_id)
            if finished.all():
                logger.debug(f"所有序列在第 {step + 1} 步完成")
                break
    
    logger.debug(f"生成完成: 最终序列长度={input_ids.shape[1]}")
    return input_ids


def sample_next_token(
    logits: Logits,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    do_sample: bool = True,
) -> TokenIds:
    """
    从logits中采样下一个token
    
    Args:
        logits: 未归一化的logits, shape [batch, vocab_size]
        temperature: 温度参数
        top_k: Top-K采样
        top_p: Top-P采样
        do_sample: 是否采样
        
    Returns:
        采样的token ID, shape [batch]
    """
    if not do_sample:
        return logits.argmax(dim=-1)
    
    if temperature != 1.0:
        logits = logits / temperature
    
    if top_k > 0 and top_k < logits.shape[-1]:
        logits = top_k_filtering(logits, top_k)
    
    if top_p < 1.0:
        logits = top_p_filtering(logits, top_p)
    
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return next_token


def top_k_filtering(logits: Tensor, top_k: int) -> Tensor:
    """Top-K过滤"""
    top_k = min(top_k, logits.shape[-1])
    indices_to_remove = logits < torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits


def top_p_filtering(logits: Tensor, top_p: float) -> Tensor:
    """Top-P (Nucleus) 过滤"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, float("-inf"))
    
    return logits


def greedy_decode(
    model: "DecoderOnlyModel",
    input_ids: TokenIds,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
) -> TokenIds:
    """贪婪解码（简化版本，不使用KV Cache）"""
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
    
    return input_ids


def benchmark_generation(
    model: "DecoderOnlyModel",
    input_ids: TokenIds,
    model_config: ModelConfig,
    engine_config: EngineConfig,
    gen_config: GenerationConfig,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """
    对比有无KV Cache的生成性能
    
    Returns:
        包含时间和结果对比的字典
    """
    if logger is None:
        logger = get_logger(__name__)
    
    # 无KV Cache
    start_time = time.time()
    result_no_cache = generate(
        model, input_ids.clone(), model_config, engine_config, gen_config,
        use_kv_cache=False, logger=logger
    )
    time_no_cache = time.time() - start_time
    
    # 有KV Cache
    start_time = time.time()
    result_with_cache = generate(
        model, input_ids.clone(), model_config, engine_config, gen_config,
        use_kv_cache=True, logger=logger
    )
    time_with_cache = time.time() - start_time
    
    speedup = time_no_cache / time_with_cache if time_with_cache > 0 else float('inf')
    
    return {
        "time_no_cache": time_no_cache,
        "time_with_cache": time_with_cache,
        "speedup": speedup,
        "result_no_cache": result_no_cache,
        "result_with_cache": result_with_cache,
    }
