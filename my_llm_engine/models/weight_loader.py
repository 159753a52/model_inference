"""
权重加载器模块

支持从ModelScope或HuggingFace加载预训练模型权重。
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

from my_llm_engine.config import ModelConfig, EngineConfig
from my_llm_engine.models.transformer import DecoderOnlyModel
from my_llm_engine.logging_utils import get_logger


def load_model_from_modelscope(
    model_id: str,
    engine_config: Optional[EngineConfig] = None,
    cache_dir: Optional[str] = None,
    logger=None,
) -> tuple:
    """
    从ModelScope加载预训练模型
    
    Args:
        model_id: 模型ID，如 "qwen/Qwen2-0.5B"
        engine_config: 引擎配置
        cache_dir: 缓存目录
        logger: 日志器
        
    Returns:
        (model, tokenizer, model_config, engine_config) 元组
    """
    if logger is None:
        logger = get_logger(__name__)
    
    logger.info(f"加载模型: {model_id}")
    
    # 使用transformers加载（它会自动处理ModelScope镜像）
    from transformers import AutoTokenizer, AutoConfig
    
    # 设置ModelScope镜像
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface")
    
    # 加载tokenizer
    logger.info("加载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    logger.info(f"Tokenizer加载完成, vocab_size={len(tokenizer)}")
    
    # 加载配置
    logger.info("加载模型配置...")
    hf_config = AutoConfig.from_pretrained(
        model_id, 
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    # 转换为我们的ModelConfig
    model_config = convert_hf_config(hf_config.to_dict())
    logger.info(f"模型配置: layers={model_config.num_layers}, "
                f"hidden={model_config.hidden_dim}, heads={model_config.num_heads}")
    
    # 创建引擎配置
    if engine_config is None:
        engine_config = EngineConfig(
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype="float32",
            max_seq_len=min(model_config.max_position_embeddings, 2048),
            max_batch_size=1,
        )
    
    # 创建模型
    logger.info("创建模型结构...")
    model = DecoderOnlyModel(model_config)
    
    # 下载并加载权重
    logger.info("下载并加载权重...")
    from transformers.utils import cached_file
    
    # 尝试找到权重文件
    try:
        # 尝试safetensors
        weight_file = cached_file(
            model_id, 
            "model.safetensors",
            cache_dir=cache_dir,
        )
        model_dir = Path(weight_file).parent
    except Exception:
        try:
            # 尝试分片的safetensors
            weight_file = cached_file(
                model_id,
                "model.safetensors.index.json", 
                cache_dir=cache_dir,
            )
            model_dir = Path(weight_file).parent
        except Exception:
            # 尝试pytorch格式
            weight_file = cached_file(
                model_id,
                "pytorch_model.bin",
                cache_dir=cache_dir,
            )
            model_dir = Path(weight_file).parent
    
    logger.info(f"权重目录: {model_dir}")
    load_weights_from_hf_format(model, str(model_dir), logger)
    
    # 移动到设备
    device = engine_config.torch_device
    dtype = engine_config.torch_dtype
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    logger.info(f"模型加载完成, device={device}, dtype={dtype}")
    
    return model, tokenizer, model_config, engine_config


def convert_hf_config(hf_config: Dict) -> ModelConfig:
    """
    将HuggingFace配置转换为我们的ModelConfig
    
    支持Qwen2、LLaMA等模型格式
    """
    # 尝试获取各种可能的配置键名
    vocab_size = hf_config.get("vocab_size", 32000)
    hidden_dim = hf_config.get("hidden_size", hf_config.get("hidden_dim", 4096))
    num_layers = hf_config.get("num_hidden_layers", hf_config.get("num_layers", 32))
    num_heads = hf_config.get("num_attention_heads", hf_config.get("num_heads", 32))
    num_kv_heads = hf_config.get("num_key_value_heads", num_heads)
    intermediate_dim = hf_config.get("intermediate_size", hf_config.get("intermediate_dim", 11008))
    max_position = hf_config.get("max_position_embeddings", 4096)
    rope_theta = hf_config.get("rope_theta", 10000.0)
    rms_norm_eps = hf_config.get("rms_norm_eps", 1e-5)
    tie_embeddings = hf_config.get("tie_word_embeddings", False)
    
    # Qwen2有attention bias，LLaMA没有
    # 默认True以兼容Qwen2
    attention_bias = hf_config.get("attention_bias", True)
    
    return ModelConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_dim=intermediate_dim,
        max_position_embeddings=max_position,
        rope_theta=rope_theta,
        rms_norm_eps=rms_norm_eps,
        attention_bias=attention_bias,
        tie_word_embeddings=tie_embeddings,
    )


def load_weights_from_hf_format(
    model: DecoderOnlyModel,
    model_dir: str,
    logger=None,
) -> None:
    """
    从HuggingFace格式加载权重到我们的模型
    
    支持safetensors和pytorch格式
    """
    if logger is None:
        logger = get_logger(__name__)
    
    model_dir = Path(model_dir)
    
    # 查找权重文件
    safetensor_files = list(model_dir.glob("*.safetensors"))
    bin_files = list(model_dir.glob("*.bin"))
    
    if safetensor_files:
        logger.info(f"使用safetensors格式加载, 共{len(safetensor_files)}个文件")
        state_dict = load_safetensors(safetensor_files)
    elif bin_files:
        logger.info(f"使用pytorch格式加载, 共{len(bin_files)}个文件")
        state_dict = load_pytorch_bins(bin_files)
    else:
        raise FileNotFoundError(f"未找到权重文件: {model_dir}")
    
    # 映射权重
    mapped_state_dict = map_weights(state_dict, model, logger)
    
    # 加载到模型
    missing, unexpected = model.load_state_dict(mapped_state_dict, strict=False)
    
    if missing:
        logger.warning(f"缺失的权重: {missing[:5]}..." if len(missing) > 5 else f"缺失的权重: {missing}")
    if unexpected:
        logger.warning(f"未使用的权重: {unexpected[:5]}..." if len(unexpected) > 5 else f"未使用的权重: {unexpected}")
    
    logger.info("权重加载完成")


def load_safetensors(files: list) -> Dict[str, torch.Tensor]:
    """加载safetensors文件"""
    from safetensors import safe_open
    
    state_dict = {}
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                state_dict[key] = sf.get_tensor(key)
    return state_dict


def load_pytorch_bins(files: list) -> Dict[str, torch.Tensor]:
    """加载pytorch bin文件"""
    state_dict = {}
    for f in files:
        data = torch.load(f, map_location="cpu")
        state_dict.update(data)
    return state_dict


def map_weights(
    hf_state_dict: Dict[str, torch.Tensor],
    model: DecoderOnlyModel,
    logger=None,
) -> Dict[str, torch.Tensor]:
    """
    将HuggingFace权重映射到我们的模型结构
    
    支持Qwen2和LLaMA格式
    """
    if logger is None:
        logger = get_logger(__name__)
    
    mapped = {}
    
    # 检测模型类型
    sample_key = list(hf_state_dict.keys())[0]
    if "model.embed_tokens" in sample_key or any("model.embed_tokens" in k for k in hf_state_dict.keys()):
        prefix = "model."
    else:
        prefix = ""
    
    logger.info(f"检测到权重前缀: '{prefix}'")
    
    # Embedding
    embed_key = f"{prefix}embed_tokens.weight"
    if embed_key in hf_state_dict:
        mapped["embed_tokens.weight"] = hf_state_dict[embed_key]
    
    # 每一层的权重映射
    num_layers = model.num_layers
    for i in range(num_layers):
        layer_prefix = f"{prefix}layers.{i}."
        our_prefix = f"layers.{i}."
        
        # Attention
        # Q/K/V projections (weight and bias)
        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            weight_key = f"{layer_prefix}self_attn.{proj}.weight"
            bias_key = f"{layer_prefix}self_attn.{proj}.bias"
            
            if weight_key in hf_state_dict:
                mapped[f"{our_prefix}self_attn.{proj}.weight"] = hf_state_dict[weight_key]
            if bias_key in hf_state_dict:
                mapped[f"{our_prefix}self_attn.{proj}.bias"] = hf_state_dict[bias_key]
        
        # MLP
        gate_key = f"{layer_prefix}mlp.gate_proj.weight"
        up_key = f"{layer_prefix}mlp.up_proj.weight"
        down_key = f"{layer_prefix}mlp.down_proj.weight"
        
        if gate_key in hf_state_dict:
            mapped[f"{our_prefix}mlp.gate_proj.weight"] = hf_state_dict[gate_key]
        if up_key in hf_state_dict:
            mapped[f"{our_prefix}mlp.up_proj.weight"] = hf_state_dict[up_key]
        if down_key in hf_state_dict:
            mapped[f"{our_prefix}mlp.down_proj.weight"] = hf_state_dict[down_key]
        
        # RMSNorm
        input_norm_key = f"{layer_prefix}input_layernorm.weight"
        post_norm_key = f"{layer_prefix}post_attention_layernorm.weight"
        
        if input_norm_key in hf_state_dict:
            mapped[f"{our_prefix}input_layernorm.weight"] = hf_state_dict[input_norm_key]
        if post_norm_key in hf_state_dict:
            mapped[f"{our_prefix}post_attention_layernorm.weight"] = hf_state_dict[post_norm_key]
    
    # Final norm
    final_norm_key = f"{prefix}norm.weight"
    if final_norm_key in hf_state_dict:
        mapped["norm.weight"] = hf_state_dict[final_norm_key]
    
    # LM head
    lm_head_key = "lm_head.weight"
    if lm_head_key in hf_state_dict:
        mapped["lm_head.weight"] = hf_state_dict[lm_head_key]
    elif model.config.tie_word_embeddings:
        # 共享embedding权重
        mapped["lm_head.weight"] = mapped.get("embed_tokens.weight")
    
    logger.info(f"映射了 {len(mapped)} 个权重")
    return mapped
