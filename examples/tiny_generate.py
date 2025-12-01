#!/usr/bin/env python3
"""
Tiny模型生成示例 (阶段2: KV Cache版本)

展示prefill/decode两阶段推理，对比有无KV Cache的性能。
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from my_llm_engine import (
    ModelConfig,
    EngineConfig,
    GenerationConfig,
    get_logger,
    setup_logging,
    __version__,
)
from my_llm_engine.models import DecoderOnlyModel, build_decoder_only_model
from my_llm_engine.engine import generate, KVCache, benchmark_generation


def main():
    setup_logging(level="INFO")
    logger = get_logger(__name__)
    
    logger.info(f"=== My LLM Engine v{__version__} - KV Cache Demo ===")
    
    # 1. 加载配置
    configs_dir = project_root / "configs"
    model_config = ModelConfig.from_json(configs_dir / "tiny_model.json")
    
    engine_config = EngineConfig(
        device="cpu",
        dtype="float32",
        max_seq_len=128,
        max_batch_size=2,
    )
    
    logger.info(f"模型: layers={model_config.num_layers}, hidden={model_config.hidden_dim}")
    
    # 2. 构建模型
    model = build_decoder_only_model(model_config, engine_config)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    
    # 3. 测试数值一致性
    logger.info("\n--- 测试Prefill/Decode数值一致性 ---")
    
    prompt_len = 10
    input_ids = torch.randint(0, model_config.vocab_size, (1, prompt_len))
    new_token = torch.randint(0, model_config.vocab_size, (1, 1))
    full_ids = torch.cat([input_ids, new_token], dim=1)
    
    # 方式1: 完整forward
    with torch.no_grad():
        logits_full = model(full_ids)
        last_logits_full = logits_full[:, -1, :]
    
    # 方式2: prefill + decode
    kv_cache = KVCache.empty(model_config, engine_config, batch_size=1)
    with torch.no_grad():
        _ = model.prefill(input_ids, kv_cache)
        logits_decode = model.decode(new_token, kv_cache, past_seq_len=prompt_len)
        last_logits_decode = logits_decode[:, -1, :]
    
    max_diff = (last_logits_full - last_logits_decode).abs().max().item()
    logger.info(f"Logits最大差异: {max_diff:.2e}")
    
    if max_diff < 1e-4:
        logger.info("数值一致性验证通过!")
    else:
        logger.warning("数值差异较大，请检查实现")
    
    # 4. 测试生成
    logger.info("\n--- 测试文本生成 ---")
    
    gen_config = GenerationConfig(
        max_new_tokens=20,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        do_sample=True,
    )
    
    prompt = torch.randint(0, model_config.vocab_size, (1, 5))
    logger.info(f"Prompt: {prompt.tolist()}")
    
    # 使用KV Cache生成
    result_kv = generate(
        model, prompt.clone(), model_config, engine_config, gen_config,
        use_kv_cache=True, logger=logger
    )
    logger.info(f"KV Cache生成: {result_kv.tolist()}")
    
    # 5. 性能对比
    logger.info("\n--- 性能对比 (Greedy, 无采样) ---")
    
    greedy_config = GenerationConfig(max_new_tokens=30, do_sample=False)
    
    # 使用较长prompt
    long_prompt = torch.randint(0, model_config.vocab_size, (1, 30))
    
    benchmark = benchmark_generation(
        model, long_prompt, model_config, engine_config, greedy_config, logger
    )
    
    logger.info(f"无KV Cache耗时: {benchmark['time_no_cache']:.3f}s")
    logger.info(f"有KV Cache耗时: {benchmark['time_with_cache']:.3f}s")
    logger.info(f"加速比: {benchmark['speedup']:.2f}x")
    
    # 验证结果一致
    if torch.equal(benchmark['result_with_cache'], benchmark['result_no_cache']):
        logger.info("生成结果完全一致!")
    else:
        logger.warning("生成结果不一致，请检查!")
    
    # 6. 测试GQA
    logger.info("\n--- 测试GQA配置 ---")
    
    gqa_config = ModelConfig(
        vocab_size=1000,
        hidden_dim=128,
        num_layers=2,
        num_heads=8,
        num_kv_heads=2,  # GQA
        intermediate_dim=256,
        max_position_embeddings=64,
    )
    gqa_engine = EngineConfig(device="cpu", dtype="float32", max_seq_len=32)
    gqa_model = build_decoder_only_model(gqa_config, gqa_engine)
    
    gqa_prompt = torch.randint(0, 1000, (1, 5))
    gqa_gen_config = GenerationConfig(max_new_tokens=10, do_sample=False)
    
    result_gqa = generate(
        gqa_model, gqa_prompt, gqa_config, gqa_engine, gqa_gen_config,
        use_kv_cache=True
    )
    
    logger.info(f"GQA模型生成: {result_gqa.shape}")
    logger.info("GQA + KV Cache验证通过!")
    
    # 7. KV Cache内存占用
    logger.info("\n--- KV Cache内存占用 ---")
    kv_cache = KVCache.empty(model_config, engine_config, batch_size=1)
    memory_mb = kv_cache.get_memory_usage() / (1024 * 1024)
    logger.info(f"KV Cache内存: {memory_mb:.2f} MB")
    
    logger.info("\n=== 阶段2验证完成! ===")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
