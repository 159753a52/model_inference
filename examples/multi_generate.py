#!/usr/bin/env python3
"""
多请求生成示例 (阶段3)

演示如何使用LLMEngine同时处理多个生成请求。
"""

import sys
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
from my_llm_engine.models import build_decoder_only_model
from my_llm_engine.engine import LLMEngine, Scheduler, generate


def main():
    setup_logging(level="INFO")
    logger = get_logger(__name__)
    
    logger.info(f"=== My LLM Engine v{__version__} - Multi-Request Demo ===")
    
    # 1. 加载配置和模型
    configs_dir = project_root / "configs"
    model_config = ModelConfig.from_json(configs_dir / "tiny_model.json")
    
    engine_config = EngineConfig(
        device="cpu",
        dtype="float32",
        max_seq_len=128,
        max_batch_size=4,  # 最多同时处理4个请求
    )
    
    model = build_decoder_only_model(model_config, engine_config)
    logger.info(f"模型加载完成: layers={model_config.num_layers}, hidden={model_config.hidden_dim}")
    
    # 2. 创建Engine
    scheduler = Scheduler(
        max_batch_size=engine_config.max_batch_size,
        max_seq_len=engine_config.max_seq_len,
    )
    engine = LLMEngine(
        model=model,
        model_config=model_config,
        engine_config=engine_config,
        scheduler=scheduler,
        logger=logger,
    )
    
    logger.info("\n--- 测试1: 单请求生成 ---")
    
    # 单请求
    prompt1 = torch.randint(0, model_config.vocab_size, (10,))
    gen_config = GenerationConfig(max_new_tokens=15, do_sample=False)
    
    req_id1 = engine.add_request(prompt1, gen_config)
    engine.run_until_complete()
    
    response1 = engine.get_response(req_id1)
    logger.info(f"请求 {req_id1}: prompt_len={response1.prompt_len}, "
                f"generated={response1.num_generated}, status={response1.status.value}")
    logger.info(f"输出序列长度: {len(response1.get_full_sequence())}")
    
    # 3. 测试多请求
    logger.info("\n--- 测试2: 多请求批量生成 ---")
    
    engine.reset()
    
    # 创建多个不同长度的prompt
    prompts = [
        torch.randint(0, model_config.vocab_size, (5,)),   # 短prompt
        torch.randint(0, model_config.vocab_size, (10,)),  # 中等prompt
        torch.randint(0, model_config.vocab_size, (15,)),  # 较长prompt
        torch.randint(0, model_config.vocab_size, (8,)),   # 另一个prompt
    ]
    
    gen_config = GenerationConfig(max_new_tokens=10, do_sample=False)
    
    # 添加所有请求
    request_ids = []
    for i, prompt in enumerate(prompts):
        req_id = engine.add_request(prompt, gen_config, request_id=f"req_{i}")
        request_ids.append(req_id)
        logger.info(f"添加请求 {req_id}, prompt_len={len(prompt)}")
    
    # 逐步执行并观察
    step = 0
    while scheduler.has_unfinished_requests():
        stats = engine.step()
        step += 1
        
        if stats["prefill"] > 0 or stats["finished"] > 0:
            logger.info(f"Step {step}: prefill={stats['prefill']}, "
                       f"decode={stats['decode']}, finished={stats['finished']}")
    
    logger.info(f"总共执行 {step} 步")
    
    # 获取结果
    for req_id in request_ids:
        req = engine.get_response(req_id)
        logger.info(f"请求 {req_id}: prompt={req.prompt_len}, "
                   f"generated={req.num_generated}, total={req.current_len}")
    
    # 4. 测试generate_batch
    logger.info("\n--- 测试3: generate_batch API ---")
    
    engine.reset()
    
    batch_prompts = [
        torch.randint(0, model_config.vocab_size, (6,)),
        torch.randint(0, model_config.vocab_size, (8,)),
        torch.randint(0, model_config.vocab_size, (4,)),
    ]
    
    results = engine.generate_batch(
        batch_prompts,
        gen_config=GenerationConfig(max_new_tokens=12, do_sample=False),
    )
    
    for i, result in enumerate(results):
        logger.info(f"Batch结果 {i}: prompt={len(batch_prompts[i])}, "
                   f"total={len(result)}, generated={len(result) - len(batch_prompts[i])}")
    
    # 5. 对比单请求generate和Engine的一致性
    logger.info("\n--- 测试4: 单请求generate vs Engine一致性 ---")
    
    test_prompt = torch.randint(0, model_config.vocab_size, (7,))
    test_config = GenerationConfig(max_new_tokens=10, do_sample=False)
    
    # 使用单请求generate
    result_single = generate(
        model, test_prompt.unsqueeze(0), model_config, engine_config, test_config,
        use_kv_cache=True
    )
    
    # 使用Engine
    engine.reset()
    engine.add_request(test_prompt, test_config, request_id="compare")
    engine.run_until_complete()
    result_engine = engine.get_response("compare").get_full_sequence()
    
    if torch.equal(result_single.squeeze(0), result_engine):
        logger.info("单请求generate与Engine结果完全一致!")
    else:
        logger.warning("结果不一致!")
        logger.info(f"Single: {result_single.squeeze(0).tolist()}")
        logger.info(f"Engine: {result_engine.tolist()}")
    
    # 6. 动态添加请求测试
    logger.info("\n--- 测试5: 动态添加请求（Continuous Batching模拟） ---")
    
    engine.reset()
    
    # 先添加2个请求
    prompt_a = torch.randint(0, model_config.vocab_size, (8,))
    prompt_b = torch.randint(0, model_config.vocab_size, (6,))
    
    engine.add_request(prompt_a, GenerationConfig(max_new_tokens=20, do_sample=False), request_id="A")
    engine.add_request(prompt_b, GenerationConfig(max_new_tokens=20, do_sample=False), request_id="B")
    
    logger.info("添加请求 A 和 B")
    
    # 执行几步
    for _ in range(5):
        engine.step()
    
    stats = engine.get_stats()
    logger.info(f"执行5步后: waiting={stats['waiting']}, running={stats['running']}")
    
    # 在中途添加新请求
    prompt_c = torch.randint(0, model_config.vocab_size, (10,))
    engine.add_request(prompt_c, GenerationConfig(max_new_tokens=15, do_sample=False), request_id="C")
    logger.info("动态添加请求 C")
    
    # 继续执行直到完成
    engine.run_until_complete()
    
    stats = engine.get_stats()
    logger.info(f"完成: finished={stats['finished']}, total_steps={stats['total_steps']}")
    
    for req_id in ["A", "B", "C"]:
        req = engine.get_response(req_id)
        logger.info(f"请求 {req_id}: generated={req.num_generated}")
    
    logger.info("\n=== 阶段3验证完成! ===")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
