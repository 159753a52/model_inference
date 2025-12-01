#!/usr/bin/env python3
"""
真实模型推理示例

从ModelScope加载预训练模型，使用我们的推理框架进行生成。
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from my_llm_engine import (
    GenerationConfig,
    get_logger,
    setup_logging,
    __version__,
)
from my_llm_engine.models.weight_loader import load_model_from_modelscope
from my_llm_engine.engine import generate, LLMEngine


def main():
    setup_logging(level="INFO")
    logger = get_logger(__name__)
    
    logger.info(f"=== My LLM Engine v{__version__} - Real Model Inference ===")
    
    # 检查可用内存
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, 内存: {gpu_mem:.1f}GB")
        device = "cuda"
        dtype = "float16"
    else:
        logger.info("使用CPU推理")
        device = "cpu"
        dtype = "float32"
    
    # 选择模型（考虑内存限制，使用较小的模型）
    model_id = "Qwen/Qwen2-0.5B"  # ~1GB，最安全
    # model_id = "Qwen/Qwen2-1.5B"  # ~3GB
    # model_id = "Qwen/Qwen2-7B"  # ~14GB
    
    logger.info(f"\n=== 加载模型: {model_id} ===")
    
    from my_llm_engine.config import EngineConfig
    engine_config = EngineConfig(
        device=device,
        dtype=dtype,
        max_seq_len=512,  # 限制最大序列长度以节省内存
        max_batch_size=1,
    )
    
    try:
        model, tokenizer, model_config, engine_config = load_model_from_modelscope(
            model_id=model_id,
            engine_config=engine_config,
            logger=logger,
        )
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return 1
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {total_params:,} ({total_params/1e9:.2f}B)")
    
    # 测试推理
    logger.info("\n=== 开始推理测试 ===")
    
    prompts = [
        "Hello, my name is",
        "The capital of China is",
        "人工智能的未来发展",
        "请用Python写一个快速排序算法：",
    ]
    
    gen_config = GenerationConfig(
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=True,
    )
    
    for prompt in prompts:
        logger.info(f"\n--- Prompt: {prompt} ---")
        
        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(engine_config.torch_device)
        
        logger.info(f"Input tokens: {input_ids.shape[1]}")
        
        # Generate
        with torch.no_grad():
            output_ids = generate(
                model=model,
                input_ids=input_ids,
                model_config=model_config,
                engine_config=engine_config,
                gen_config=gen_config,
                use_kv_cache=True,
                logger=logger,
            )
        
        # Decode
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logger.info(f"Output: {output_text}")
    
    # Greedy解码测试
    logger.info("\n=== Greedy解码测试 ===")
    
    prompt = "1 + 1 ="
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(engine_config.torch_device)
    
    greedy_config = GenerationConfig(max_new_tokens=20, do_sample=False)
    
    with torch.no_grad():
        output_ids = generate(
            model=model,
            input_ids=input_ids,
            model_config=model_config,
            engine_config=engine_config,
            gen_config=greedy_config,
            use_kv_cache=True,
        )
    
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Output: {output_text}")
    
    logger.info("\n=== 推理测试完成! ===")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
