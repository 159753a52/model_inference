#!/usr/bin/env python3
"""
基础示例：展示配置系统和日志工具的使用

这是阶段0的烟雾测试脚本，验证：
1. 配置类可以正常导入和实例化
2. 日志系统可以正常工作
3. 类型别名可以正常使用
"""

import sys
from pathlib import Path

# 添加项目根目录到路径（开发模式下使用）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from my_llm_engine import (
    ModelConfig,
    EngineConfig,
    GenerationConfig,
    get_logger,
    setup_logging,
    Tensor,
    HiddenStates,
    __version__,
)


def main():
    # 初始化日志
    setup_logging(level="DEBUG")
    logger = get_logger(__name__)
    
    logger.info(f"=== My LLM Engine v{__version__} ===")
    logger.info("阶段0：项目骨架验证")
    
    # 1. 测试默认配置
    logger.info("\n--- 测试默认配置 ---")
    model_config = ModelConfig()
    engine_config = EngineConfig()
    
    logger.info(f"模型配置: hidden_dim={model_config.hidden_dim}, "
                f"num_layers={model_config.num_layers}, "
                f"num_heads={model_config.num_heads}")
    logger.info(f"引擎配置: device={engine_config.device}, "
                f"dtype={engine_config.dtype}, "
                f"max_seq_len={engine_config.max_seq_len}")
    
    # 2. 测试从JSON加载配置
    logger.info("\n--- 测试从JSON加载配置 ---")
    configs_dir = project_root / "configs"
    
    if (configs_dir / "tiny_model.json").exists():
        tiny_config = ModelConfig.from_json(configs_dir / "tiny_model.json")
        logger.info(f"加载tiny_model配置: hidden_dim={tiny_config.hidden_dim}, "
                    f"num_layers={tiny_config.num_layers}")
    
    if (configs_dir / "default_engine.json").exists():
        engine_from_json = EngineConfig.from_json(configs_dir / "default_engine.json")
        logger.info(f"加载引擎配置: device={engine_from_json.device}, "
                    f"max_memory_gb={engine_from_json.max_memory_gb}")
    
    # 3. 测试配置覆盖
    logger.info("\n--- 测试配置覆盖 ---")
    custom_config = engine_config.merge({"max_seq_len": 4096, "max_batch_size": 16})
    logger.info(f"覆盖后: max_seq_len={custom_config.max_seq_len}, "
                f"max_batch_size={custom_config.max_batch_size}")
    
    # 4. 测试torch属性
    logger.info("\n--- 测试torch属性转换 ---")
    logger.info(f"torch_device: {engine_config.torch_device}")
    logger.info(f"torch_dtype: {engine_config.torch_dtype}")
    
    # 5. 测试类型别名（仅作展示，不影响运行）
    logger.info("\n--- 测试类型别名 ---")
    batch_size, seq_len, hidden_dim = 2, 10, 256
    
    # 创建一个示例张量
    sample_tensor: HiddenStates = torch.randn(batch_size, seq_len, hidden_dim)
    logger.info(f"创建HiddenStates张量: shape={sample_tensor.shape}, "
                f"dtype={sample_tensor.dtype}")
    
    # 6. 测试生成配置
    logger.info("\n--- 测试生成配置 ---")
    gen_config = GenerationConfig(
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    logger.info(f"生成配置: max_new_tokens={gen_config.max_new_tokens}, "
                f"temperature={gen_config.temperature}, "
                f"top_p={gen_config.top_p}")
    
    # 7. 测试配置保存
    logger.info("\n--- 测试配置保存 ---")
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    tiny_config.save_json(output_dir / "saved_model_config.json")
    engine_config.save_json(output_dir / "saved_engine_config.json")
    logger.info(f"配置已保存到 {output_dir}")
    
    logger.info("\n=== 阶段0验证完成！===")
    logger.info("所有基础组件工作正常。")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
