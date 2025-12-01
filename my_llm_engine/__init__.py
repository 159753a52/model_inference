"""
My LLM Engine - 轻量级大语言模型推理框架

一个从零开始构建的LLM推理引擎，支持:
- 加载decoder-only Transformer模型（类LLaMA/GPT风格）
- KV Cache优化
- Prefill + Decode两阶段推理
- 批量推理与简单调度

阶段0: 项目骨架与基础配置系统
"""

__version__ = "0.1.0"
__author__ = "LLM Engine Team"

from my_llm_engine.config import (
    ModelConfig,
    EngineConfig,
    GenerationConfig,
)
from my_llm_engine.logging_utils import (
    get_logger,
    setup_logging,
    set_log_level,
    LoggerMixin,
)
from my_llm_engine.tensor_types import (
    Tensor,
    TokenIds,
    HiddenStates,
    AttentionTensor,
    AttentionScores,
    Logits,
    AttentionMask,
    KVPair,
    KVCache,
    OptionalKVCache,
    Device,
    DType,
)

__all__ = [
    # 版本信息
    "__version__",
    # 配置类
    "ModelConfig",
    "EngineConfig",
    "GenerationConfig",
    # 日志工具
    "get_logger",
    "setup_logging",
    "set_log_level",
    "LoggerMixin",
    # 类型别名
    "Tensor",
    "TokenIds",
    "HiddenStates",
    "AttentionTensor",
    "AttentionScores",
    "Logits",
    "AttentionMask",
    "KVPair",
    "KVCache",
    "OptionalKVCache",
    "Device",
    "DType",
]
