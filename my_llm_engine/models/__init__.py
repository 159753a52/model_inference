"""
模型模块

包含:
- layers.py: RMSNorm, MLP等基础层
- rope.py: 旋转位置编码
- attention.py: 多头自注意力(支持GQA)
- transformer.py: DecoderLayer & DecoderOnlyModel
- weight_loader.py: 预训练权重加载
"""

from my_llm_engine.models.layers import RMSNorm, MLP
from my_llm_engine.models.rope import RotaryEmbedding, apply_rotary_pos_emb
from my_llm_engine.models.attention import SelfAttention
from my_llm_engine.models.transformer import (
    DecoderLayer,
    DecoderOnlyModel,
    build_decoder_only_model,
)
from my_llm_engine.models.weight_loader import (
    load_model_from_modelscope,
    convert_hf_config,
    load_weights_from_hf_format,
)

__all__ = [
    "RMSNorm",
    "MLP",
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "SelfAttention",
    "DecoderLayer",
    "DecoderOnlyModel",
    "build_decoder_only_model",
    "load_model_from_modelscope",
    "convert_hf_config",
    "load_weights_from_hf_format",
]
