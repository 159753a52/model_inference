"""
LLM推理框架配置模块

提供模型配置(ModelConfig)和推理引擎配置(EngineConfig)的定义与加载。
支持从默认值、字典、JSON文件加载配置。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Union, Literal

import torch


@dataclass
class ModelConfig:
    """
    模型结构配置
    
    定义decoder-only Transformer模型的核心超参数。
    """
    # 模型架构
    vocab_size: int = 32000
    hidden_dim: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: Optional[int] = None  # GQA支持，None表示与num_heads相同
    intermediate_dim: int = 11008  # FFN中间层维度
    
    # 位置编码
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0  # RoPE基频
    
    # 归一化
    rms_norm_eps: float = 1e-5
    
    # Attention配置
    attention_bias: bool = True  # QKV是否有bias (Qwen2有, LLaMA没有)
    
    # 其他
    tie_word_embeddings: bool = False
    
    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> ModelConfig:
        """从字典创建配置"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> ModelConfig:
        """从JSON文件加载配置"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def save_json(self, json_path: Union[str, Path]) -> None:
        """保存为JSON文件"""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


@dataclass
class EngineConfig:
    """
    推理引擎配置
    
    控制推理行为：设备、精度、序列长度、批量大小等。
    """
    # 设备与精度
    device: str = "cuda"  # "cuda" 或 "cpu"
    dtype: str = "float16"  # "float16", "bfloat16", "float32"
    
    # 序列与批量
    max_seq_len: int = 2048
    max_batch_size: int = 8
    
    # 生成参数默认值
    default_max_new_tokens: int = 256
    default_temperature: float = 1.0
    default_top_p: float = 1.0
    default_top_k: int = 50
    
    # 内存管理
    max_memory_gb: Optional[float] = None  # 最大GPU内存使用限制
    
    # KV Cache配置（后续阶段使用）
    use_kv_cache: bool = True
    
    # 日志级别
    log_level: str = "INFO"
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self) -> None:
        """验证配置有效性"""
        valid_devices = {"cuda", "cpu"}
        if self.device not in valid_devices:
            raise ValueError(f"device必须是{valid_devices}之一，得到: {self.device}")
        
        valid_dtypes = {"float16", "bfloat16", "float32"}
        if self.dtype not in valid_dtypes:
            raise ValueError(f"dtype必须是{valid_dtypes}之一，得到: {self.dtype}")
        
        if self.device == "cpu" and self.dtype == "float16":
            # CPU不支持float16，自动降级到float32
            self.dtype = "float32"
    
    @property
    def torch_dtype(self) -> torch.dtype:
        """获取对应的torch.dtype"""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map[self.dtype]
    
    @property
    def torch_device(self) -> torch.device:
        """获取对应的torch.device"""
        return torch.device(self.device)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> EngineConfig:
        """从字典创建配置"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> EngineConfig:
        """从JSON文件加载配置"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))
    
    @classmethod
    def from_env(cls, prefix: str = "LLM_ENGINE_") -> EngineConfig:
        """
        从环境变量加载配置
        
        例如: LLM_ENGINE_DEVICE=cuda, LLM_ENGINE_MAX_SEQ_LEN=4096
        """
        config_dict = {}
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        
        for name in field_names:
            env_key = f"{prefix}{name.upper()}"
            if env_key in os.environ:
                value = os.environ[env_key]
                # 尝试类型转换
                field_type = cls.__dataclass_fields__[name].type
                if field_type in (int, "int"):
                    value = int(value)
                elif field_type in (float, "float"):
                    value = float(value)
                elif field_type in (bool, "bool"):
                    value = value.lower() in ("true", "1", "yes")
                config_dict[name] = value
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def save_json(self, json_path: Union[str, Path]) -> None:
        """保存为JSON文件"""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def merge(self, overrides: Dict[str, Any]) -> EngineConfig:
        """合并配置覆盖项，返回新配置"""
        merged = self.to_dict()
        merged.update(overrides)
        return EngineConfig.from_dict(merged)


@dataclass 
class GenerationConfig:
    """
    文本生成配置
    
    控制单次生成的参数。
    """
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.0
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> GenerationConfig:
        """从字典创建配置"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
