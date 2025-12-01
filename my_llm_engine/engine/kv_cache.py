"""
KV Cache模块

管理Transformer各层的Key/Value缓存，支持prefill和decode两阶段推理。
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from my_llm_engine.config import ModelConfig, EngineConfig
from my_llm_engine.tensor_types import Tensor, KVPair


class KVCache:
    """
    KV缓存管理器
    
    为每一层维护K和V的缓存张量，支持增量更新。
    
    存储形状:
        k_cache[layer]: [batch, num_kv_heads, max_seq_len, head_dim]
        v_cache[layer]: [batch, num_kv_heads, max_seq_len, head_dim]
    """
    
    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Args:
            num_layers: Transformer层数
            batch_size: 批量大小
            max_seq_len: 最大序列长度
            num_kv_heads: KV头数量（GQA时可能小于Q头数）
            head_dim: 每个头的维度
            device: 设备
            dtype: 数据类型
        """
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        
        # 当前已填充的序列长度
        self._seq_len = 0
        
        # 预分配缓存空间
        cache_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        self.k_cache: List[Tensor] = [
            torch.zeros(cache_shape, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.v_cache: List[Tensor] = [
            torch.zeros(cache_shape, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
    
    @classmethod
    def empty(
        cls,
        model_config: ModelConfig,
        engine_config: EngineConfig,
        batch_size: int,
    ) -> "KVCache":
        """
        工厂方法：创建空的KVCache
        
        Args:
            model_config: 模型配置
            engine_config: 引擎配置
            batch_size: 批量大小
            
        Returns:
            初始化的KVCache实例
        """
        num_kv_heads = model_config.num_kv_heads or model_config.num_heads
        head_dim = model_config.hidden_dim // model_config.num_heads
        
        return cls(
            num_layers=model_config.num_layers,
            batch_size=batch_size,
            max_seq_len=engine_config.max_seq_len,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device=engine_config.torch_device,
            dtype=engine_config.torch_dtype,
        )
    
    @property
    def seq_len(self) -> int:
        """当前已缓存的序列长度"""
        return self._seq_len
    
    def get_layer_kv(self, layer_idx: int, seq_len: Optional[int] = None) -> KVPair:
        """
        获取指定层的KV缓存
        
        Args:
            layer_idx: 层索引
            seq_len: 要获取的序列长度，None表示获取全部已缓存的
            
        Returns:
            (k, v) 元组，每个shape为 [batch, num_kv_heads, seq_len, head_dim]
        """
        if seq_len is None:
            seq_len = self._seq_len
        
        k = self.k_cache[layer_idx][:, :, :seq_len, :]
        v = self.v_cache[layer_idx][:, :, :seq_len, :]
        return k, v
    
    def update(
        self,
        layer_idx: int,
        key: Tensor,
        value: Tensor,
        start_pos: int,
    ) -> KVPair:
        """
        更新指定层的KV缓存
        
        Args:
            layer_idx: 层索引
            key: 新的K张量, shape [batch, num_kv_heads, new_seq_len, head_dim]
            value: 新的V张量, shape [batch, num_kv_heads, new_seq_len, head_dim]
            start_pos: 写入的起始位置
            
        Returns:
            更新后的完整KV对 (k, v)，包含历史+新增部分
        """
        new_seq_len = key.shape[2]
        end_pos = start_pos + new_seq_len
        
        # 写入缓存
        self.k_cache[layer_idx][:, :, start_pos:end_pos, :] = key
        self.v_cache[layer_idx][:, :, start_pos:end_pos, :] = value
        
        # 更新序列长度（取最大值，因为可能多层并行更新）
        self._seq_len = max(self._seq_len, end_pos)
        
        # 返回完整的KV（从0到end_pos）
        k_full = self.k_cache[layer_idx][:, :, :end_pos, :]
        v_full = self.v_cache[layer_idx][:, :, :end_pos, :]
        
        return k_full, v_full
    
    def reset(self) -> None:
        """重置缓存（清零并重置序列长度）"""
        self._seq_len = 0
        for layer_idx in range(self.num_layers):
            self.k_cache[layer_idx].zero_()
            self.v_cache[layer_idx].zero_()
    
    def get_memory_usage(self) -> int:
        """获取缓存占用的内存（字节）"""
        single_cache_size = (
            self.batch_size * self.num_kv_heads * 
            self.max_seq_len * self.head_dim
        )
        element_size = self.k_cache[0].element_size()
        # K和V各num_layers个
        return 2 * self.num_layers * single_cache_size * element_size
