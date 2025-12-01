"""
LLM推理框架张量类型定义模块

提供类型别名以增强代码可读性和类型安全性。

约定:
- B: batch size
- S: sequence length  
- H: hidden dimension
- N: number of heads
- D: head dimension (H // N)
- V: vocabulary size
"""

from __future__ import annotations

from typing import TypeVar, Tuple, Union, List

import torch


# 基础张量别名
Tensor = torch.Tensor

# 按维度语义命名的张量类型别名
# 这些是类型注释，用于提高代码可读性

# [B, S] - token ids
TokenIds = torch.Tensor

# [B, S, H] - hidden states
HiddenStates = torch.Tensor

# [B, N, S, D] - attention中的Q/K/V张量
AttentionTensor = torch.Tensor

# [B, N, S, S] - attention分数/权重
AttentionScores = torch.Tensor

# [B, S, V] - logits输出
Logits = torch.Tensor

# [B, S] - attention mask
AttentionMask = torch.Tensor

# [S, D] - 位置编码
PositionalEncoding = torch.Tensor

# KV Cache相关类型
# 单层KV: (K, V), 每个形状 [B, N, S, D]
KVPair = Tuple[torch.Tensor, torch.Tensor]

# 所有层的KV Cache: List[(K, V)]
KVCache = List[KVPair]

# 可选的KV Cache
OptionalKVCache = Union[KVCache, None]


# 设备类型
Device = Union[str, torch.device]

# 数据类型
DType = torch.dtype


# 形状相关的类型变量（用于泛型）
BatchSize = TypeVar('BatchSize', bound=int)
SeqLen = TypeVar('SeqLen', bound=int)
HiddenDim = TypeVar('HiddenDim', bound=int)


def check_tensor_shape(
    tensor: Tensor,
    expected_dims: int,
    name: str = "tensor"
) -> None:
    """
    检查张量维度数量
    
    Args:
        tensor: 要检查的张量
        expected_dims: 期望的维度数
        name: 张量名称（用于错误信息）
        
    Raises:
        ValueError: 如果维度不匹配
    """
    if tensor.dim() != expected_dims:
        raise ValueError(
            f"{name}应该有{expected_dims}维，但得到{tensor.dim()}维, "
            f"形状为{tuple(tensor.shape)}"
        )


def check_tensor_dtype(
    tensor: Tensor,
    expected_dtype: DType,
    name: str = "tensor"
) -> None:
    """
    检查张量数据类型
    
    Args:
        tensor: 要检查的张量
        expected_dtype: 期望的数据类型
        name: 张量名称（用于错误信息）
        
    Raises:
        ValueError: 如果类型不匹配
    """
    if tensor.dtype != expected_dtype:
        raise ValueError(
            f"{name}应该是{expected_dtype}，但得到{tensor.dtype}"
        )
