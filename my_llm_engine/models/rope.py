"""
旋转位置编码 (Rotary Position Embedding, RoPE)

RoPE通过旋转变换将位置信息编码到Q和K中，
使得注意力分数只依赖于相对位置。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from my_llm_engine.tensor_types import Tensor


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码模块
    
    预计算频率表，支持动态序列长度。
    RoPE作用于head_dim的全部维度（按对划分）。
    """
    
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        device: torch.device = None,
    ):
        """
        Args:
            head_dim: 每个注意力头的维度
            max_position_embeddings: 最大位置数
            rope_theta: RoPE的基础频率
            device: 设备
        """
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        
        # 预计算逆频率: 1 / (theta^(2i/d)) for i = 0, 1, ..., d/2-1
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # 缓存cos/sin表
        self._cos_cached = None
        self._sin_cached = None
        self._cached_seq_len = 0
    
    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """更新cos/sin缓存表"""
        if seq_len <= self._cached_seq_len and self._cos_cached is not None:
            return
        
        self._cached_seq_len = max(seq_len, self.max_position_embeddings)
        
        # 位置序列: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(self._cached_seq_len, device=device, dtype=torch.float32)
        
        # 计算 t * inv_freq: [seq_len, head_dim/2]
        freqs = torch.outer(t, self.inv_freq.to(device))
        
        # 扩展到完整head_dim: [seq_len, head_dim]
        # 每个频率对应两个维度(cos和sin作用于相邻对)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self._cos_cached = emb.cos().to(dtype)
        self._sin_cached = emb.sin().to(dtype)
    
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        position_ids: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        对Q和K应用旋转位置编码
        
        Args:
            q: Query张量, shape [batch, num_heads, seq_len, head_dim]
            k: Key张量, shape [batch, num_kv_heads, seq_len, head_dim]
            position_ids: 位置ID, shape [batch, seq_len]
            
        Returns:
            (q_rotated, k_rotated): 应用RoPE后的Q和K
        """
        seq_len = q.shape[2]
        self._update_cos_sin_cache(seq_len, q.device, q.dtype)
        
        # 根据position_ids索引cos/sin
        # position_ids: [batch, seq_len] -> 需要获取对应位置的cos/sin
        cos = self._cos_cached[position_ids]  # [batch, seq_len, head_dim]
        sin = self._sin_cached[position_ids]  # [batch, seq_len, head_dim]
        
        # 扩展维度以匹配q/k: [batch, 1, seq_len, head_dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        
        # 应用旋转
        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)
        
        return q_rotated, k_rotated
    
    def _apply_rotary(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """
        应用旋转变换
        
        对于每对相邻维度(x0, x1)，旋转公式为:
        x0' = x0 * cos - x1 * sin
        x1' = x0 * sin + x1 * cos
        """
        # 将x分成两半: [..., head_dim/2] each
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]
        
        # 构造旋转后的向量
        # rotate_half: [x2, -x1] 的效果等价于将x旋转90度
        cos1 = cos[..., : self.head_dim // 2]
        cos2 = cos[..., self.head_dim // 2 :]
        sin1 = sin[..., : self.head_dim // 2]
        sin2 = sin[..., self.head_dim // 2 :]
        
        # 标准RoPE旋转
        rotated_x1 = x1 * cos1 - x2 * sin1
        rotated_x2 = x1 * sin2 + x2 * cos2
        
        return torch.cat([rotated_x1, rotated_x2], dim=-1)


def apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    函数式RoPE应用（备选接口）
    
    Args:
        q: [batch, num_heads, seq_len, head_dim]
        k: [batch, num_kv_heads, seq_len, head_dim]
        cos: [batch, 1, seq_len, head_dim] or broadcastable
        sin: [batch, 1, seq_len, head_dim] or broadcastable
        
    Returns:
        (q_rotated, k_rotated)
    """
    def rotate_half(x: Tensor) -> Tensor:
        """将后半部分取负并与前半部分交换"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)
    
    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)
    
    return q_rotated, k_rotated
