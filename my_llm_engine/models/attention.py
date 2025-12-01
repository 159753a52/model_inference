"""
多头自注意力模块

支持标准MHA和GQA (Grouped Query Attention)。
支持KV Cache用于增量推理。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from my_llm_engine.config import ModelConfig
from my_llm_engine.tensor_types import HiddenStates, Tensor
from my_llm_engine.models.rope import RotaryEmbedding

if TYPE_CHECKING:
    from my_llm_engine.engine.kv_cache import KVCache


class SelfAttention(nn.Module):
    """
    多头自注意力层，支持GQA和KV Cache
    
    GQA: Q有num_heads个头，K/V有num_kv_heads个头。
    当num_kv_heads < num_heads时，多个Q头共享同一组K/V头。
    
    KV Cache模式:
    - prefill: past_seq_len=0, 对整段序列计算并缓存K/V
    - decode: past_seq_len>0, 只对新token计算，复用历史K/V
    """
    
    def __init__(self, config: ModelConfig):
        """
        Args:
            config: 模型配置
        """
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads or config.num_heads
        self.head_dim = self.hidden_dim // self.num_heads
        
        # GQA: 每个KV头对应多少个Q头
        self.num_groups = self.num_heads // self.num_kv_heads
        
        # Q/K/V投影 (Qwen2等模型的QKV有bias，o_proj没有)
        attention_bias = getattr(config, 'attention_bias', True)  # 默认启用bias以兼容Qwen2
        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim, bias=False)
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
        )
        
        # 缩放因子
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self,
        hidden_states: HiddenStates,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        kv_cache: Optional["KVCache"] = None,
        layer_idx: int = 0,
        past_seq_len: int = 0,
    ) -> HiddenStates:
        """
        前向传播，支持prefill和decode两种模式
        
        Args:
            hidden_states: 输入, shape [batch, seq_len, hidden_dim]
                          prefill时seq_len为prompt长度，decode时通常为1
            attention_mask: 注意力掩码, shape [batch, 1, seq_len, total_seq_len]
                           None时自动构造causal mask
            position_ids: 位置ID, shape [batch, seq_len]
                         prefill时为[0,1,...,S-1]，decode时为[past_seq_len]
            kv_cache: KV缓存对象，None表示不使用缓存
            layer_idx: 当前层索引，用于访问kv_cache
            past_seq_len: 已缓存的序列长度，0表示prefill模式
            
        Returns:
            输出张量, shape [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. 线性投影到Q/K/V
        q = self.q_proj(hidden_states)  # [B, S, num_heads * head_dim]
        k = self.k_proj(hidden_states)  # [B, S, num_kv_heads * head_dim]
        v = self.v_proj(hidden_states)  # [B, S, num_kv_heads * head_dim]
        
        # 2. Reshape为多头格式
        # Q: [B, S, num_heads, head_dim] -> [B, num_heads, S, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 3. 生成默认position_ids
        if position_ids is None:
            # 根据past_seq_len生成正确的位置ID
            position_ids = torch.arange(
                past_seq_len, past_seq_len + seq_len,
                device=hidden_states.device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # 4. 应用RoPE到Q和K（在缓存之前）
        q, k = self.rotary_emb(q, k, position_ids)
        
        # 5. 处理KV Cache
        if kv_cache is not None:
            # 更新缓存并获取完整的K/V（历史+新增）
            k, v = kv_cache.update(layer_idx, k, v, past_seq_len)
            # 现在k, v的shape为 [B, num_kv_heads, past_seq_len + seq_len, head_dim]
        
        # 获取attention计算时的总序列长度
        total_seq_len = k.shape[2]
        
        # 6. GQA: 扩展K/V以匹配Q的头数
        if self.num_groups > 1:
            k = self._repeat_kv(k, self.num_groups)
            v = self._repeat_kv(v, self.num_groups)
        
        # 7. 计算注意力分数: Q @ K^T / sqrt(d)
        # Q: [B, num_heads, seq_len, head_dim]
        # K: [B, num_heads, total_seq_len, head_dim]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # attn_weights: [B, num_heads, seq_len, total_seq_len]
        
        # 8. 应用注意力掩码
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        else:
            # 自动构造causal mask
            causal_mask = self._make_causal_mask(
                seq_len, total_seq_len, past_seq_len,
                hidden_states.device, hidden_states.dtype
            )
            attn_weights = attn_weights + causal_mask
        
        # 9. Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        
        # 10. 加权求和: attn @ V
        attn_output = torch.matmul(attn_weights, v)
        # attn_output: [B, num_heads, seq_len, head_dim]
        
        # 11. Reshape回原始形状
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        
        # 12. 输出投影
        output = self.o_proj(attn_output)
        
        return output
    
    def _repeat_kv(self, x: Tensor, n_rep: int) -> Tensor:
        """
        将KV头扩展n_rep次以匹配Q头数量
        
        Args:
            x: [batch, num_kv_heads, seq_len, head_dim]
            n_rep: 重复次数
            
        Returns:
            [batch, num_kv_heads * n_rep, seq_len, head_dim]
        """
        if n_rep == 1:
            return x
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
    
    def _make_causal_mask(
        self,
        query_len: int,
        key_len: int,
        past_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """
        构造因果注意力掩码
        
        在decode模式下，query只有1个token，需要能看到所有历史token。
        
        Args:
            query_len: 当前查询长度（prefill时=prompt_len，decode时=1）
            key_len: 键的总长度（past_seq_len + query_len）
            past_seq_len: 已缓存的序列长度
            device: 设备
            dtype: 数据类型
            
        Returns:
            掩码张量 [1, 1, query_len, key_len]
        """
        # 创建causal mask矩阵
        # 对于decode模式(query_len=1): 新token可以看到所有历史token
        # 对于prefill模式: 标准的下三角掩码
        
        mask = torch.zeros(query_len, key_len, device=device, dtype=dtype)
        
        if query_len > 1:
            # Prefill模式：构造下三角掩码
            # 位置i只能看到位置0到i（包含）
            for i in range(query_len):
                # 当前位置i对应的绝对位置是 past_seq_len + i
                # 它可以看到位置 0 到 past_seq_len + i
                # 在key中，这对应索引 0 到 past_seq_len + i
                mask[i, past_seq_len + i + 1:] = float("-inf")
        # 对于decode模式(query_len=1)，新token可以看到所有历史，不需要mask
        
        return mask.unsqueeze(0).unsqueeze(0)
