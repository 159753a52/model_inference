"""
基础层模块

实现LLaMA风格的RMSNorm和MLP(SwiGLU)。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from my_llm_engine.tensor_types import HiddenStates


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    公式: x * weight / sqrt(mean(x^2) + eps)
    相比LayerNorm，RMSNorm去掉了均值中心化，计算更高效。
    """
    
    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        """
        Args:
            hidden_dim: 隐藏层维度
            eps: 数值稳定性的小常数
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))
    
    def forward(self, x: HiddenStates) -> HiddenStates:
        """
        Args:
            x: 输入张量, shape [batch, seq, hidden_dim]
        Returns:
            归一化后的张量, shape [batch, seq, hidden_dim]
        """
        # 在float32下计算以避免float16下溢
        input_dtype = x.dtype
        x = x.float()
        
        # 计算RMS: sqrt(mean(x^2))
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        
        return (self.weight * x_normed).to(input_dtype)


class MLP(nn.Module):
    """
    LLaMA风格的MLP层 (SwiGLU变体)
    
    结构: output = W_down(SiLU(W_gate(x)) * W_up(x))
    
    其中SiLU(x) = x * sigmoid(x)，也称为Swish激活函数。
    """
    
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        """
        Args:
            hidden_dim: 输入/输出维度
            intermediate_dim: 中间层维度
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        
        # LLaMA SwiGLU: gate_proj和up_proj同时作用
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
    
    def forward(self, x: HiddenStates) -> HiddenStates:
        """
        Args:
            x: 输入张量, shape [batch, seq, hidden_dim]
        Returns:
            输出张量, shape [batch, seq, hidden_dim]
        """
        # SwiGLU: SiLU(gate) * up
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
