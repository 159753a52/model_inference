"""
Transformer模型模块

实现LLaMA风格的Decoder-only Transformer。
支持prefill和decode两阶段推理。
"""

from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

from my_llm_engine.config import ModelConfig, EngineConfig
from my_llm_engine.tensor_types import TokenIds, HiddenStates, Logits, Tensor
from my_llm_engine.models.layers import RMSNorm, MLP
from my_llm_engine.models.attention import SelfAttention

if TYPE_CHECKING:
    from my_llm_engine.engine.kv_cache import KVCache


class DecoderLayer(nn.Module):
    """
    单个Transformer解码层
    
    结构 (Pre-Norm, LLaMA风格):
        hidden = hidden + self_attn(norm1(hidden))
        hidden = hidden + mlp(norm2(hidden))
    """
    
    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        """
        Args:
            config: 模型配置
            layer_idx: 层索引 (用于KV Cache索引)
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_dim = config.hidden_dim
        
        # 自注意力
        self.self_attn = SelfAttention(config)
        
        # MLP
        self.mlp = MLP(
            hidden_dim=config.hidden_dim,
            intermediate_dim=config.intermediate_dim,
        )
        
        # RMSNorm (Pre-Norm)
        self.input_layernorm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: HiddenStates,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        kv_cache: Optional["KVCache"] = None,
        past_seq_len: int = 0,
    ) -> HiddenStates:
        """
        前向传播
        
        Args:
            hidden_states: 输入, shape [batch, seq_len, hidden_dim]
            attention_mask: 注意力掩码
            position_ids: 位置ID
            kv_cache: KV缓存
            past_seq_len: 已缓存的序列长度
            
        Returns:
            输出张量, shape [batch, seq_len, hidden_dim]
        """
        # 残差连接1: Self-Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            layer_idx=self.layer_idx,
            past_seq_len=past_seq_len,
        )
        hidden_states = residual + hidden_states
        
        # 残差连接2: MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class DecoderOnlyModel(nn.Module):
    """
    Decoder-only Transformer模型
    
    支持三种调用方式:
    1. forward(): 兼容模式，不使用KV Cache
    2. prefill(): 处理完整prompt，返回logits和填充好的KVCache
    3. decode(): 增量解码，使用KVCache
    """
    
    def __init__(self, config: ModelConfig):
        """
        Args:
            config: 模型配置
        """
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        
        # Token Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Decoder Layers
        self.layers = nn.ModuleList([
            DecoderLayer(config, layer_idx=i) for i in range(config.num_layers)
        ])
        
        # Final Norm
        self.norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        
        # LM Head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # 权重共享 (可选)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: TokenIds,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Logits:
        """
        兼容模式的前向传播（不使用KV Cache）
        
        Args:
            input_ids: 输入token ID, shape [batch, seq_len]
            attention_mask: 可选的attention mask
            position_ids: 可选的位置ID
            
        Returns:
            logits: 输出logits, shape [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token Embedding
        hidden_states = self.embed_tokens(input_ids)
        
        # 生成position_ids
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            position_ids = position_ids.expand(batch_size, -1)
        
        # 构建causal attention mask（传None让attention层自动构建）
        # 这里保持与阶段1的兼容性
        
        # 通过所有Decoder层（不使用KV Cache）
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=None,
                past_seq_len=0,
            )
        
        # Final Norm
        hidden_states = self.norm(hidden_states)
        
        # LM Head
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def prefill(
        self,
        input_ids: TokenIds,
        kv_cache: "KVCache",
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Logits:
        """
        Prefill阶段：处理完整prompt，填充KV Cache
        
        Args:
            input_ids: 输入token ID, shape [batch, seq_len]
            kv_cache: 空的KV Cache，将被填充
            attention_mask: 可选的attention mask
            position_ids: 可选的位置ID, 默认为[0, 1, ..., seq_len-1]
            
        Returns:
            logits: 输出logits, shape [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token Embedding
        hidden_states = self.embed_tokens(input_ids)
        
        # 生成position_ids（从0开始）
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            position_ids = position_ids.expand(batch_size, -1)
        
        # 通过所有Decoder层，填充KV Cache
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
                past_seq_len=0,  # prefill从0开始
            )
        
        # Final Norm
        hidden_states = self.norm(hidden_states)
        
        # LM Head
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def decode(
        self,
        input_ids: TokenIds,
        kv_cache: "KVCache",
        past_seq_len: int,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Logits:
        """
        Decode阶段：增量解码，使用KV Cache
        
        Args:
            input_ids: 新token ID, shape [batch, new_seq_len]，通常new_seq_len=1
            kv_cache: 包含历史KV的Cache
            past_seq_len: 已缓存的序列长度
            attention_mask: 可选的attention mask
            position_ids: 可选的位置ID, 默认为[past_seq_len, past_seq_len+1, ...]
            
        Returns:
            logits: 输出logits, shape [batch, new_seq_len, vocab_size]
        """
        batch_size, new_seq_len = input_ids.shape
        device = input_ids.device
        
        # Token Embedding
        hidden_states = self.embed_tokens(input_ids)
        
        # 生成position_ids（从past_seq_len开始）
        if position_ids is None:
            position_ids = torch.arange(
                past_seq_len, past_seq_len + new_seq_len, device=device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # 通过所有Decoder层，更新KV Cache
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
                past_seq_len=past_seq_len,
            )
        
        # Final Norm
        hidden_states = self.norm(hidden_states)
        
        # LM Head
        logits = self.lm_head(hidden_states)
        
        return logits


def build_decoder_only_model(
    model_config: ModelConfig,
    engine_config: EngineConfig,
) -> DecoderOnlyModel:
    """
    工厂函数：构建DecoderOnlyModel
    
    Args:
        model_config: 模型配置
        engine_config: 引擎配置
        
    Returns:
        配置好的DecoderOnlyModel实例
    """
    model = DecoderOnlyModel(model_config)
    
    # 移动到指定设备和精度
    device = engine_config.torch_device
    dtype = engine_config.torch_dtype
    
    model = model.to(device=device, dtype=dtype)
    model.eval()  # 推理模式
    
    return model
