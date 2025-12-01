"""
分页式KV缓存模块

基于BlockManager的分页KV缓存实现。
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import torch

from my_llm_engine.engine.block_manager import BlockManager
from my_llm_engine.tensor_types import KVPair
from my_llm_engine.logging_utils import get_logger


@dataclass
class SequenceKVState:
    """
    单个序列在各层的KV状态
    
    记录每个序列使用的block和当前长度。
    """
    seq_id: int
    num_layers: int
    block_size: int
    
    # 每层使用的block列表
    block_tables: List[List[int]] = field(default_factory=list)
    
    # 当前序列长度
    seq_len: int = 0
    
    def __post_init__(self):
        if not self.block_tables:
            self.block_tables = [[] for _ in range(self.num_layers)]
    
    def get_num_blocks(self) -> int:
        """获取已使用的block数（所有层共享同一组block）"""
        if self.block_tables:
            return len(self.block_tables[0])
        return 0
    
    def get_current_block_offset(self) -> int:
        """获取当前block内的偏移"""
        return self.seq_len % self.block_size
    
    def needs_new_block(self) -> bool:
        """是否需要新block"""
        if self.seq_len == 0:
            return True
        return self.seq_len % self.block_size == 0


class PagedKVCache:
    """
    分页式KV缓存
    
    为每个序列管理SequenceKVState，通过BlockManager进行底层存储。
    
    接口与DenseKVCache类似，便于切换。
    """
    
    def __init__(
        self,
        block_manager: BlockManager,
        max_blocks_per_seq: int = 64,
    ):
        """
        Args:
            block_manager: 全局block管理器
            max_blocks_per_seq: 每个序列最大block数
        """
        self.block_manager = block_manager
        self.max_blocks_per_seq = max_blocks_per_seq
        
        self._seq_states: Dict[int, SequenceKVState] = {}
        self._next_seq_id: int = 0
        
        self._logger = get_logger(__name__)
    
    @property
    def num_layers(self) -> int:
        return self.block_manager.num_layers
    
    @property
    def block_size(self) -> int:
        return self.block_manager.block_size
    
    def create_sequence(self, seq_id: Optional[int] = None) -> int:
        """
        为新序列创建KV状态
        
        Args:
            seq_id: 序列ID，None则自动分配
            
        Returns:
            分配的序列ID
        """
        if seq_id is None:
            seq_id = self._next_seq_id
            self._next_seq_id += 1
        
        state = SequenceKVState(
            seq_id=seq_id,
            num_layers=self.num_layers,
            block_size=self.block_size,
        )
        self._seq_states[seq_id] = state
        
        self._logger.debug(f"创建序列 {seq_id}")
        return seq_id
    
    def delete_sequence(self, seq_id: int) -> None:
        """
        删除序列，释放其所有block
        
        Args:
            seq_id: 序列ID
        """
        if seq_id not in self._seq_states:
            return
        
        state = self._seq_states[seq_id]
        
        # 释放所有层的block（实际上所有层共用同一组block）
        if state.block_tables and state.block_tables[0]:
            self.block_manager.free_blocks(state.block_tables[0])
        
        del self._seq_states[seq_id]
        self._logger.debug(f"删除序列 {seq_id}")
    
    def append_kv(
        self,
        seq_id: int,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """
        向序列追加K/V
        
        Args:
            seq_id: 序列ID
            layer_idx: 层索引
            key: K张量, shape [1, num_kv_heads, num_tokens, head_dim]
            value: V张量, shape [1, num_kv_heads, num_tokens, head_dim]
        """
        if seq_id not in self._seq_states:
            raise ValueError(f"序列 {seq_id} 不存在")
        
        state = self._seq_states[seq_id]
        num_tokens = key.shape[2]
        
        # 移除batch维度 [num_kv_heads, num_tokens, head_dim]
        key = key.squeeze(0)
        value = value.squeeze(0)
        
        # 逐token写入（简化实现，后续可优化为批量写入）
        for i in range(num_tokens):
            # 检查是否需要新block
            if state.needs_new_block():
                self._allocate_block_for_seq(state)
            
            # 获取当前block和偏移
            block_id = state.block_tables[layer_idx][-1]
            offset = state.get_current_block_offset()
            
            # 写入单个token
            k_token = key[:, i:i+1, :]  # [num_kv_heads, 1, head_dim]
            v_token = value[:, i:i+1, :]
            
            self.block_manager.write_kv(layer_idx, block_id, offset, k_token, v_token)
            
            # 只在第一层更新seq_len（所有层共享）
            if layer_idx == 0:
                state.seq_len += 1
    
    def _allocate_block_for_seq(self, state: SequenceKVState) -> None:
        """为序列分配新block"""
        # 分配一个新block
        block_ids = self.block_manager.allocate_blocks(1, state.seq_id)
        new_block_id = block_ids[0]
        
        # 所有层共用同一个block（简化实现）
        for layer_idx in range(self.num_layers):
            state.block_tables[layer_idx].append(new_block_id)
    
    def get_kv(
        self,
        seq_id: int,
        layer_idx: int,
    ) -> KVPair:
        """
        获取序列指定层的完整K/V
        
        Args:
            seq_id: 序列ID
            layer_idx: 层索引
            
        Returns:
            (K, V) 元组, 每个shape [1, num_kv_heads, seq_len, head_dim]
        """
        if seq_id not in self._seq_states:
            raise ValueError(f"序列 {seq_id} 不存在")
        
        state = self._seq_states[seq_id]
        block_ids = state.block_tables[layer_idx]
        
        return self.block_manager.read_kv(layer_idx, block_ids, state.seq_len)
    
    def get_seq_len(self, seq_id: int) -> int:
        """获取序列当前长度"""
        if seq_id not in self._seq_states:
            return 0
        return self._seq_states[seq_id].seq_len
    
    def update(
        self,
        seq_id: int,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        start_pos: int,
    ) -> KVPair:
        """
        更新K/V并返回完整KV（与DenseKVCache接口兼容）
        
        Args:
            seq_id: 序列ID
            layer_idx: 层索引
            key: 新的K, shape [1, num_kv_heads, num_tokens, head_dim]
            value: 新的V, shape [1, num_kv_heads, num_tokens, head_dim]
            start_pos: 起始位置（用于验证）
            
        Returns:
            更新后的完整KV对
        """
        # 追加新KV
        self.append_kv(seq_id, layer_idx, key, value)
        
        # 返回完整KV
        return self.get_kv(seq_id, layer_idx)
    
    def reset(self) -> None:
        """重置所有序列"""
        for seq_id in list(self._seq_states.keys()):
            self.delete_sequence(seq_id)
        self._next_seq_id = 0


class PagedKVCacheWrapper:
    """
    PagedKVCache的单序列包装器
    
    提供与DenseKVCache相同的接口，用于Engine中的请求级别操作。
    """
    
    def __init__(
        self,
        paged_cache: PagedKVCache,
        seq_id: int,
    ):
        self.paged_cache = paged_cache
        self.seq_id = seq_id
        self._seq_len = 0
    
    @property
    def seq_len(self) -> int:
        return self.paged_cache.get_seq_len(self.seq_id)
    
    def get_layer_kv(self, layer_idx: int, seq_len: Optional[int] = None) -> KVPair:
        """获取指定层的KV"""
        return self.paged_cache.get_kv(self.seq_id, layer_idx)
    
    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        start_pos: int,
    ) -> KVPair:
        """更新并返回完整KV"""
        return self.paged_cache.update(self.seq_id, layer_idx, key, value, start_pos)
