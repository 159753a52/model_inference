"""
Block管理器模块

管理全局KV缓存的block池，支持分页式KV存储。
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import torch

from my_llm_engine.config import ModelConfig, EngineConfig
from my_llm_engine.logging_utils import get_logger


class BlockManager:
    """
    全局KV Block池管理器
    
    预分配固定数量的block，用于存储所有序列的K/V缓存。
    每个block存储固定数量的token（block_size）。
    
    存储形状:
        k_storage: [num_layers, num_blocks, num_kv_heads, block_size, head_dim]
        v_storage: [num_layers, num_blocks, num_kv_heads, block_size, head_dim]
    """
    
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: int = 256,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        """
        Args:
            num_layers: Transformer层数
            num_kv_heads: KV头数量
            head_dim: 每个头的维度
            block_size: 每个block存储的token数
            num_blocks: block总数
            device: 设备
            dtype: 数据类型
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.device = device or torch.device("cpu")
        self.dtype = dtype or torch.float32
        
        self._logger = get_logger(__name__)
        
        # 预分配全局存储
        storage_shape = (num_layers, num_blocks, num_kv_heads, block_size, head_dim)
        self.k_storage = torch.zeros(storage_shape, device=self.device, dtype=self.dtype)
        self.v_storage = torch.zeros(storage_shape, device=self.device, dtype=self.dtype)
        
        # 空闲block列表（使用set以O(1)查找）
        self._free_blocks: set = set(range(num_blocks))
        
        # 已分配block映射：block_id -> seq_id（可选，用于调试）
        self._allocated: dict = {}
        
        self._logger.debug(f"BlockManager初始化: {num_blocks} blocks, "
                          f"block_size={block_size}, 总容量={self.max_tokens_supported()} tokens")
    
    @classmethod
    def from_config(
        cls,
        model_config: ModelConfig,
        engine_config: EngineConfig,
        block_size: int = 16,
        num_blocks: Optional[int] = None,
    ) -> "BlockManager":
        """
        从配置创建BlockManager
        
        Args:
            model_config: 模型配置
            engine_config: 引擎配置
            block_size: block大小
            num_blocks: block数量，None则自动计算
        """
        num_kv_heads = model_config.num_kv_heads or model_config.num_heads
        head_dim = model_config.hidden_dim // model_config.num_heads
        
        if num_blocks is None:
            # 根据max_seq_len和max_batch_size估算需要的block数
            tokens_needed = engine_config.max_seq_len * engine_config.max_batch_size
            num_blocks = (tokens_needed + block_size - 1) // block_size
            # 额外预留一些空间
            num_blocks = int(num_blocks * 1.2)
        
        return cls(
            num_layers=model_config.num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            num_blocks=num_blocks,
            device=engine_config.torch_device,
            dtype=engine_config.torch_dtype,
        )
    
    def allocate_blocks(self, num_blocks_needed: int, seq_id: int = -1) -> List[int]:
        """
        分配指定数量的block
        
        Args:
            num_blocks_needed: 需要的block数量
            seq_id: 序列ID（用于调试追踪）
            
        Returns:
            分配的block_id列表
            
        Raises:
            RuntimeError: 空闲block不足
        """
        if num_blocks_needed > len(self._free_blocks):
            raise RuntimeError(
                f"Block不足: 需要{num_blocks_needed}, 可用{len(self._free_blocks)}"
            )
        
        allocated = []
        for _ in range(num_blocks_needed):
            block_id = self._free_blocks.pop()
            allocated.append(block_id)
            self._allocated[block_id] = seq_id
        
        return allocated
    
    def free_blocks(self, block_ids: List[int]) -> None:
        """
        释放指定的block
        
        Args:
            block_ids: 要释放的block_id列表
        """
        for block_id in block_ids:
            if block_id in self._allocated:
                del self._allocated[block_id]
            self._free_blocks.add(block_id)
    
    def write_kv(
        self,
        layer_idx: int,
        block_id: int,
        offset: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """
        向指定block写入K/V
        
        Args:
            layer_idx: 层索引
            block_id: block ID
            offset: block内的起始偏移
            key: K张量, shape [num_kv_heads, num_tokens, head_dim]
            value: V张量, shape [num_kv_heads, num_tokens, head_dim]
        """
        num_tokens = key.shape[1]
        end_offset = offset + num_tokens
        
        # 写入存储
        # storage shape: [num_layers, num_blocks, num_kv_heads, block_size, head_dim]
        # key shape: [num_kv_heads, num_tokens, head_dim]
        self.k_storage[layer_idx, block_id, :, offset:end_offset, :] = key
        self.v_storage[layer_idx, block_id, :, offset:end_offset, :] = value
    
    def read_kv(
        self,
        layer_idx: int,
        block_ids: List[int],
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从多个block读取K/V
        
        Args:
            layer_idx: 层索引
            block_ids: block ID列表
            seq_len: 实际序列长度
            
        Returns:
            (K, V) 元组, 每个shape [1, num_kv_heads, seq_len, head_dim]
        """
        if not block_ids:
            return (
                torch.zeros(1, self.num_kv_heads, 0, self.head_dim, 
                           device=self.device, dtype=self.dtype),
                torch.zeros(1, self.num_kv_heads, 0, self.head_dim,
                           device=self.device, dtype=self.dtype),
            )
        
        # 收集所有block的数据
        k_parts = []
        v_parts = []
        tokens_remaining = seq_len
        
        for block_id in block_ids:
            tokens_in_block = min(tokens_remaining, self.block_size)
            
            # 读取 [num_kv_heads, tokens_in_block, head_dim]
            k_part = self.k_storage[layer_idx, block_id, :, :tokens_in_block, :]
            v_part = self.v_storage[layer_idx, block_id, :, :tokens_in_block, :]
            
            k_parts.append(k_part)
            v_parts.append(v_part)
            
            tokens_remaining -= tokens_in_block
            if tokens_remaining <= 0:
                break
        
        # 拼接 [num_kv_heads, seq_len, head_dim]
        k = torch.cat(k_parts, dim=1)
        v = torch.cat(v_parts, dim=1)
        
        # 添加batch维度 [1, num_kv_heads, seq_len, head_dim]
        return k.unsqueeze(0), v.unsqueeze(0)
    
    def num_free_blocks(self) -> int:
        """当前空闲block数量"""
        return len(self._free_blocks)
    
    def num_used_blocks(self) -> int:
        """当前已使用block数量"""
        return self.num_blocks - len(self._free_blocks)
    
    def max_tokens_supported(self) -> int:
        """最大可支持的token数"""
        return self.num_blocks * self.block_size
    
    def get_memory_usage(self) -> int:
        """获取存储占用的内存（字节）"""
        element_size = self.k_storage.element_size()
        total_elements = 2 * self.k_storage.numel()  # K和V
        return total_elements * element_size
    
    def reset(self) -> None:
        """重置所有block为空闲状态"""
        self._free_blocks = set(range(self.num_blocks))
        self._allocated.clear()
        self.k_storage.zero_()
        self.v_storage.zero_()
