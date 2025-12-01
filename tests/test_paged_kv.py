"""
Paged KV Cache单元测试

测试BlockManager和PagedKVCache的正确性和与DenseKVCache的一致性。
"""

import pytest
import torch

from my_llm_engine import ModelConfig, EngineConfig, GenerationConfig
from my_llm_engine.models import build_decoder_only_model
from my_llm_engine.engine import KVCache
from my_llm_engine.engine.block_manager import BlockManager
from my_llm_engine.engine.paged_kv_cache import PagedKVCache, PagedKVCacheWrapper
from my_llm_engine.engine.profiler import SimpleProfiler, create_profiler


@pytest.fixture
def tiny_config():
    """小型模型配置"""
    return ModelConfig(
        vocab_size=1000,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        intermediate_dim=128,
        max_position_embeddings=128,
    )


@pytest.fixture
def engine_config():
    """引擎配置"""
    return EngineConfig(
        device="cpu",
        dtype="float32",
        max_seq_len=64,
        max_batch_size=4,
    )


class TestBlockManager:
    """BlockManager测试"""
    
    def test_creation(self, tiny_config, engine_config):
        """测试创建BlockManager"""
        bm = BlockManager.from_config(tiny_config, engine_config, block_size=8)
        
        assert bm.num_layers == tiny_config.num_layers
        assert bm.block_size == 8
        assert bm.num_free_blocks() == bm.num_blocks
    
    def test_allocate_and_free(self, tiny_config, engine_config):
        """测试分配和释放block"""
        bm = BlockManager.from_config(tiny_config, engine_config, block_size=8, num_blocks=10)
        
        initial_free = bm.num_free_blocks()
        
        # 分配3个block
        blocks = bm.allocate_blocks(3)
        assert len(blocks) == 3
        assert bm.num_free_blocks() == initial_free - 3
        
        # 释放block
        bm.free_blocks(blocks)
        assert bm.num_free_blocks() == initial_free
    
    def test_allocate_insufficient(self, tiny_config, engine_config):
        """测试分配不足"""
        bm = BlockManager.from_config(tiny_config, engine_config, block_size=8, num_blocks=5)
        
        with pytest.raises(RuntimeError):
            bm.allocate_blocks(10)
    
    def test_write_and_read_kv(self, tiny_config, engine_config):
        """测试写入和读取KV"""
        bm = BlockManager.from_config(tiny_config, engine_config, block_size=4, num_blocks=10)
        
        num_kv_heads = tiny_config.num_kv_heads
        head_dim = tiny_config.hidden_dim // tiny_config.num_heads
        
        # 分配block
        block_id = bm.allocate_blocks(1)[0]
        
        # 写入2个token
        k = torch.randn(num_kv_heads, 2, head_dim)
        v = torch.randn(num_kv_heads, 2, head_dim)
        
        bm.write_kv(layer_idx=0, block_id=block_id, offset=0, key=k, value=v)
        
        # 读取
        k_read, v_read = bm.read_kv(layer_idx=0, block_ids=[block_id], seq_len=2)
        
        assert k_read.shape == (1, num_kv_heads, 2, head_dim)
        assert torch.allclose(k_read.squeeze(0), k)
        assert torch.allclose(v_read.squeeze(0), v)
    
    def test_reset(self, tiny_config, engine_config):
        """测试重置"""
        bm = BlockManager.from_config(tiny_config, engine_config, block_size=8, num_blocks=10)
        
        bm.allocate_blocks(5)
        assert bm.num_free_blocks() == 5
        
        bm.reset()
        assert bm.num_free_blocks() == 10


class TestPagedKVCache:
    """PagedKVCache测试"""
    
    def test_create_sequence(self, tiny_config, engine_config):
        """测试创建序列"""
        bm = BlockManager.from_config(tiny_config, engine_config, block_size=4)
        cache = PagedKVCache(bm)
        
        seq_id = cache.create_sequence()
        assert seq_id == 0
        assert cache.get_seq_len(seq_id) == 0
    
    def test_append_and_get_kv(self, tiny_config, engine_config):
        """测试追加和获取KV"""
        bm = BlockManager.from_config(tiny_config, engine_config, block_size=4)
        cache = PagedKVCache(bm)
        
        num_kv_heads = tiny_config.num_kv_heads
        head_dim = tiny_config.hidden_dim // tiny_config.num_heads
        
        seq_id = cache.create_sequence()
        
        # 追加3个token的KV
        k = torch.randn(1, num_kv_heads, 3, head_dim)
        v = torch.randn(1, num_kv_heads, 3, head_dim)
        
        cache.append_kv(seq_id, layer_idx=0, key=k, value=v)
        
        # 获取
        k_out, v_out = cache.get_kv(seq_id, layer_idx=0)
        
        assert k_out.shape == (1, num_kv_heads, 3, head_dim)
        assert cache.get_seq_len(seq_id) == 3
    
    def test_append_across_blocks(self, tiny_config, engine_config):
        """测试跨block追加"""
        bm = BlockManager.from_config(tiny_config, engine_config, block_size=4)
        cache = PagedKVCache(bm)
        
        num_kv_heads = tiny_config.num_kv_heads
        head_dim = tiny_config.hidden_dim // tiny_config.num_heads
        
        seq_id = cache.create_sequence()
        
        # 追加6个token（需要2个block）
        k = torch.randn(1, num_kv_heads, 6, head_dim)
        v = torch.randn(1, num_kv_heads, 6, head_dim)
        
        cache.append_kv(seq_id, layer_idx=0, key=k, value=v)
        
        assert cache.get_seq_len(seq_id) == 6
        
        k_out, v_out = cache.get_kv(seq_id, layer_idx=0)
        assert k_out.shape == (1, num_kv_heads, 6, head_dim)
    
    def test_delete_sequence(self, tiny_config, engine_config):
        """测试删除序列"""
        bm = BlockManager.from_config(tiny_config, engine_config, block_size=4, num_blocks=10)
        cache = PagedKVCache(bm)
        
        initial_free = bm.num_free_blocks()
        
        seq_id = cache.create_sequence()
        
        # 追加一些token
        num_kv_heads = tiny_config.num_kv_heads
        head_dim = tiny_config.hidden_dim // tiny_config.num_heads
        k = torch.randn(1, num_kv_heads, 5, head_dim)
        v = torch.randn(1, num_kv_heads, 5, head_dim)
        cache.append_kv(seq_id, layer_idx=0, key=k, value=v)
        
        # 删除序列
        cache.delete_sequence(seq_id)
        
        # block应该被释放
        assert bm.num_free_blocks() == initial_free
    
    def test_wrapper(self, tiny_config, engine_config):
        """测试PagedKVCacheWrapper"""
        bm = BlockManager.from_config(tiny_config, engine_config, block_size=4)
        cache = PagedKVCache(bm)
        
        seq_id = cache.create_sequence()
        wrapper = PagedKVCacheWrapper(cache, seq_id)
        
        num_kv_heads = tiny_config.num_kv_heads
        head_dim = tiny_config.hidden_dim // tiny_config.num_heads
        
        # 使用update接口
        k = torch.randn(1, num_kv_heads, 3, head_dim)
        v = torch.randn(1, num_kv_heads, 3, head_dim)
        
        k_out, v_out = wrapper.update(layer_idx=0, key=k, value=v, start_pos=0)
        
        assert wrapper.seq_len == 3


class TestDenseVsPagedConsistency:
    """Dense和Paged KV一致性测试"""
    
    def test_kv_content_match(self, tiny_config, engine_config):
        """测试KV内容一致"""
        num_kv_heads = tiny_config.num_kv_heads
        head_dim = tiny_config.hidden_dim // tiny_config.num_heads
        
        # Dense KV
        dense_cache = KVCache.empty(tiny_config, engine_config, batch_size=1)
        
        # Paged KV
        bm = BlockManager.from_config(tiny_config, engine_config, block_size=4)
        paged_cache = PagedKVCache(bm)
        seq_id = paged_cache.create_sequence()
        
        # 写入相同的数据
        k = torch.randn(1, num_kv_heads, 5, head_dim)
        v = torch.randn(1, num_kv_heads, 5, head_dim)
        
        # Dense
        dense_cache.update(layer_idx=0, key=k, value=v, start_pos=0)
        
        # Paged
        paged_cache.append_kv(seq_id, layer_idx=0, key=k, value=v)
        
        # 比较
        k_dense, v_dense = dense_cache.get_layer_kv(layer_idx=0, seq_len=5)
        k_paged, v_paged = paged_cache.get_kv(seq_id, layer_idx=0)
        
        assert torch.allclose(k_dense, k_paged, atol=1e-6)
        assert torch.allclose(v_dense, v_paged, atol=1e-6)
    
    def test_incremental_append(self, tiny_config, engine_config):
        """测试增量追加一致性"""
        num_kv_heads = tiny_config.num_kv_heads
        head_dim = tiny_config.hidden_dim // tiny_config.num_heads
        
        # Dense
        dense_cache = KVCache.empty(tiny_config, engine_config, batch_size=1)
        
        # Paged
        bm = BlockManager.from_config(tiny_config, engine_config, block_size=4)
        paged_cache = PagedKVCache(bm)
        seq_id = paged_cache.create_sequence()
        
        # 增量写入
        for i in range(10):
            k = torch.randn(1, num_kv_heads, 1, head_dim)
            v = torch.randn(1, num_kv_heads, 1, head_dim)
            
            dense_cache.update(layer_idx=0, key=k, value=v, start_pos=i)
            paged_cache.append_kv(seq_id, layer_idx=0, key=k, value=v)
        
        # 比较
        k_dense, v_dense = dense_cache.get_layer_kv(layer_idx=0, seq_len=10)
        k_paged, v_paged = paged_cache.get_kv(seq_id, layer_idx=0)
        
        assert torch.allclose(k_dense, k_paged, atol=1e-6)


class TestProfiler:
    """Profiler测试"""
    
    def test_basic_timing(self):
        """测试基本计时"""
        profiler = SimpleProfiler()
        
        with profiler.record("test"):
            # 模拟一些工作
            _ = sum(range(1000))
        
        stats = profiler.get_stats("test")
        assert stats is not None
        assert stats.count == 1
        assert stats.total_time > 0
    
    def test_multiple_records(self):
        """测试多次记录"""
        profiler = SimpleProfiler()
        
        for _ in range(5):
            with profiler.record("loop"):
                _ = sum(range(100))
        
        stats = profiler.get_stats("loop")
        assert stats.count == 5
    
    def test_counters(self):
        """测试计数器"""
        profiler = SimpleProfiler()
        
        profiler.increment("tokens", 10)
        profiler.increment("tokens", 5)
        
        summary = profiler.summary()
        assert summary["counters"]["tokens"] == 15
    
    def test_disabled_profiler(self):
        """测试禁用的profiler"""
        profiler = create_profiler(enabled=False)
        
        with profiler.record("test"):
            pass
        
        # 不应该有统计
        summary = profiler.summary()
        assert summary == {}
    
    def test_reset(self):
        """测试重置"""
        profiler = SimpleProfiler()
        
        with profiler.record("test"):
            pass
        
        profiler.reset()
        assert profiler.get_stats("test") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
