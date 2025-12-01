"""
KV Cache单元测试

测试KV Cache的正确性和prefill/decode的数值一致性。
"""

import pytest
import torch

from my_llm_engine import ModelConfig, EngineConfig, GenerationConfig
from my_llm_engine.models import DecoderOnlyModel, build_decoder_only_model
from my_llm_engine.engine import KVCache, generate, benchmark_generation


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
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
    )


@pytest.fixture
def gqa_config():
    """GQA模型配置"""
    return ModelConfig(
        vocab_size=1000,
        hidden_dim=64,
        num_layers=2,
        num_heads=8,
        num_kv_heads=2,  # GQA: 8个Q头共享2个KV头
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
        max_batch_size=2,
    )


class TestKVCache:
    """KVCache类测试"""
    
    def test_empty_creation(self, tiny_config, engine_config):
        """测试创建空缓存"""
        batch_size = 2
        kv_cache = KVCache.empty(tiny_config, engine_config, batch_size)
        
        assert kv_cache.num_layers == tiny_config.num_layers
        assert kv_cache.batch_size == batch_size
        assert kv_cache.seq_len == 0
        assert len(kv_cache.k_cache) == tiny_config.num_layers
        assert len(kv_cache.v_cache) == tiny_config.num_layers
    
    def test_cache_shape(self, tiny_config, engine_config):
        """测试缓存张量形状"""
        batch_size = 2
        kv_cache = KVCache.empty(tiny_config, engine_config, batch_size)
        
        num_kv_heads = tiny_config.num_kv_heads or tiny_config.num_heads
        head_dim = tiny_config.hidden_dim // tiny_config.num_heads
        expected_shape = (batch_size, num_kv_heads, engine_config.max_seq_len, head_dim)
        
        for layer_idx in range(tiny_config.num_layers):
            assert kv_cache.k_cache[layer_idx].shape == expected_shape
            assert kv_cache.v_cache[layer_idx].shape == expected_shape
    
    def test_update_and_get(self, tiny_config, engine_config):
        """测试缓存更新和获取"""
        batch_size = 2
        kv_cache = KVCache.empty(tiny_config, engine_config, batch_size)
        
        num_kv_heads = tiny_config.num_kv_heads or tiny_config.num_heads
        head_dim = tiny_config.hidden_dim // tiny_config.num_heads
        
        # 模拟prefill: 写入5个token的KV
        seq_len = 5
        k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
        
        k_full, v_full = kv_cache.update(layer_idx=0, key=k, value=v, start_pos=0)
        
        assert k_full.shape == (batch_size, num_kv_heads, seq_len, head_dim)
        assert v_full.shape == (batch_size, num_kv_heads, seq_len, head_dim)
        assert kv_cache.seq_len == seq_len
        
        # 验证内容正确
        assert torch.allclose(k_full, k)
        assert torch.allclose(v_full, v)
        
        # 模拟decode: 追加1个token
        k_new = torch.randn(batch_size, num_kv_heads, 1, head_dim)
        v_new = torch.randn(batch_size, num_kv_heads, 1, head_dim)
        
        k_full, v_full = kv_cache.update(layer_idx=0, key=k_new, value=v_new, start_pos=seq_len)
        
        assert k_full.shape == (batch_size, num_kv_heads, seq_len + 1, head_dim)
        assert kv_cache.seq_len == seq_len + 1
    
    def test_get_layer_kv(self, tiny_config, engine_config):
        """测试获取指定层的KV"""
        batch_size = 1
        kv_cache = KVCache.empty(tiny_config, engine_config, batch_size)
        
        num_kv_heads = tiny_config.num_kv_heads or tiny_config.num_heads
        head_dim = tiny_config.hidden_dim // tiny_config.num_heads
        
        # 写入数据
        k = torch.randn(batch_size, num_kv_heads, 3, head_dim)
        v = torch.randn(batch_size, num_kv_heads, 3, head_dim)
        kv_cache.update(layer_idx=0, key=k, value=v, start_pos=0)
        
        # 获取
        k_ret, v_ret = kv_cache.get_layer_kv(layer_idx=0)
        assert k_ret.shape == (batch_size, num_kv_heads, 3, head_dim)
        assert torch.allclose(k_ret, k)
    
    def test_reset(self, tiny_config, engine_config):
        """测试重置缓存"""
        kv_cache = KVCache.empty(tiny_config, engine_config, batch_size=1)
        
        # 写入一些数据
        num_kv_heads = tiny_config.num_kv_heads or tiny_config.num_heads
        head_dim = tiny_config.hidden_dim // tiny_config.num_heads
        k = torch.randn(1, num_kv_heads, 5, head_dim)
        v = torch.randn(1, num_kv_heads, 5, head_dim)
        kv_cache.update(0, k, v, 0)
        
        assert kv_cache.seq_len == 5
        
        # 重置
        kv_cache.reset()
        assert kv_cache.seq_len == 0


class TestPrefillDecodeConsistency:
    """Prefill/Decode数值一致性测试"""
    
    def test_logits_consistency(self, tiny_config, engine_config):
        """测试有无KV Cache时logits一致性"""
        model = build_decoder_only_model(tiny_config, engine_config)
        batch_size = 1
        prompt_len = 5
        
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, prompt_len))
        
        # 方式1: 不使用KV Cache（forward）
        with torch.no_grad():
            logits_no_cache = model(input_ids)
        
        # 方式2: 使用KV Cache（prefill）
        kv_cache = KVCache.empty(tiny_config, engine_config, batch_size)
        with torch.no_grad():
            logits_with_cache = model.prefill(input_ids, kv_cache)
        
        # 验证logits一致
        assert logits_no_cache.shape == logits_with_cache.shape
        assert torch.allclose(logits_no_cache, logits_with_cache, atol=1e-5)
    
    def test_decode_consistency(self, tiny_config, engine_config):
        """测试decode阶段的一致性"""
        model = build_decoder_only_model(tiny_config, engine_config)
        batch_size = 1
        prompt_len = 5
        
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, prompt_len))
        new_token = torch.randint(0, tiny_config.vocab_size, (batch_size, 1))
        
        # 方式1: 完整序列一次forward
        full_ids = torch.cat([input_ids, new_token], dim=1)
        with torch.no_grad():
            logits_full = model(full_ids)
            last_logits_full = logits_full[:, -1, :]  # 最后一个token的logits
        
        # 方式2: prefill + decode
        kv_cache = KVCache.empty(tiny_config, engine_config, batch_size)
        with torch.no_grad():
            _ = model.prefill(input_ids, kv_cache)
            logits_decode = model.decode(new_token, kv_cache, past_seq_len=prompt_len)
            last_logits_decode = logits_decode[:, -1, :]
        
        # 验证一致性
        assert torch.allclose(last_logits_full, last_logits_decode, atol=1e-5)
    
    def test_multi_step_decode(self, tiny_config, engine_config):
        """测试多步decode的一致性"""
        model = build_decoder_only_model(tiny_config, engine_config)
        batch_size = 1
        prompt_len = 3
        num_decode_steps = 5
        
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, prompt_len))
        
        # 预生成要decode的token
        decode_tokens = torch.randint(
            0, tiny_config.vocab_size, (batch_size, num_decode_steps)
        )
        
        # 方式1: 完整序列一次forward
        full_ids = torch.cat([input_ids, decode_tokens], dim=1)
        with torch.no_grad():
            logits_full = model(full_ids)
        
        # 方式2: prefill + 多步decode
        kv_cache = KVCache.empty(tiny_config, engine_config, batch_size)
        logits_list = []
        
        with torch.no_grad():
            logits_prefill = model.prefill(input_ids, kv_cache)
            logits_list.append(logits_prefill)
            
            for i in range(num_decode_steps):
                token = decode_tokens[:, i:i+1]
                logits_decode = model.decode(
                    token, kv_cache, past_seq_len=prompt_len + i
                )
                logits_list.append(logits_decode)
        
        # 拼接所有logits
        logits_incremental = torch.cat(logits_list, dim=1)
        
        # 验证一致性
        assert logits_full.shape == logits_incremental.shape
        assert torch.allclose(logits_full, logits_incremental, atol=1e-5)
    
    def test_gqa_consistency(self, gqa_config, engine_config):
        """测试GQA模型的一致性"""
        model = build_decoder_only_model(gqa_config, engine_config)
        batch_size = 1
        prompt_len = 4
        
        input_ids = torch.randint(0, gqa_config.vocab_size, (batch_size, prompt_len))
        new_token = torch.randint(0, gqa_config.vocab_size, (batch_size, 1))
        
        # 完整forward
        full_ids = torch.cat([input_ids, new_token], dim=1)
        with torch.no_grad():
            logits_full = model(full_ids)[:, -1, :]
        
        # prefill + decode
        kv_cache = KVCache.empty(gqa_config, engine_config, batch_size)
        with torch.no_grad():
            _ = model.prefill(input_ids, kv_cache)
            logits_decode = model.decode(new_token, kv_cache, past_seq_len=prompt_len)[:, -1, :]
        
        assert torch.allclose(logits_full, logits_decode, atol=1e-5)


class TestGeneration:
    """生成功能测试"""
    
    def test_generate_with_kv_cache(self, tiny_config, engine_config):
        """测试使用KV Cache的生成"""
        model = build_decoder_only_model(tiny_config, engine_config)
        gen_config = GenerationConfig(max_new_tokens=5, do_sample=False)
        
        prompt = torch.randint(0, tiny_config.vocab_size, (1, 3))
        
        result = generate(
            model, prompt, tiny_config, engine_config, gen_config,
            use_kv_cache=True
        )
        
        assert result.shape[1] == 3 + 5  # prompt + generated
    
    def test_generate_without_kv_cache(self, tiny_config, engine_config):
        """测试不使用KV Cache的生成"""
        model = build_decoder_only_model(tiny_config, engine_config)
        gen_config = GenerationConfig(max_new_tokens=5, do_sample=False)
        
        prompt = torch.randint(0, tiny_config.vocab_size, (1, 3))
        
        result = generate(
            model, prompt, tiny_config, engine_config, gen_config,
            use_kv_cache=False
        )
        
        assert result.shape[1] == 3 + 5
    
    def test_generate_results_match(self, tiny_config, engine_config):
        """测试有无KV Cache生成结果一致（greedy）"""
        model = build_decoder_only_model(tiny_config, engine_config)
        gen_config = GenerationConfig(max_new_tokens=10, do_sample=False)
        
        prompt = torch.randint(0, tiny_config.vocab_size, (1, 5))
        
        result_with_cache = generate(
            model, prompt.clone(), tiny_config, engine_config, gen_config,
            use_kv_cache=True
        )
        
        result_without_cache = generate(
            model, prompt.clone(), tiny_config, engine_config, gen_config,
            use_kv_cache=False
        )
        
        # Greedy解码应该产生完全相同的结果
        assert torch.equal(result_with_cache, result_without_cache)
    
    def test_batch_generation(self, tiny_config, engine_config):
        """测试批量生成"""
        model = build_decoder_only_model(tiny_config, engine_config)
        gen_config = GenerationConfig(max_new_tokens=5, do_sample=False)
        
        batch_size = 2
        prompt = torch.randint(0, tiny_config.vocab_size, (batch_size, 4))
        
        result = generate(
            model, prompt, tiny_config, engine_config, gen_config,
            use_kv_cache=True
        )
        
        assert result.shape == (batch_size, 4 + 5)


class TestPerformance:
    """性能测试"""
    
    def test_kv_cache_faster(self, tiny_config, engine_config):
        """验证KV Cache版本更快（可能在小模型上不明显）"""
        model = build_decoder_only_model(tiny_config, engine_config)
        gen_config = GenerationConfig(max_new_tokens=10, do_sample=False)
        
        # 使用较长的prompt来体现KV Cache的优势
        prompt = torch.randint(0, tiny_config.vocab_size, (1, 20))
        
        result = benchmark_generation(
            model, prompt, tiny_config, engine_config, gen_config
        )
        
        # 验证结果一致
        assert torch.equal(result["result_with_cache"], result["result_no_cache"])
        
        # 打印性能数据（不强制断言加速比，因为小模型可能差异不大）
        print(f"\n无KV Cache耗时: {result['time_no_cache']:.4f}s")
        print(f"有KV Cache耗时: {result['time_with_cache']:.4f}s")
        print(f"加速比: {result['speedup']:.2f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
