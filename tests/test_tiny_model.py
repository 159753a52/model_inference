"""
Tiny模型基础单元测试

测试模型组件的shape和基本功能。
"""

import pytest
import torch

from my_llm_engine import ModelConfig, EngineConfig, GenerationConfig
from my_llm_engine.models import (
    RMSNorm,
    MLP,
    RotaryEmbedding,
    SelfAttention,
    DecoderLayer,
    DecoderOnlyModel,
    build_decoder_only_model,
)
from my_llm_engine.engine import generate, sample_next_token


# 测试用的小配置
@pytest.fixture
def tiny_config():
    return ModelConfig(
        vocab_size=1000,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        intermediate_dim=128,
        max_position_embeddings=64,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
    )


@pytest.fixture
def engine_config():
    return EngineConfig(
        device="cpu",
        dtype="float32",
        max_seq_len=32,
        max_batch_size=2,
    )


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(hidden_dim=64)
        x = torch.randn(2, 10, 64)
        y = norm(x)
        assert y.shape == x.shape
    
    def test_normalization(self):
        norm = RMSNorm(hidden_dim=64)
        x = torch.randn(2, 10, 64)
        y = norm(x)
        # 输出应该有合理的范围
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()


class TestMLP:
    def test_output_shape(self):
        mlp = MLP(hidden_dim=64, intermediate_dim=128)
        x = torch.randn(2, 10, 64)
        y = mlp(x)
        assert y.shape == x.shape


class TestRotaryEmbedding:
    def test_output_shape(self):
        rope = RotaryEmbedding(head_dim=16, max_position_embeddings=64)
        q = torch.randn(2, 4, 10, 16)  # [B, heads, S, head_dim]
        k = torch.randn(2, 4, 10, 16)
        pos_ids = torch.arange(10).unsqueeze(0).expand(2, -1)
        
        q_rot, k_rot = rope(q, k, pos_ids)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


class TestSelfAttention:
    def test_output_shape(self, tiny_config):
        attn = SelfAttention(tiny_config)
        x = torch.randn(2, 10, tiny_config.hidden_dim)
        y = attn(x)
        assert y.shape == x.shape
    
    def test_gqa(self):
        gqa_config = ModelConfig(
            vocab_size=1000,
            hidden_dim=64,
            num_layers=1,
            num_heads=8,
            num_kv_heads=2,  # GQA
            intermediate_dim=128,
        )
        attn = SelfAttention(gqa_config)
        x = torch.randn(2, 10, 64)
        y = attn(x)
        assert y.shape == x.shape


class TestDecoderLayer:
    def test_output_shape(self, tiny_config):
        layer = DecoderLayer(tiny_config)
        x = torch.randn(2, 10, tiny_config.hidden_dim)
        y = layer(x)
        assert y.shape == x.shape


class TestDecoderOnlyModel:
    def test_forward_shape(self, tiny_config, engine_config):
        model = build_decoder_only_model(tiny_config, engine_config)
        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 10))
        
        logits = model(input_ids)
        
        assert logits.shape == (2, 10, tiny_config.vocab_size)
    
    def test_causal_mask(self, tiny_config, engine_config):
        model = build_decoder_only_model(tiny_config, engine_config)
        
        # 测试不同输入产生不同输出
        input1 = torch.randint(0, tiny_config.vocab_size, (1, 5))
        input2 = torch.randint(0, tiny_config.vocab_size, (1, 5))
        
        with torch.no_grad():
            out1 = model(input1)
            out2 = model(input2)
        
        # 不同输入应该产生不同输出
        assert not torch.allclose(out1, out2)


class TestGeneration:
    def test_generate_length(self, tiny_config, engine_config):
        model = build_decoder_only_model(tiny_config, engine_config)
        gen_config = GenerationConfig(max_new_tokens=5, do_sample=False)
        
        prompt = torch.randint(0, tiny_config.vocab_size, (1, 3))
        
        output = generate(
            model=model,
            input_ids=prompt,
            model_config=tiny_config,
            engine_config=engine_config,
            gen_config=gen_config,
        )
        
        # 输出长度应该是 prompt + max_new_tokens
        assert output.shape[1] == 3 + 5
    
    def test_sample_next_token_greedy(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 0.5]])  # 最大值在索引2
        token = sample_next_token(logits, do_sample=False)
        assert token.item() == 2
    
    def test_sample_next_token_with_temperature(self):
        logits = torch.randn(1, 100)
        # 低温度应该更确定性
        token1 = sample_next_token(logits, temperature=0.1, do_sample=True)
        # 应该返回有效的token id
        assert 0 <= token1.item() < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
