"""
Engine和Scheduler单元测试

测试多请求场景下的行为和数值一致性。
"""

import pytest
import torch

from my_llm_engine import ModelConfig, EngineConfig, GenerationConfig
from my_llm_engine.models import build_decoder_only_model
from my_llm_engine.engine import (
    GenerationRequest,
    RequestStatus,
    Scheduler,
    LLMEngine,
    generate,
)


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


@pytest.fixture
def model(tiny_config, engine_config):
    """构建模型"""
    return build_decoder_only_model(tiny_config, engine_config)


class TestGenerationRequest:
    """GenerationRequest测试"""
    
    def test_create_request(self):
        """测试创建请求"""
        prompt = torch.randint(0, 1000, (10,))
        req = GenerationRequest.create(prompt)
        
        assert req.prompt_len == 10
        assert req.num_generated == 0
        assert req.status == RequestStatus.WAITING
    
    def test_add_token(self):
        """测试添加token"""
        prompt = torch.randint(0, 1000, (5,))
        req = GenerationRequest.create(prompt)
        req.past_seq_len = 5
        
        req.add_token(100)
        assert req.num_generated == 1
        assert req.output_ids == [100]
        # past_seq_len由外部管理，add_token不修改它
        assert req.past_seq_len == 5
    
    def test_eos_detection(self):
        """测试EOS检测"""
        prompt = torch.randint(0, 1000, (5,))
        req = GenerationRequest.create(prompt, eos_token_id=999)
        req.past_seq_len = 5
        req.mark_running()
        
        req.add_token(100)
        assert req.status == RequestStatus.RUNNING
        
        req.add_token(999)  # EOS
        assert req.status == RequestStatus.FINISHED
    
    def test_max_tokens_limit(self):
        """测试最大token限制"""
        prompt = torch.randint(0, 1000, (5,))
        gen_config = GenerationConfig(max_new_tokens=3)
        req = GenerationRequest.create(prompt, gen_config=gen_config)
        req.past_seq_len = 5
        req.mark_running()
        
        req.add_token(1)
        req.add_token(2)
        assert req.can_continue
        
        req.add_token(3)
        assert req.status == RequestStatus.FINISHED
    
    def test_get_full_sequence(self):
        """测试获取完整序列"""
        prompt = torch.tensor([1, 2, 3])
        req = GenerationRequest.create(prompt)
        req.output_ids = [4, 5]
        
        full = req.get_full_sequence()
        assert full.tolist() == [1, 2, 3, 4, 5]


class TestScheduler:
    """Scheduler测试"""
    
    def test_add_request(self):
        """测试添加请求"""
        scheduler = Scheduler(max_batch_size=4, max_seq_len=64)
        prompt = torch.randint(0, 1000, (10,))
        req = GenerationRequest.create(prompt)
        
        scheduler.add_request(req)
        
        assert scheduler.get_num_waiting() == 1
        assert scheduler.has_unfinished_requests()
    
    def test_schedule_prefill_batch(self):
        """测试prefill调度"""
        scheduler = Scheduler(max_batch_size=4, max_seq_len=64)
        
        for i in range(3):
            prompt = torch.randint(0, 1000, (5 + i,))
            req = GenerationRequest.create(prompt, request_id=f"req_{i}")
            scheduler.add_request(req)
        
        batch = scheduler.schedule_prefill_batch()
        assert len(batch) == 3
    
    def test_schedule_decode_batch(self):
        """测试decode调度"""
        scheduler = Scheduler(max_batch_size=4, max_seq_len=64)
        
        for i in range(2):
            prompt = torch.randint(0, 1000, (5,))
            req = GenerationRequest.create(prompt, request_id=f"req_{i}")
            req.mark_running()
            scheduler._running[req.request_id] = req
        
        batch = scheduler.schedule_decode_batch()
        assert len(batch) == 2
    
    def test_mark_prefill_done(self):
        """测试prefill完成标记"""
        scheduler = Scheduler(max_batch_size=4, max_seq_len=64)
        
        prompt = torch.randint(0, 1000, (5,))
        req = GenerationRequest.create(prompt)
        scheduler.add_request(req)
        
        batch = scheduler.schedule_prefill_batch()
        scheduler.mark_prefill_done(batch)
        
        assert scheduler.get_num_waiting() == 0
        assert scheduler.get_num_running() == 1
    
    def test_prompt_too_long(self):
        """测试prompt过长"""
        scheduler = Scheduler(max_batch_size=4, max_seq_len=10)
        
        prompt = torch.randint(0, 1000, (20,))  # 超过max_seq_len
        req = GenerationRequest.create(prompt)
        scheduler.add_request(req)
        
        assert scheduler.get_num_waiting() == 0
        assert scheduler.get_num_finished() == 1
        assert req.status == RequestStatus.ERROR


class TestLLMEngine:
    """LLMEngine测试"""
    
    def test_add_request(self, model, tiny_config, engine_config):
        """测试添加请求"""
        engine = LLMEngine(model, tiny_config, engine_config)
        
        prompt = torch.randint(0, tiny_config.vocab_size, (5,))
        req_id = engine.add_request(prompt)
        
        assert req_id is not None
        response = engine.get_response(req_id)
        assert response is not None
        assert response.prompt_len == 5
    
    def test_single_request_generation(self, model, tiny_config, engine_config):
        """测试单请求生成"""
        engine = LLMEngine(model, tiny_config, engine_config)
        
        prompt = torch.randint(0, tiny_config.vocab_size, (5,))
        gen_config = GenerationConfig(max_new_tokens=10, do_sample=False)
        
        req_id = engine.add_request(prompt, gen_config)
        engine.run_until_complete()
        
        response = engine.get_response(req_id)
        assert response.status == RequestStatus.FINISHED
        assert response.num_generated == 10
        assert response.current_len == 15
    
    def test_multi_request_generation(self, model, tiny_config, engine_config):
        """测试多请求生成"""
        engine = LLMEngine(model, tiny_config, engine_config)
        gen_config = GenerationConfig(max_new_tokens=5, do_sample=False)
        
        prompts = [
            torch.randint(0, tiny_config.vocab_size, (3,)),
            torch.randint(0, tiny_config.vocab_size, (5,)),
            torch.randint(0, tiny_config.vocab_size, (4,)),
        ]
        
        req_ids = []
        for prompt in prompts:
            req_id = engine.add_request(prompt, gen_config)
            req_ids.append(req_id)
        
        engine.run_until_complete()
        
        for req_id in req_ids:
            response = engine.get_response(req_id)
            assert response.status == RequestStatus.FINISHED
            assert response.num_generated == 5
    
    def test_generate_batch(self, model, tiny_config, engine_config):
        """测试batch生成API"""
        engine = LLMEngine(model, tiny_config, engine_config)
        
        prompts = [
            torch.randint(0, tiny_config.vocab_size, (4,)),
            torch.randint(0, tiny_config.vocab_size, (6,)),
        ]
        
        gen_config = GenerationConfig(max_new_tokens=8, do_sample=False)
        results = engine.generate_batch(prompts, gen_config)
        
        assert len(results) == 2
        assert len(results[0]) == 4 + 8
        assert len(results[1]) == 6 + 8
    
    def test_dynamic_request_addition(self, model, tiny_config, engine_config):
        """测试动态添加请求"""
        engine = LLMEngine(model, tiny_config, engine_config)
        gen_config = GenerationConfig(max_new_tokens=15, do_sample=False)
        
        # 添加第一个请求
        prompt1 = torch.randint(0, tiny_config.vocab_size, (5,))
        req_id1 = engine.add_request(prompt1, gen_config)
        
        # 执行几步
        for _ in range(5):
            engine.step()
        
        # 动态添加第二个请求
        prompt2 = torch.randint(0, tiny_config.vocab_size, (6,))
        req_id2 = engine.add_request(prompt2, gen_config)
        
        # 运行到完成
        engine.run_until_complete()
        
        response1 = engine.get_response(req_id1)
        response2 = engine.get_response(req_id2)
        
        assert response1.status == RequestStatus.FINISHED
        assert response2.status == RequestStatus.FINISHED


class TestConsistency:
    """一致性测试"""
    
    def test_engine_vs_generate(self, model, tiny_config, engine_config):
        """测试Engine与单请求generate的一致性"""
        prompt = torch.randint(0, tiny_config.vocab_size, (8,))
        gen_config = GenerationConfig(max_new_tokens=10, do_sample=False)
        
        # 使用单请求generate
        result_generate = generate(
            model, prompt.unsqueeze(0), tiny_config, engine_config, gen_config,
            use_kv_cache=True
        ).squeeze(0)
        
        # 使用Engine
        engine = LLMEngine(model, tiny_config, engine_config)
        engine.add_request(prompt.clone(), gen_config, request_id="test")
        engine.run_until_complete()
        result_engine = engine.get_response("test").get_full_sequence()
        
        assert torch.equal(result_generate, result_engine)
    
    def test_batch_vs_individual(self, model, tiny_config, engine_config):
        """测试batch生成与逐个生成的一致性"""
        prompts = [
            torch.randint(0, tiny_config.vocab_size, (5,)),
            torch.randint(0, tiny_config.vocab_size, (7,)),
        ]
        gen_config = GenerationConfig(max_new_tokens=8, do_sample=False)
        
        # 逐个生成
        individual_results = []
        for prompt in prompts:
            result = generate(
                model, prompt.unsqueeze(0), tiny_config, engine_config, gen_config,
                use_kv_cache=True
            ).squeeze(0)
            individual_results.append(result)
        
        # batch生成
        engine = LLMEngine(model, tiny_config, engine_config)
        batch_results = engine.generate_batch(
            [p.clone() for p in prompts], gen_config
        )
        
        for i in range(len(prompts)):
            assert torch.equal(individual_results[i], batch_results[i])
    
    def test_eos_handling(self, model, tiny_config, engine_config):
        """测试EOS处理"""
        engine = LLMEngine(model, tiny_config, engine_config)
        
        prompt = torch.randint(0, tiny_config.vocab_size, (5,))
        # 使用vocab_size-1作为EOS
        gen_config = GenerationConfig(max_new_tokens=50, do_sample=False)
        
        req_id = engine.add_request(prompt, gen_config, eos_token_id=tiny_config.vocab_size - 1)
        engine.run_until_complete()
        
        response = engine.get_response(req_id)
        assert response.status == RequestStatus.FINISHED
        # 应该在50步之内完成（可能遇到EOS或达到限制）
        assert response.num_generated <= 50


class TestEdgeCases:
    """边界情况测试"""
    
    def test_empty_scheduler(self):
        """测试空调度器"""
        scheduler = Scheduler(max_batch_size=4, max_seq_len=64)
        
        assert not scheduler.has_unfinished_requests()
        assert scheduler.schedule_prefill_batch() == []
        assert scheduler.schedule_decode_batch() == []
    
    def test_single_token_generation(self, model, tiny_config, engine_config):
        """测试只生成1个token"""
        engine = LLMEngine(model, tiny_config, engine_config)
        
        prompt = torch.randint(0, tiny_config.vocab_size, (10,))
        gen_config = GenerationConfig(max_new_tokens=1, do_sample=False)
        
        req_id = engine.add_request(prompt, gen_config)
        engine.run_until_complete()
        
        response = engine.get_response(req_id)
        assert response.num_generated == 1
    
    def test_reset_engine(self, model, tiny_config, engine_config):
        """测试重置引擎"""
        engine = LLMEngine(model, tiny_config, engine_config)
        
        prompt = torch.randint(0, tiny_config.vocab_size, (5,))
        engine.add_request(prompt, GenerationConfig(max_new_tokens=5))
        engine.run_until_complete()
        
        stats1 = engine.get_stats()
        assert stats1["finished"] == 1
        
        engine.reset()
        
        stats2 = engine.get_stats()
        assert stats2["finished"] == 0
        assert stats2["total_steps"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
