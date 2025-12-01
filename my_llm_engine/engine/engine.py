"""
LLM推理引擎模块

统一管理模型、调度器和KVCache，提供高层API。
"""

from __future__ import annotations

import logging
from typing import Optional, List, Sequence, Dict, TYPE_CHECKING

import torch

from my_llm_engine.config import ModelConfig, EngineConfig, GenerationConfig
from my_llm_engine.tensor_types import TokenIds
from my_llm_engine.logging_utils import get_logger
from my_llm_engine.engine.kv_cache import KVCache
from my_llm_engine.engine.request import GenerationRequest, RequestStatus
from my_llm_engine.engine.scheduler import Scheduler

if TYPE_CHECKING:
    from my_llm_engine.models.transformer import DecoderOnlyModel


class LLMEngine:
    """
    LLM推理引擎
    
    管理模型、调度器和请求生命周期，支持批量推理。
    
    使用方式:
    1. 创建Engine
    2. 调用add_request()添加请求
    3. 循环调用step()或一次性调用run_until_complete()
    4. 通过get_response()获取结果
    """
    
    def __init__(
        self,
        model: "DecoderOnlyModel",
        model_config: ModelConfig,
        engine_config: EngineConfig,
        scheduler: Optional[Scheduler] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            model: Transformer模型
            model_config: 模型配置
            engine_config: 引擎配置
            scheduler: 调度器，None则创建默认调度器
            logger: 日志器
        """
        self.model = model
        self.model_config = model_config
        self.engine_config = engine_config
        
        if scheduler is None:
            scheduler = Scheduler(
                max_batch_size=engine_config.max_batch_size,
                max_seq_len=engine_config.max_seq_len,
            )
        self.scheduler = scheduler
        
        self._logger = logger or get_logger(__name__)
        
        # 请求ID到请求的映射
        self._requests: Dict[str, GenerationRequest] = {}
        
        # 统计
        self._total_steps = 0
    
    @property
    def device(self) -> torch.device:
        """模型所在设备"""
        return self.engine_config.torch_device
    
    @property
    def dtype(self) -> torch.dtype:
        """模型数据类型"""
        return self.engine_config.torch_dtype
    
    def add_request(
        self,
        prompt_ids: TokenIds,
        gen_config: Optional[GenerationConfig] = None,
        eos_token_id: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """
        添加新的生成请求
        
        Args:
            prompt_ids: prompt的token ID, shape [S] 或 [1, S]
            gen_config: 生成配置
            eos_token_id: EOS token ID
            request_id: 请求ID，None则自动生成
            
        Returns:
            请求ID
        """
        # 创建请求
        request = GenerationRequest.create(
            prompt_ids=prompt_ids,
            gen_config=gen_config,
            eos_token_id=eos_token_id,
            request_id=request_id,
        )
        
        # 将prompt移到正确设备
        request.prompt_ids = request.prompt_ids.to(self.device)
        
        # 检查长度限制
        max_total = request.prompt_len + request.gen_config.max_new_tokens
        if max_total > self.engine_config.max_seq_len:
            request.max_total_len = self.engine_config.max_seq_len
        
        # 创建KVCache
        request.kv_cache = KVCache.empty(
            self.model_config,
            self.engine_config,
            batch_size=1,
        )
        
        # 添加到调度器
        self.scheduler.add_request(request)
        self._requests[request.request_id] = request
        
        self._logger.debug(f"添加请求: {request}")
        return request.request_id
    
    def step(self) -> Dict[str, int]:
        """
        执行一步推理
        
        流程:
        1. 如果有waiting请求，执行batched prefill
        2. 对所有running请求执行batched decode
        3. 更新请求状态
        
        Returns:
            本步统计信息
        """
        stats = {"prefill": 0, "decode": 0, "finished": 0}
        
        # 1. Prefill阶段
        prefill_batch = self.scheduler.schedule_prefill_batch()
        if prefill_batch:
            self._execute_prefill(prefill_batch)
            self.scheduler.mark_prefill_done(prefill_batch)
            stats["prefill"] = len(prefill_batch)
        
        # 2. Decode阶段
        decode_batch = self.scheduler.schedule_decode_batch()
        if decode_batch:
            self._execute_decode(decode_batch)
            stats["decode"] = len(decode_batch)
        
        # 3. 更新完成状态
        newly_finished = self.scheduler.update_finished()
        stats["finished"] = len(newly_finished)
        
        self._total_steps += 1
        
        return stats
    
    def _execute_prefill(self, requests: List[GenerationRequest]) -> None:
        """
        执行batched prefill
        
        当前实现：逐个请求执行prefill（简化版）
        未来优化：可以将多个请求padding后组成真正的batch
        
        注意：prefill只执行前向计算和采样，不增加past_seq_len。
        past_seq_len的增加在decode阶段处理，以保持与generation.py的一致性。
        """
        with torch.no_grad():
            for req in requests:
                # 准备输入 [1, prompt_len]
                input_ids = req.prompt_ids.unsqueeze(0)
                
                # 执行prefill
                logits = self.model.prefill(input_ids, req.kv_cache)
                
                # prefill后past_seq_len等于prompt长度（不+1，与generation.py一致）
                req.past_seq_len = req.prompt_len
                
                # 采样第一个token（这个token会在下一步decode中被处理）
                next_token_logits = logits[0, -1, :]
                next_token = self._sample_token(next_token_logits, req.gen_config)
                req.add_token(next_token)
    
    def _execute_decode(self, requests: List[GenerationRequest]) -> None:
        """
        执行batched decode
        
        当前实现：逐个请求执行decode（简化版）
        未来优化：可以将多个请求的last_token组成batch
        """
        with torch.no_grad():
            for req in requests:
                if not req.can_continue:
                    continue
                
                # 获取最后一个token
                last_token = torch.tensor(
                    [[req.get_last_token_id()]],
                    dtype=torch.long,
                    device=self.device
                )
                
                # 记录当前past_seq_len（decode前的长度）
                current_past_len = req.past_seq_len
                
                # 执行decode
                logits = self.model.decode(
                    last_token,
                    req.kv_cache,
                    past_seq_len=current_past_len,
                )
                
                # decode后past_seq_len增加1
                req.past_seq_len += 1
                
                # 采样下一个token
                next_token_logits = logits[0, -1, :]
                next_token = self._sample_token(next_token_logits, req.gen_config)
                req.add_token(next_token)
    
    def _sample_token(self, logits: torch.Tensor, gen_config: GenerationConfig) -> int:
        """
        从logits采样token
        
        Args:
            logits: [vocab_size]
            gen_config: 生成配置
            
        Returns:
            采样的token ID
        """
        if not gen_config.do_sample:
            return logits.argmax().item()
        
        # 应用温度
        if gen_config.temperature != 1.0:
            logits = logits / gen_config.temperature
        
        # Top-K
        if gen_config.top_k > 0 and gen_config.top_k < logits.shape[-1]:
            indices_to_remove = logits < torch.topk(logits, gen_config.top_k).values[-1]
            logits = logits.masked_fill(indices_to_remove, float("-inf"))
        
        # Top-P
        if gen_config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > gen_config.top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                0, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))
        
        # 采样
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()
    
    def run_until_complete(self, max_steps: Optional[int] = None) -> None:
        """
        运行直到所有请求完成
        
        Args:
            max_steps: 最大步数限制，None则无限制
        """
        steps = 0
        while self.scheduler.has_unfinished_requests():
            if max_steps is not None and steps >= max_steps:
                self._logger.warning(f"达到最大步数{max_steps}，停止")
                break
            
            self.step()
            steps += 1
        
        self._logger.debug(f"运行完成，共{steps}步")
    
    def get_response(self, request_id: str) -> Optional[GenerationRequest]:
        """
        获取指定请求的状态
        
        Args:
            request_id: 请求ID
            
        Returns:
            请求对象，不存在则返回None
        """
        return self._requests.get(request_id)
    
    def get_all_responses(self) -> Dict[str, GenerationRequest]:
        """获取所有请求"""
        return self._requests.copy()
    
    def generate_batch(
        self,
        batch_prompts: Sequence[TokenIds],
        gen_config: Optional[GenerationConfig] = None,
        eos_token_id: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        同步批量生成
        
        Args:
            batch_prompts: 多个prompt的token ID列表
            gen_config: 生成配置（所有请求共用）
            eos_token_id: EOS token ID
            
        Returns:
            每个请求的完整序列列表
        """
        # 添加所有请求
        request_ids = []
        for prompt in batch_prompts:
            req_id = self.add_request(
                prompt_ids=prompt,
                gen_config=gen_config,
                eos_token_id=eos_token_id,
            )
            request_ids.append(req_id)
        
        # 运行直到完成
        self.run_until_complete()
        
        # 收集结果
        results = []
        for req_id in request_ids:
            req = self.get_response(req_id)
            if req is not None:
                results.append(req.get_full_sequence())
            else:
                results.append(torch.tensor([], dtype=torch.long))
        
        return results
    
    def get_stats(self) -> Dict[str, int]:
        """获取引擎统计信息"""
        scheduler_stats = self.scheduler.get_stats()
        return {
            **scheduler_stats,
            "total_requests": len(self._requests),
            "total_steps": self._total_steps,
        }
    
    def reset(self) -> None:
        """重置引擎状态"""
        self.scheduler.reset()
        self._requests.clear()
        self._total_steps = 0
