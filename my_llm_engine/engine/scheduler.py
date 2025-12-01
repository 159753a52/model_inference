"""
调度器模块

管理请求队列，组装prefill和decode的batch。
"""

from __future__ import annotations

from typing import List, Dict, Optional
from collections import OrderedDict

from my_llm_engine.engine.request import GenerationRequest, RequestStatus
from my_llm_engine.logging_utils import get_logger


class Scheduler:
    """
    请求调度器
    
    管理waiting/running/finished队列，每步选择请求组成batch。
    
    调度策略（简化版）:
    - prefill优先：先处理waiting队列中的请求
    - decode批量：将所有running请求一起decode
    """
    
    def __init__(
        self,
        max_batch_size: int = 8,
        max_seq_len: int = 2048,
    ):
        """
        Args:
            max_batch_size: 最大批量大小
            max_seq_len: 最大序列长度
        """
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        
        # 请求队列（使用OrderedDict保持顺序）
        self._waiting: OrderedDict[str, GenerationRequest] = OrderedDict()
        self._running: OrderedDict[str, GenerationRequest] = OrderedDict()
        self._finished: OrderedDict[str, GenerationRequest] = OrderedDict()
        
        self._logger = get_logger(__name__)
    
    def add_request(self, request: GenerationRequest) -> None:
        """
        添加新请求到waiting队列
        
        Args:
            request: 生成请求
        """
        if request.prompt_len > self.max_seq_len:
            request.mark_error(f"Prompt长度{request.prompt_len}超过最大限制{self.max_seq_len}")
            self._finished[request.request_id] = request
            return
        
        request.status = RequestStatus.WAITING
        self._waiting[request.request_id] = request
        self._logger.debug(f"添加请求 {request.request_id}, prompt_len={request.prompt_len}")
    
    def has_unfinished_requests(self) -> bool:
        """是否还有未完成的请求"""
        return len(self._waiting) > 0 or len(self._running) > 0
    
    def get_num_waiting(self) -> int:
        """等待中的请求数"""
        return len(self._waiting)
    
    def get_num_running(self) -> int:
        """运行中的请求数"""
        return len(self._running)
    
    def get_num_finished(self) -> int:
        """已完成的请求数"""
        return len(self._finished)
    
    def get_request(self, request_id: str) -> Optional[GenerationRequest]:
        """获取指定请求"""
        if request_id in self._waiting:
            return self._waiting[request_id]
        if request_id in self._running:
            return self._running[request_id]
        if request_id in self._finished:
            return self._finished[request_id]
        return None
    
    def get_finished_requests(self) -> List[GenerationRequest]:
        """获取所有已完成的请求"""
        return list(self._finished.values())
    
    def schedule_prefill_batch(self) -> List[GenerationRequest]:
        """
        选择一批waiting请求进行prefill
        
        策略：
        - 按FIFO顺序选择
        - 数量不超过 max_batch_size - running数量（留空间给decode）
        - 可以一次prefill多个请求
        
        Returns:
            选中的请求列表
        """
        if not self._waiting:
            return []
        
        # 计算可用的batch空间
        available_slots = self.max_batch_size - len(self._running)
        if available_slots <= 0:
            return []
        
        batch = []
        for req_id, req in list(self._waiting.items()):
            if len(batch) >= available_slots:
                break
            
            # 检查prompt长度
            if req.prompt_len <= self.max_seq_len:
                batch.append(req)
        
        return batch
    
    def schedule_decode_batch(self) -> List[GenerationRequest]:
        """
        选择一批running请求进行decode
        
        策略：
        - 返回所有running请求（简化版，不做复杂筛选）
        - 数量受max_batch_size限制
        
        Returns:
            选中的请求列表
        """
        batch = []
        for req_id, req in self._running.items():
            if len(batch) >= self.max_batch_size:
                break
            if req.can_continue:
                batch.append(req)
        
        return batch
    
    def mark_prefill_done(self, requests: List[GenerationRequest]) -> None:
        """
        将一批请求从waiting移到running
        
        Args:
            requests: 完成prefill的请求列表
        """
        for req in requests:
            if req.request_id in self._waiting:
                del self._waiting[req.request_id]
            req.mark_running()
            self._running[req.request_id] = req
            self._logger.debug(f"请求 {req.request_id} prefill完成，进入running")
    
    def mark_finished(self, request: GenerationRequest) -> None:
        """
        将请求标记为finished
        
        Args:
            request: 要标记的请求
        """
        req_id = request.request_id
        
        if req_id in self._running:
            del self._running[req_id]
        elif req_id in self._waiting:
            del self._waiting[req_id]
        
        request.status = RequestStatus.FINISHED
        self._finished[req_id] = request
        self._logger.debug(f"请求 {req_id} 完成，生成{request.num_generated}个token")
    
    def update_finished(self) -> List[GenerationRequest]:
        """
        检查running队列，将已完成的请求移到finished
        
        Returns:
            新完成的请求列表
        """
        newly_finished = []
        
        for req_id in list(self._running.keys()):
            req = self._running[req_id]
            if req.is_finished or not req.can_continue:
                del self._running[req_id]
                req.status = RequestStatus.FINISHED
                self._finished[req_id] = req
                newly_finished.append(req)
        
        return newly_finished
    
    def get_stats(self) -> Dict[str, int]:
        """获取调度器统计信息"""
        return {
            "waiting": len(self._waiting),
            "running": len(self._running),
            "finished": len(self._finished),
        }
    
    def reset(self) -> None:
        """重置调度器状态"""
        self._waiting.clear()
        self._running.clear()
        self._finished.clear()
