"""
生成请求模块

定义单个推理请求的结构和状态管理。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, TYPE_CHECKING

import torch

from my_llm_engine.config import GenerationConfig
from my_llm_engine.tensor_types import TokenIds

if TYPE_CHECKING:
    from my_llm_engine.engine.kv_cache import KVCache


class RequestStatus(Enum):
    """请求状态枚举"""
    WAITING = "waiting"      # 等待prefill
    RUNNING = "running"      # 正在decode
    FINISHED = "finished"    # 生成完成
    ERROR = "error"          # 发生错误


@dataclass
class GenerationRequest:
    """
    单个生成请求的完整表示
    
    包含输入、配置、运行时状态和输出。
    """
    
    # 基本信息
    request_id: str
    prompt_ids: torch.Tensor  # [prompt_len] 1D张量
    gen_config: GenerationConfig
    
    # 可选配置
    eos_token_id: Optional[int] = None
    
    # 运行时状态
    kv_cache: Optional["KVCache"] = None
    output_ids: List[int] = field(default_factory=list)  # 仅存储新生成的token
    past_seq_len: int = 0  # KVCache中已缓存的长度
    
    # 状态机
    status: RequestStatus = RequestStatus.WAITING
    error_msg: Optional[str] = None
    
    # 限制
    max_total_len: Optional[int] = None  # prompt_len + max_new_tokens
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保prompt_ids是1D
        if self.prompt_ids.dim() == 2:
            self.prompt_ids = self.prompt_ids.squeeze(0)
        
        # 计算最大总长度
        if self.max_total_len is None:
            self.max_total_len = len(self.prompt_ids) + self.gen_config.max_new_tokens
    
    @classmethod
    def create(
        cls,
        prompt_ids: TokenIds,
        gen_config: Optional[GenerationConfig] = None,
        eos_token_id: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> "GenerationRequest":
        """
        工厂方法：创建新请求
        
        Args:
            prompt_ids: prompt的token ID
            gen_config: 生成配置，None则使用默认
            eos_token_id: EOS token ID
            request_id: 请求ID，None则自动生成
        """
        if request_id is None:
            request_id = str(uuid.uuid4())[:8]
        
        if gen_config is None:
            gen_config = GenerationConfig()
        
        # 确保是tensor
        if not isinstance(prompt_ids, torch.Tensor):
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        
        return cls(
            request_id=request_id,
            prompt_ids=prompt_ids,
            gen_config=gen_config,
            eos_token_id=eos_token_id,
        )
    
    @property
    def prompt_len(self) -> int:
        """Prompt长度"""
        return len(self.prompt_ids)
    
    @property
    def num_generated(self) -> int:
        """已生成的token数量"""
        return len(self.output_ids)
    
    @property
    def current_len(self) -> int:
        """当前总长度（prompt + 已生成）"""
        return self.prompt_len + self.num_generated
    
    @property
    def is_finished(self) -> bool:
        """是否已完成"""
        return self.status in (RequestStatus.FINISHED, RequestStatus.ERROR)
    
    @property
    def can_continue(self) -> bool:
        """是否可以继续生成"""
        if self.is_finished:
            return False
        if self.num_generated >= self.gen_config.max_new_tokens:
            return False
        if self.max_total_len and self.current_len >= self.max_total_len:
            return False
        return True
    
    def get_last_token_id(self) -> int:
        """获取最后一个token ID（用于decode）"""
        if self.output_ids:
            return self.output_ids[-1]
        return self.prompt_ids[-1].item()
    
    def add_token(self, token_id: int) -> None:
        """添加新生成的token（不修改past_seq_len，由外部管理）"""
        self.output_ids.append(token_id)
        
        # 检查是否遇到EOS
        if self.eos_token_id is not None and token_id == self.eos_token_id:
            self.status = RequestStatus.FINISHED
        # 检查是否达到最大长度
        elif not self.can_continue:
            self.status = RequestStatus.FINISHED
    
    def get_full_sequence(self) -> torch.Tensor:
        """获取完整序列（prompt + 生成）"""
        if not self.output_ids:
            return self.prompt_ids
        
        output_tensor = torch.tensor(self.output_ids, dtype=torch.long, 
                                     device=self.prompt_ids.device)
        return torch.cat([self.prompt_ids, output_tensor])
    
    def mark_error(self, msg: str) -> None:
        """标记错误状态"""
        self.status = RequestStatus.ERROR
        self.error_msg = msg
    
    def mark_running(self) -> None:
        """标记为运行状态（prefill完成后）"""
        self.status = RequestStatus.RUNNING
    
    def __repr__(self) -> str:
        return (f"GenerationRequest(id={self.request_id}, "
                f"prompt_len={self.prompt_len}, "
                f"generated={self.num_generated}, "
                f"status={self.status.value})")
