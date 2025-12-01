"""
推理引擎模块

包含:
- kv_cache.py: Dense KV Cache
- block_manager.py: Block管理器
- paged_kv_cache.py: Paged KV Cache
- request.py: 生成请求抽象
- scheduler.py: 请求调度器
- engine.py: LLM推理引擎
- generation.py: 文本生成逻辑
- profiler.py: 性能分析工具
"""

from my_llm_engine.engine.kv_cache import KVCache
from my_llm_engine.engine.block_manager import BlockManager
from my_llm_engine.engine.paged_kv_cache import PagedKVCache, PagedKVCacheWrapper
from my_llm_engine.engine.request import GenerationRequest, RequestStatus
from my_llm_engine.engine.scheduler import Scheduler
from my_llm_engine.engine.engine import LLMEngine
from my_llm_engine.engine.profiler import SimpleProfiler, create_profiler
from my_llm_engine.engine.generation import (
    generate,
    sample_next_token,
    greedy_decode,
    top_k_filtering,
    top_p_filtering,
    benchmark_generation,
)

__all__ = [
    # KV Cache
    "KVCache",
    "BlockManager",
    "PagedKVCache",
    "PagedKVCacheWrapper",
    # 请求与调度
    "GenerationRequest",
    "RequestStatus",
    "Scheduler",
    "LLMEngine",
    # 性能分析
    "SimpleProfiler",
    "create_profiler",
    # 生成函数
    "generate",
    "sample_next_token",
    "greedy_decode",
    "top_k_filtering",
    "top_p_filtering",
    "benchmark_generation",
]
