"""
轻量级性能分析工具

提供简单的计时和统计功能。
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class TimingStats:
    """单个计时项的统计"""
    name: str
    total_time: float = 0.0
    count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0
    
    def add(self, duration: float) -> None:
        self.total_time += duration
        self.count += 1
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
    
    @property
    def avg_time(self) -> float:
        return self.total_time / self.count if self.count > 0 else 0.0
    
    def __repr__(self) -> str:
        if self.count == 0:
            return f"{self.name}: (no data)"
        return (f"{self.name}: total={self.total_time:.4f}s, "
                f"count={self.count}, avg={self.avg_time*1000:.2f}ms, "
                f"min={self.min_time*1000:.2f}ms, max={self.max_time*1000:.2f}ms")


class SimpleProfiler:
    """
    简单性能分析器
    
    用法:
        profiler = SimpleProfiler()
        
        with profiler.record("prefill"):
            # 执行prefill
            ...
        
        with profiler.record("decode"):
            # 执行decode
            ...
        
        print(profiler.summary())
    """
    
    def __init__(self, enabled: bool = True):
        """
        Args:
            enabled: 是否启用profiling
        """
        self.enabled = enabled
        self._stats: Dict[str, TimingStats] = {}
        self._start_time: Optional[float] = None
        self._counters: Dict[str, int] = defaultdict(int)
    
    @contextmanager
    def record(self, name: str):
        """
        记录代码块的执行时间
        
        Args:
            name: 计时项名称
        """
        if not self.enabled:
            yield
            return
        
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            if name not in self._stats:
                self._stats[name] = TimingStats(name=name)
            self._stats[name].add(duration)
    
    def increment(self, name: str, value: int = 1) -> None:
        """
        增加计数器
        
        Args:
            name: 计数器名称
            value: 增加的值
        """
        if self.enabled:
            self._counters[name] += value
    
    def start_session(self) -> None:
        """开始一个profiling会话"""
        self._start_time = time.perf_counter()
    
    def end_session(self) -> float:
        """结束会话，返回总时间"""
        if self._start_time is None:
            return 0.0
        duration = time.perf_counter() - self._start_time
        self._start_time = None
        return duration
    
    def get_stats(self, name: str) -> Optional[TimingStats]:
        """获取指定项的统计"""
        return self._stats.get(name)
    
    def summary(self) -> Dict[str, dict]:
        """
        返回统计摘要
        
        Returns:
            字典形式的统计数据
        """
        result = {}
        for name, stats in self._stats.items():
            result[name] = {
                "total_time": stats.total_time,
                "count": stats.count,
                "avg_time": stats.avg_time,
                "min_time": stats.min_time if stats.count > 0 else 0,
                "max_time": stats.max_time,
            }
        
        if self._counters:
            result["counters"] = dict(self._counters)
        
        return result
    
    def print_summary(self) -> None:
        """打印统计摘要"""
        print("\n=== Profiler Summary ===")
        
        # 按总时间排序
        sorted_stats = sorted(
            self._stats.values(), 
            key=lambda x: x.total_time, 
            reverse=True
        )
        
        for stats in sorted_stats:
            print(stats)
        
        if self._counters:
            print("\nCounters:")
            for name, value in self._counters.items():
                print(f"  {name}: {value}")
        
        print("========================\n")
    
    def reset(self) -> None:
        """重置所有统计"""
        self._stats.clear()
        self._counters.clear()
        self._start_time = None


class DummyProfiler:
    """
    空实现的Profiler，用于禁用profiling时
    """
    
    @contextmanager
    def record(self, name: str):
        yield
    
    def increment(self, name: str, value: int = 1) -> None:
        pass
    
    def start_session(self) -> None:
        pass
    
    def end_session(self) -> float:
        return 0.0
    
    def summary(self) -> Dict:
        return {}
    
    def print_summary(self) -> None:
        pass
    
    def reset(self) -> None:
        pass


def create_profiler(enabled: bool = True) -> SimpleProfiler:
    """
    创建Profiler实例
    
    Args:
        enabled: 是否启用
        
    Returns:
        Profiler实例
    """
    if enabled:
        return SimpleProfiler(enabled=True)
    return DummyProfiler()
