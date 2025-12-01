"""
LLM推理框架日志工具模块

提供统一的日志配置和获取接口。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union


# 默认日志格式
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 全局日志级别缓存
_log_level: int = logging.INFO
_log_file: Optional[Path] = None
_initialized: bool = False


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_str: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> None:
    """
    全局日志初始化
    
    Args:
        level: 日志级别，可以是字符串("DEBUG", "INFO"等)或int
        log_file: 可选的日志文件路径
        format_str: 日志格式字符串
        date_format: 时间格式字符串
    """
    global _log_level, _log_file, _initialized
    
    # 转换字符串级别为int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    _log_level = level
    _log_file = Path(log_file) if log_file else None
    
    # 配置根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除已有的handlers
    root_logger.handlers.clear()
    
    # 创建formatter
    formatter = logging.Formatter(format_str, date_format)
    
    # 添加控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件handler
    if _log_file:
        _log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(_log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的logger
    
    Args:
        name: logger名称，通常使用模块名 __name__
        
    Returns:
        配置好的Logger实例
    """
    global _initialized
    
    # 如果尚未初始化，使用默认配置
    if not _initialized:
        setup_logging()
    
    logger = logging.getLogger(name)
    return logger


def set_log_level(level: Union[str, int]) -> None:
    """
    动态修改全局日志级别
    
    Args:
        level: 新的日志级别
    """
    global _log_level
    
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    _log_level = level
    
    # 更新根logger和所有handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)


class LoggerMixin:
    """
    为类提供logger属性的Mixin
    
    使用方式:
        class MyClass(LoggerMixin):
            def do_something(self):
                self.logger.info("Doing something...")
    """
    
    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger
