"""
日志工具模块
提供统一的日志记录接口
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def get_logger(name: str, level: Optional[str] = None, log_dir: str = "./logs") -> logging.Logger:
    """
    获取日志记录器

    Args:
        name: 日志记录器名称
        level: 日志级别
        log_dir: 日志文件目录

    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    if level is None:
        level = "INFO"

    logger.setLevel(getattr(logging, level.upper()))

    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器 - 按日期生成日志文件
    today = datetime.now().strftime('%Y-%m-%d')
    log_file = log_path / f"darwin_{today}.log"

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger