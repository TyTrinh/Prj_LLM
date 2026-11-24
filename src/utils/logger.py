"""
==============================================================================
FILE: src/utils/logger.py
MODULE: src.utils.logger
RESPONSIBILITY:
    - Đơn trách nhiệm: Thiết lập hệ thống ghi log có cấu trúc cho ứng dụng và quá trình huấn luyện.
    - Ghi log ra đồng thời màn hình Console và file `runs.log`.
==============================================================================
"""

import logging
import sys


def get_logger(name: str = "LLM") -> logging.Logger:
    """Tạo hoặc lấy logger chuẩn định dạng."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
