"""
==============================================================================
FILE: frontend/__init__.py
MODULE: frontend
RESPONSIBILITY: 
    - Đơn trách nhiệm: Đóng gói tầng giao diện người dùng (Presentation Layer).
    - Khởi tạo package frontend và expose hàm launch_app.
INPUT: None
OUTPUT: None
DEPENDENCY: Phụ thuộc vào frontend.app
==============================================================================
"""

from frontend.app import create_ui

__all__ = ["create_ui"]
