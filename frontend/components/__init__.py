"""
==============================================================================
FILE: frontend/components/__init__.py
MODULE: frontend.components
RESPONSIBILITY:
    - Đơn trách nhiệm: Đóng gói các tab giao diện độc lập trong kiến trúc Clean.
    - Expose các hàm khởi tạo giao diện cho từng màn hình nghiệp vụ.
==============================================================================
"""

from frontend.components.chat_tab import render_chat_tab
from frontend.components.comparison_tab import render_comparison_tab
from frontend.components.guardrails_tab import render_guardrails_tab
from frontend.components.profiling_tab import render_profiling_tab

__all__ = [
    "render_chat_tab",
    "render_comparison_tab",
    "render_guardrails_tab",
    "render_profiling_tab",
]
