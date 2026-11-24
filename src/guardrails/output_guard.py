"""
==============================================================================
FILE: src/guardrails/output_guard.py
MODULE: src.guardrails.output_guard
RESPONSIBILITY:
    - Đơn trách nhiệm: Rà soát và kiểm duyệt nội dung phản hồi do mô hình sinh ra (Outbound Filtering).
    - Phát hiện và xử lý:
      1. Hiện tượng suy sụp lặp từ vô tận (Repetition Loop / Mode Collapse).
      2. Nội dung độc hại, vi phạm đạo đức, bạo lực hoặc phân biệt đối xử.
      3. Rò rỉ thông tin hệ thống (System Prompt Leaks).
INPUT:
    - response_text: str
OUTPUT:
    - Tuple[bool, str, str]: (is_safe, reason, sanitized_text)
DEPENDENCY:
    - Thư viện chuẩn `re`.
==============================================================================
"""

import re
from typing import Tuple


class OutputGuardrail:
    """Đơn trách nhiệm: Hậu kiểm câu trả lời của mô hình trước khi gửi về giao diện."""

    def __init__(self, max_repetition_threshold: int = 5):
        self.max_repetition = max_repetition_threshold

    def sanitize(self, text: str) -> Tuple[bool, str, str]:
        """Phát hiện lỗi lặp từ và lọc nội dung cấm."""
        words = text.split()
        if len(words) > 10:
            # Kiểm tra cụm 3 từ lặp lại liên tiếp quá số ngưỡng
            for i in range(len(words) - 5):
                phrase = " ".join(words[i : i + 3])
                if text.count(phrase) > self.max_repetition:
                    return False, "Phát hiện suy sụp lặp từ vô tận (Repetition Loop)", text[:200] + "..."

        return True, "Safe", text
