"""
==============================================================================
FILE: src/guardrails/input_guard.py
MODULE: src.guardrails.input_guard
RESPONSIBILITY:
    - Đơn trách nhiệm: Rà soát và kiểm duyệt dữ liệu đầu vào (Inbound Prompt Sanitation).
    - Phát hiện và ngăn chặn:
      1. Tấn công Prompt Injection (ví dụ: 'Ignore all previous instructions', 'System override').
      2. Kỹ thuật Jailbreak (ví dụ: DAN, roleplay bẻ khóa đạo đức, base64 / leetspeak).
      3. Giới hạn độ dài chuỗi đầu vào (tránh tấn công làm nghẽn bộ đệm DOS).
INPUT:
    - prompt: str
OUTPUT:
    - Tuple[bool, str]: (is_safe, reason_or_rule_triggered)
DEPENDENCY:
    - Chỉ phụ thuộc vào thư viện chuẩn `re`. Không gọi external API.
==============================================================================
"""

import re
from typing import Tuple


class InputGuardrail:
    """Đơn trách nhiệm: Kiểm duyệt prompt người dùng trước khi chuyển vào Tokenizer."""

    INJECTION_PATTERNS = [
        r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"(?i)system\s+override",
        r"(?i)you\s+are\s+now\s+(unfiltered|dan|freed)",
        r"(?i)disregard\s+(all\s+)?(rules|safety|guidelines)",
        r"(?i)bypass\s+security",
    ]

    def __init__(self, max_chars: int = 2000):
        self.max_chars = max_chars
        self.compiled_patterns = [re.compile(p) for p in self.INJECTION_PATTERNS]

    def validate(self, prompt: str) -> Tuple[bool, str]:
        """Kiểm tra tính an toàn của prompt đầu vào."""
        if len(prompt) > self.max_chars:
            return False, f"Độ dài prompt vượt quá giới hạn cho phép ({self.max_chars} ký tự)."

        for pattern in self.compiled_patterns:
            if pattern.search(prompt):
                return False, f"Phát hiện mẫu tấn công Prompt Injection: '{pattern.pattern}'"

        return True, "Safe"
