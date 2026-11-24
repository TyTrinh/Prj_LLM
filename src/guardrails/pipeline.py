"""
==============================================================================
FILE: src/guardrails/pipeline.py
MODULE: src.guardrails.pipeline
RESPONSIBILITY:
    - Đơn trách nhiệm: Điều phối thống nhất hệ sinh thái Guardrails 2 tầng (Hybrid Pipeline).
    - Cung cấp 2 cổng kiểm tra:
      - `check_input(prompt)` -> (is_safe, reason, refusal_message)
      - `check_output(response)` -> (is_safe, reason, sanitized_response)
INPUT:
    - prompt (str) hoặc response (str)
OUTPUT:
    - Phán quyết an toàn và văn bản đã qua xử lý
DEPENDENCY:
    - Phụ thuộc `src.guardrails.input_guard` và `src.guardrails.output_guard`.
==============================================================================
"""

from typing import Tuple
from src.guardrails.input_guard import InputGuardrail
from src.guardrails.output_guard import OutputGuardrail
from src.config import GuardrailConfig


class GuardrailPipeline:
    """Đơn trách nhiệm: Màng bọc an toàn 2 tầng bao bọc vòng đời của mô hình."""

    def __init__(self, config: GuardrailConfig = GuardrailConfig()):
        self.config = config
        self.input_guard = InputGuardrail(max_chars=config.max_input_length)
        self.output_guard = OutputGuardrail()

    def check_input(self, prompt: str) -> Tuple[bool, str, str]:
        """Kiểm duyệt tiền xử lý prompt."""
        if not self.config.enable_input_guard:
            return True, "Disabled", prompt

        is_safe, reason = self.input_guard.validate(prompt)
        if not is_safe:
            return False, reason, self.config.refusal_response
        return True, "Safe", prompt

    def check_output(self, response: str) -> Tuple[bool, str, str]:
        """Kiểm duyệt hậu xử lý câu trả lời."""
        if not self.config.enable_output_guard:
            return True, "Disabled", response

        is_safe, reason, sanitized = self.output_guard.sanitize(response)
        if not is_safe:
            return False, reason, self.config.refusal_response
        return True, "Safe", sanitized
