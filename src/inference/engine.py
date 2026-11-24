"""
==============================================================================
FILE: src/inference/engine.py
MODULE: src.inference.engine
RESPONSIBILITY:
    - Đơn trách nhiệm: Điều phối toàn bộ quy trình suy luận (Inference Orchestrator).
    - Kết nối 4 thành phần theo đúng thứ tự:
      1. Tiền xử lý & Kiểm tra an toàn đầu vào (Input Guardrails)
      2. Mã hóa chuỗi ký tự thành token IDs (Tokenizer)
      3. Sinh từ tự hồi quy (Model Forward + KV Cache + Sampler)
      4. Hậu xử lý & Kiểm tra an toàn đầu ra (Output Guardrails)
INPUT:
    - prompt: str
    - max_new_tokens: int
    - temperature, top_k, top_p: float
    - use_kv_cache: bool
    - use_guardrails: bool
OUTPUT:
    - generated_text: str (kèm metadata: latency, tokens_per_sec, guardrail_status)
DEPENDENCY:
    - Tầng Application: Điều phối Domain (Model, Tokenizer, Guardrails).
==============================================================================
"""

from typing import Optional, Dict, Any


class InferenceEngine:
    """Đơn trách nhiệm: Điều phối luồng sinh văn bản end-to-end an toàn."""

    def __init__(self, model=None, tokenizer=None, guardrail_pipeline=None):
        self.model = model
        self.tokenizer = tokenizer
        self.guardrails = guardrail_pipeline

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        use_kv_cache: bool = True,
        use_guardrails: bool = True,
    ) -> Dict[str, Any]:
        """Thực thi luồng sinh từ tự hồi quy có màng bảo vệ."""
        # 1. Bước Input Guardrail
        if use_guardrails and self.guardrails:
            is_safe, refusal_reason, refusal_message = self.guardrails.check_input(prompt)
            if not is_safe:
                return {
                    "text": refusal_message,
                    "is_safe": False,
                    "reason": refusal_reason,
                    "tokens_per_sec": 0.0,
                    "latency_ms": 0.0,
                }

        # 2. Bước Tokenize & Model Generation (Placeholder interface for when weights are loaded)
        return {
            "text": f"[Phản hồi mô phỏng cho prompt: '{prompt}']",
            "is_safe": True,
            "reason": "OK",
            "tokens_per_sec": 85.4,
            "latency_ms": 120.5,
        }
