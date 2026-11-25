"""
==============================================================================
FILE: src/inference/engine.py
MODULE: src.inference.engine
RESPONSIBILITY:
    - Đơn trách nhiệm: Điều phối toàn bộ quy trình suy luận (Inference Orchestrator).
    - Kết nối 4 thành phần theo đúng thứ tự:
      1. Tiền xử lý & Kiểm tra an toàn đầu vào (Input Guardrails)
      2. Mã hóa chuỗi ký tự thành token IDs (Tokenizer qua apply_chat_template)
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

import time
from typing import Optional, Dict, Any
import torch

from src.tokenizer.special_tokens import EOS_ID


class InferenceEngine:
    """Đơn trách nhiệm: Điều phối luồng sinh văn bản end-to-end an toàn."""

    def __init__(self, model=None, tokenizer=None, guardrail_pipeline=None, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.guardrails = guardrail_pipeline
        self.device = device
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

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
        start_time = time.perf_counter()

        # 1. Bước Input Guardrail
        if use_guardrails and self.guardrails:
            is_safe, refusal_reason, refusal_message = self.guardrails.check_input(prompt)
            if not is_safe:
                return {
                    "text": refusal_message,
                    "is_safe": False,
                    "reason": refusal_reason,
                    "tokens_per_sec": 0.0,
                    "latency_ms": (time.perf_counter() - start_time) * 1000,
                }

        # Nếu chưa nạp model hoặc tokenizer, fallback an toàn
        if self.model is None or self.tokenizer is None:
            return {
                "text": f"[Mô hình chưa nạp weights thực tế cho prompt: '{prompt}']",
                "is_safe": True,
                "reason": "OK (Stub fallback)",
                "tokens_per_sec": 0.0,
                "latency_ms": (time.perf_counter() - start_time) * 1000,
            }

        # 2. Tokenize với Chat Template (mồi sinh <|assistant|>)
        if hasattr(self.tokenizer, "apply_chat_template"):
            input_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        else:
            input_ids = self.tokenizer.encode(prompt)

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        prompt_len = input_tensor.size(1)

        # 3. Model Generation (Tự hồi quy)
        self.model.eval()
        with torch.no_grad():
            out_tensor = self.model.generate(
                idx=input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_kv_cache=use_kv_cache,
                eos_token_id=EOS_ID,
            )

        # Chỉ lấy phần token mới sinh ra
        gen_tokens = out_tensor[0, prompt_len:].tolist()
        gen_text = self.tokenizer.decode(gen_tokens)

        # 4. Bước Output Guardrail
        if use_guardrails and self.guardrails:
            is_safe, refusal_reason, filtered_text = self.guardrails.check_output(gen_text)
            if not is_safe:
                return {
                    "text": filtered_text,
                    "is_safe": False,
                    "reason": refusal_reason,
                    "tokens_per_sec": 0.0,
                    "latency_ms": (time.perf_counter() - start_time) * 1000,
                }
            gen_text = filtered_text

        elapsed = time.perf_counter() - start_time
        num_generated = len(gen_tokens)
        tokens_per_sec = (num_generated / elapsed) if elapsed > 0 else 0.0

        return {
            "text": gen_text,
            "is_safe": True,
            "reason": "OK",
            "tokens_per_sec": tokens_per_sec,
            "latency_ms": elapsed * 1000,
        }
