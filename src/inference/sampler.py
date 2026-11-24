"""
==============================================================================
FILE: src/inference/sampler.py
MODULE: src.inference.sampler
RESPONSIBILITY:
    - Đơn trách nhiệm: Xử lý chiến lược chọn token (Sampling Logic).
    - Biến đổi phân phối Logits [B, V] thành token ID tiếp theo thông qua:
      Greedy Search (argmax), Temperature scaling, Top-k filtering, và Top-p (Nucleus) sampling.
INPUT:
    - logits: torch.Tensor dạng [B, vocab_size]
    - temperature: float
    - top_k: Optional[int]
    - top_p: Optional[float]
OUTPUT:
    - next_token_ids: torch.Tensor dạng [B, 1]
DEPENDENCY:
    - Độc lập hoàn toàn với cấu trúc mạng nơ-ron; chỉ thao tác trên xác suất tensor PyTorch.
==============================================================================
"""

from typing import Optional
import torch


class Sampler:
    """Đơn trách nhiệm: Tính toán xác suất và trích xuất token kế tiếp."""

    @staticmethod
    def sample_next_token(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Thực thi sampling dựa trên siêu tham số giải mã."""
        if temperature == 0.0:
            # Greedy search
            return torch.argmax(logits, dim=-1, keepdim=True)

        logits = logits / temperature

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float("Inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token
