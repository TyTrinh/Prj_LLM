"""
==============================================================================
FILE: src/model/layers.py
MODULE: src.model.layers
RESPONSIBILITY:
    - Đơn trách nhiệm: Triển khai các khối toán học nguyên thủy của Transformer hiện đại:
      1. RMSNorm: Chuẩn hóa Root Mean Square ổn định hơn LayerNorm thông thường.
      2. RoPE (Rotary Positional Embedding): Mã hóa vị trí quay trực tiếp trên Q và K.
      3. SwiGLU: Hàm kích hoạt FFN phi tuyến tính vượt trội hơn GELU chuẩn.
INPUT:
    - Tensor biểu diễn đặc trưng ẩn [B, T, D]
OUTPUT:
    - Tensor sau chuẩn hóa hoặc ánh xạ phi tuyến tính [B, T, D]
DEPENDENCY:
    - Chỉ phụ thuộc `torch` và `torch.nn`. Không phụ thuộc Tokenizer hay PEFT.
==============================================================================
"""

import torch
import torch.nn as nn
from typing import Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (tương tự LLaMA/Mistral)."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Cơ chế mã hóa vị trí RoPE (Rotary Positional Embedding)."""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        # Buffer frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tạo ma trận sin và cos theo độ dài seq_len."""
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class SwiGLU(nn.Module):
    """Kiến trúc Feed-Forward Network sử dụng SwiGLU: Swish(x * W_gate) * (x * W_up) * W_down."""
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(nn.functional.silu(self.w_gate(x)) * self.w_up(x))
