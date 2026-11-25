"""
==============================================================================
FILE: src/model/layers.py
MODULE: src.model.layers
RESPONSIBILITY:
    - Đơn trách nhiệm: Triển khai các khối toán học nguyên thủy của Transformer hiện đại:
      1. RMSNorm: Root Mean Square Layer Normalization chuẩn Pre-Norm.
      2. RotaryEmbedding: RoPE (Rotary Positional Embedding) áp dụng trực tiếp lên Q & K.
      3. SwiGLU / FeedForward: Khối FFN phi tuyến tính SwiGLU (hoặc Linear-GELU-Linear).
INPUT:
    - Tensor đặc trưng [B, T, D] hoặc [B, H, T, D_h] cho RoPE.
OUTPUT:
    - Tensor sau chuẩn hóa hoặc biến đổi toán học.
DEPENDENCY:
    - Chỉ phụ thuộc `torch` và `torch.nn`.
==============================================================================
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Pre-Norm).
    
    Formula: y = (x / RMS(x)) * gamma
    where RMS(x) = sqrt(mean(x^2) + eps)
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tính toán phương sai trung bình bình phương trên chiều ẩn cuối cùng
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Cơ chế mã hóa vị trí RoPE (Rotary Positional Embedding).
    
    Áp dụng phép quay trực tiếp lên Query (Q) và Key (K) theo góc theta_i = base^(-2(i-1)/dim).
    Hỗ trợ cả chế độ huấn luyện (full sequence) và suy luận tự hồi quy với offset vị trí (start_pos).
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, f"RoPE dim ({dim}) must be even!"
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # inv_freq: [dim // 2] -> theta_i = 1.0 / (base ** (2i / dim))
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tạo ra cos và sin tương ứng cho dải vị trí [start_pos, start_pos + seq_len).
        
        Args:
            seq_len: Độ dài chuỗi token cần tính RoPE.
            device: Thiết bị tính toán (CPU/CUDA).
            start_pos: Vị trí bắt đầu (dành cho KV Cache autoregressive decoding).
            
        Returns:
            cos: Tensor shape [1, 1, seq_len, dim]
            sin: Tensor shape [1, 1, seq_len, dim]
        """
        t = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim // 2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        cos = emb.cos()[None, None, :, :]  # [1, 1, seq_len, dim]
        sin = emb.sin()[None, None, :, :]  # [1, 1, seq_len, dim]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Xoay nửa sau của vector sang nửa trước với dấu âm: [-x2, x1]."""
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Áp dụng phép quay RoPE lên ma trận Q và K.
    
    Args:
        q: Tensor shape [B, n_heads, seq_len, head_dim]
        k: Tensor shape [B, n_heads, seq_len, head_dim]
        cos: Tensor shape [1, 1, seq_len, head_dim]
        sin: Tensor shape [1, 1, seq_len, head_dim]
        
    Returns:
        q_rot, k_rot: Tensor shape tương ứng sau khi quay.
    """
    # Áp dụng công thức Euler: x_rot = x * cos + rotate_half(x) * sin
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class SwiGLU(nn.Module):
    """Kiến trúc Feed-Forward Network sử dụng SwiGLU:
    
    Formula: FFN(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
    """

    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class FeedForward(nn.Module):
    """Khối FeedForward đa năng: mặc định sử dụng SwiGLU (hiện đại, ổn định),
    đồng thời hỗ trợ fallback GELU (Linear-GELU-Linear).
    """

    def __init__(self, d_model: int, d_ffn: int, use_swiglu: bool = True):
        super().__init__()
        self.use_swiglu = use_swiglu
        if use_swiglu:
            self.net = SwiGLU(d_model, d_ffn)
        else:
            self.net = nn.Sequential(
                nn.Linear(d_model, d_ffn, bias=False),
                nn.GELU(),
                nn.Linear(d_ffn, d_model, bias=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
