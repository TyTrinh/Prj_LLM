"""
==============================================================================
FILE: src/model/attention.py
MODULE: src.model.attention
RESPONSIBILITY:
    - Đơn trách nhiệm: Tính toán cơ chế Causal Multi-Head Self-Attention.
    - Hỗ trợ 2 chế độ:
      1. Prefill: Tính full attention matrix với causal mask tam giác dưới [T, T].
      2. Decode: Tận dụng KVCache chỉ tính attention của 1 token mới nhất đối với lịch sử.
INPUT:
    - x: torch.Tensor [B, T, D]
    - kv_cache: Optional[KVCache]
    - layer_idx: int
OUTPUT:
    - out: torch.Tensor [B, T, D]
DEPENDENCY:
    - Phụ thuộc `src.model.layers.RotaryEmbedding` và `src.model.kv_cache.KVCache`.
==============================================================================
"""

import math
import torch
import torch.nn as nn
from typing import Optional
from src.model.kv_cache import KVCache


class CausalSelfAttention(nn.Module):
    """Cơ chế Multi-Head Attention nhân quả có tích hợp KV Cache và RoPE."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 512):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Ma trận chiếu Q, K, V (Mục tiêu gắn LoRA sau này: q_proj, v_proj)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v, layer_idx)

        # Scaled Dot-Product Attention với Causal Mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if kv_cache is None or T > 1:
            causal_mask = torch.tril(torch.ones((T, k.size(-2)), device=x.device)).bool()
            att = att.masked_fill(~causal_mask, float("-inf"))

        att = torch.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)
