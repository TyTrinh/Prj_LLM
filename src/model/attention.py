"""
==============================================================================
FILE: src/model/attention.py
MODULE: src.model.attention
RESPONSIBILITY:
    - Đơn trách nhiệm: Tính toán cơ chế Causal Multi-Head Self-Attention.
    - Áp dụng RoPE (Rotary Positional Embedding) lên Q và K.
    - Causal Mask tam giác dưới ngăn rò rỉ thông tin token tương lai.
    - Hỗ trợ 2 chế độ:
      1. Prefill: Tính full attention matrix với causal mask [T, T].
      2. Decode: Tận dụng KVCache chỉ tính attention của 1 token mới nhất đối với lịch sử.
INPUT:
    - x: torch.Tensor [B, T, D]
    - kv_cache: Optional[KVCache]
    - layer_idx: int
    - start_pos: int (offset vị trí khi sinh autoregressive với cache)
OUTPUT:
    - out: torch.Tensor [B, T, D]
DEPENDENCY:
    - Phụ thuộc `src.model.layers.RotaryEmbedding`, `apply_rotary_pos_emb`, và `src.model.kv_cache.KVCache`.
==============================================================================
"""

import math
from typing import Optional
import torch
import torch.nn as nn
from src.model.kv_cache import KVCache
from src.model.layers import RotaryEmbedding, apply_rotary_pos_emb


class CausalSelfAttention(nn.Module):
    """Cơ chế Multi-Head Attention nhân quả có tích hợp RoPE và KV Cache."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 512,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Ma trận chiếu Q, K, V (Mục tiêu gắn LoRA sau này: q_proj, v_proj)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Bộ mã hóa vị trí quay RoPE
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            base=rope_base,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        start_pos: Optional[int] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Linear projections
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, head_dim]
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, head_dim]
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, head_dim]

        # Xác định offset vị trí cho RoPE khi có KV Cache
        if start_pos is None:
            if kv_cache is not None and kv_cache.k_cache[layer_idx] is not None:
                start_pos = kv_cache.k_cache[layer_idx].shape[2]
            else:
                start_pos = 0

        # Áp dụng RoPE lên Q & K
        cos, sin = self.rotary_emb(seq_len=T, device=x.device, start_pos=start_pos)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Cập nhật KV Cache nếu được kích hoạt
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, layer_idx)

        total_k_len = k.size(-2)

        # Scaled Dot-Product Attention: (Q @ K^T) / sqrt(head_dim)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # [B, H, T, total_k_len]

        # Áp dụng Causal Mask khi T > 1 hoặc khi chưa dùng KV Cache
        if kv_cache is None or T > 1:
            # Tạo causal mask tam giác dưới kích thước [T, total_k_len]
            # Token tại vị trí t (offset start_pos + t) chỉ được chú ý đến các key index <= (start_pos + t)
            q_idx = torch.arange(start_pos, start_pos + T, device=x.device).unsqueeze(1)
            k_idx = torch.arange(total_k_len, device=x.device).unsqueeze(0)
            causal_mask = q_idx >= k_idx  # [T, total_k_len]
            att = att.masked_fill(~causal_mask, float("-inf"))

        att = torch.softmax(att, dim=-1)
        y = att @ v  # [B, H, T, head_dim]
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        return self.out_proj(y)
