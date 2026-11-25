"""
==============================================================================
FILE: src/model/__init__.py
MODULE: src.model
RESPONSIBILITY:
    - Expose các thành phần chính của kiến trúc Transformer:
      GPTModel, TransformerBlock, CausalSelfAttention, RMSNorm, RotaryEmbedding,
      SwiGLU, FeedForward, KVCache, apply_rotary_pos_emb
==============================================================================
"""

from src.model.layers import (
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    FeedForward,
    apply_rotary_pos_emb,
)
from src.model.attention import CausalSelfAttention
from src.model.kv_cache import KVCache
from src.model.transformer import TransformerBlock, GPTModel

__all__ = [
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLU",
    "FeedForward",
    "apply_rotary_pos_emb",
    "CausalSelfAttention",
    "KVCache",
    "TransformerBlock",
    "GPTModel",
]
