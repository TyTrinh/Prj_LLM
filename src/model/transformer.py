"""
==============================================================================
FILE: src/model/transformer.py
MODULE: src.model.transformer
RESPONSIBILITY:
    - Đơn trách nhiệm: Lắp ráp kiến trúc TransformerBlock và mô hình GPTModel hoàn chỉnh.
    - Áp dụng cấu trúc Pre-LayerNorm (Pre-RMSNorm) với Residual Connections.
    - Áp dụng Weight Tying giữa ma trận nhúng từ (Token Embedding) và đầu chiếu ra Logits (LM Head).
    - Khởi tạo trọng số chuẩn cho mô hình Transformer từ số 0.
INPUT:
    - idx: torch.Tensor [B, T] (chuỗi token IDs)
OUTPUT:
    - logits: torch.Tensor [B, T, vocab_size]
DEPENDENCY:
    - Phụ thuộc `src.config.ModelConfig`, `src.model.layers`, `src.model.attention`.
==============================================================================
"""

import torch
import torch.nn as nn
from typing import Optional
from src.config import ModelConfig
from src.model.layers import RMSNorm, SwiGLU
from src.model.attention import CausalSelfAttention
from src.model.kv_cache import KVCache


class TransformerBlock(nn.Module):
    """Một khối Transformer chuẩn Pre-Norm gồm Attention và SwiGLU FFN."""

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config.d_model, config.n_heads, config.block_size)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = SwiGLU(config.d_model, config.d_ffn)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        # Pre-Norm Residual for Attention
        x = x + self.attn(self.attn_norm(x), kv_cache=kv_cache, layer_idx=self.layer_idx)
        # Pre-Norm Residual for FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GPTModel(nn.Module):
    """Mô hình Decoder-Only Transformer hoàn chỉnh lập trình từ số 0."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight Tying: Chia sẻ ma trận trọng số giữa embedding và head
        if config.tie_weights:
            self.lm_head.weight = self.tok_embeddings.weight

        # Khởi tạo trọng số chuẩn GPT
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        B, T = idx.shape
        x = self.tok_embeddings(idx)

        for block in self.blocks:
            x = block(x, kv_cache=kv_cache)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits
