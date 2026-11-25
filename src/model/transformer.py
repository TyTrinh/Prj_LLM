"""
==============================================================================
FILE: src/model/transformer.py
MODULE: src.model.transformer
RESPONSIBILITY:
    - Đơn trách nhiệm: Lắp ráp kiến trúc TransformerBlock và mô hình GPTModel hoàn chỉnh.
    - Áp dụng Pre-RMSNorm với kết nối thặng dư (Residual Connections).
    - Áp dụng Weight Tying giữa ma trận nhúng từ (Token Embedding) và đầu chiếu ra Logits (LM Head).
    - Khởi tạo trọng số chuẩn GPT: Gaussian N(0, 0.02), chia căn(2 * n_layers) tại residual projection.
INPUT:
    - idx: torch.Tensor [B, T] (chuỗi token IDs)
OUTPUT:
    - logits: torch.Tensor [B, T, vocab_size]
DEPENDENCY:
    - Phụ thuộc `src.config.ModelConfig`, `src.model.layers`, `src.model.attention`, `src.model.kv_cache`.
==============================================================================
"""

import math
from typing import Optional
import torch
import torch.nn as nn
from src.config import ModelConfig
from src.model.layers import RMSNorm, FeedForward
from src.model.attention import CausalSelfAttention
from src.model.kv_cache import KVCache
from src.inference.sampler import Sampler


class TransformerBlock(nn.Module):
    """Một khối Transformer chuẩn Pre-Norm gồm Attention (với RoPE) và FeedForward (SwiGLU)."""

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            max_seq_len=config.block_size,
            rope_base=config.rope_base,
        )
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = FeedForward(config.d_model, config.d_ffn, use_swiglu=True)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        start_pos: Optional[int] = None,
    ) -> torch.Tensor:
        # Pre-Norm Residual for Attention
        x = x + self.attn(
            self.attn_norm(x),
            kv_cache=kv_cache,
            layer_idx=self.layer_idx,
            start_pos=start_pos,
        )
        # Pre-Norm Residual for FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GPTModel(nn.Module):
    """Mô hình Decoder-Only Transformer hoàn chỉnh lập trình từ số 0 bằng PyTorch thuần."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight Tying: Chia sẻ ma trận trọng số giữa embedding và head (W_head = W_E^T)
        if config.tie_weights:
            self.lm_head.weight = self.tok_embeddings.weight

        # Khởi tạo trọng số chuẩn GPT
        self.apply(self._init_weights)

        # Tinh chỉnh khởi tạo cho residual projections: chia căn(2 * n_layers) theo chuẩn GPT-2/LLaMA
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            # attn.out_proj
            if hasattr(block.attn, "out_proj") and isinstance(block.attn.out_proj, nn.Linear):
                torch.nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=residual_std)
            # ffn.w_down
            if hasattr(block.ffn.net, "w_down") and isinstance(block.ffn.net.w_down, nn.Linear):
                torch.nn.init.normal_(block.ffn.net.w_down.weight, mean=0.0, std=residual_std)
            elif hasattr(block.ffn.net, "2") and isinstance(block.ffn.net[2], nn.Linear):
                torch.nn.init.normal_(block.ffn.net[2].weight, mean=0.0, std=residual_std)

    def _init_weights(self, module: nn.Module):
        """Khởi tạo trọng số cơ sở Gaussian N(0, 0.02) và zeros cho bias."""
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
        start_pos: Optional[int] = None,
    ) -> torch.Tensor:
        """Thực thi forward pass qua toàn bộ mô hình GPT.
        
        Args:
            idx: Tensor shape [B, T] chứa chuỗi token IDs
            kv_cache: Optional[KVCache] bộ nhớ đệm KV
            start_pos: Optional[int] vị trí bắt đầu chuỗi
            
        Returns:
            logits: Tensor shape [B, T, vocab_size]
        """
        B, T = idx.shape
        x = self.tok_embeddings(idx)  # [B, T, d_model]

        for block in self.blocks:
            x = block(x, kv_cache=kv_cache, start_pos=start_pos)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]
        return logits

    def count_parameters(self) -> int:
        """Đếm tổng số tham số (có tính đến Weight Tying)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_kv_cache: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Sinh chuỗi token tự hồi quy với hoặc không có KV Cache.
        
        Args:
            idx: Tensor shape [B, T] chứa chuỗi token IDs ban đầu (prompt)
            max_new_tokens: Số token tối đa cần sinh thêm
            temperature: Nhiệt độ chia logits (0.0: Greedy Search)
            top_k: Giới hạn top-k xác suất cao nhất
            top_p: Nucleus sampling giữ lại top tokens có tổng xác suất <= top_p
            use_kv_cache: Bật hoặc tắt KV Cache
            eos_token_id: Token ID dừng sinh (End-of-Sequence)
            
        Returns:
            Tensor shape [B, T + num_tokens_generated] chứa chuỗi token đầy đủ
        """
        B, T = idx.shape
        device = idx.device

        if use_kv_cache:
            kv_cache = KVCache(n_layers=self.config.n_layers)
            
            # Giai đoạn 1: Prefill prompt ban đầu
            # Đảm bảo prompt không vượt quá block_size
            curr_idx = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(curr_idx, kv_cache=kv_cache, start_pos=0)
            next_token_logits = logits[:, -1, :]  # [B, vocab_size]
            next_token = Sampler.sample_next_token(
                next_token_logits, temperature=temperature, top_k=top_k, top_p=top_p
            )  # [B, 1]

            out_tokens = [idx, next_token]
            if eos_token_id is not None and (next_token == eos_token_id).all():
                return torch.cat(out_tokens, dim=1)

            # Giai đoạn 2: Decoding từng token tiếp theo (T=1)
            for _ in range(max_new_tokens - 1):
                start_pos = kv_cache.get_seq_length(0)
                if start_pos >= self.config.block_size:
                    break

                logits = self(next_token, kv_cache=kv_cache, start_pos=start_pos)
                next_token_logits = logits[:, -1, :]
                next_token = Sampler.sample_next_token(
                    next_token_logits, temperature=temperature, top_k=top_k, top_p=top_p
                )
                out_tokens.append(next_token)

                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

            return torch.cat(out_tokens, dim=1)
        else:
            # Chế độ không dùng KV Cache (Naive Autoregressive Decoding)
            out = idx
            for _ in range(max_new_tokens):
                # Cắt chuỗi nếu vượt quá context length (block_size)
                idx_cond = out if out.size(1) <= self.config.block_size else out[:, -self.config.block_size:]
                logits = self(idx_cond)
                next_token_logits = logits[:, -1, :]
                next_token = Sampler.sample_next_token(
                    next_token_logits, temperature=temperature, top_k=top_k, top_p=top_p
                )
                out = torch.cat([out, next_token], dim=1)

                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

            return out
