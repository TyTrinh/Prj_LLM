"""
==============================================================================
FILE: tests/test_transformer_core.py
RESPONSIBILITY:
    - Kiểm thử chuyên sâu các khối kiến trúc Transformer:
      1. RMSNorm: Chuẩn hóa invariant scaling và tính đúng đắn.
      2. RoPE: Tính toán embedding góc quay, tính chất đối xứng và độ dài tùy ý.
      3. CausalSelfAttention: Causal mask chặn nhìn trước tương lai, multi-head output shape.
      4. SwiGLU / FeedForward: Biến đổi kích thước kênh và phi tuyến tính.
      5. GPTModel & Weight Tying: W_head chia sẻ cùng vùng nhớ với tok_embeddings.weight.
      6. Trọng số khởi tạo: Gaussian N(0, 0.02) và giảm chuẩn tại residual connections.
==============================================================================
"""

import math
import torch
import torch.nn as nn
from src.config import ModelConfig
from src.model import (
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    FeedForward,
    CausalSelfAttention,
    TransformerBlock,
    GPTModel,
    apply_rotary_pos_emb,
)


def test_rmsnorm():
    """Kiểm tra RMSNorm chuẩn hóa đúng và bảo toàn shape."""
    dim = 64
    norm = RMSNorm(dim=dim, eps=1e-5)
    x = torch.randn(2, 10, dim) * 5.0 + 3.0  # Tensor có scale lớn

    out = norm(x)
    assert out.shape == x.shape, f"RMSNorm output shape mismatch: {out.shape} vs {x.shape}"

    # Kiểm tra phương sai sau chuẩn hóa (trước khi nhân gamma weight) gần bằng 1
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-5)
    manual_norm = (x / rms) * norm.weight
    assert torch.allclose(out, manual_norm, atol=1e-5), "RMSNorm computation mismatch!"


def test_rope_rotary_embedding():
    """Kiểm tra RoPE tạo góc quay đúng và áp dụng lên Q, K."""
    head_dim = 32
    seq_len = 8
    rope = RotaryEmbedding(dim=head_dim, max_seq_len=64, base=10000.0)

    cos, sin = rope(seq_len=seq_len, device=torch.device("cpu"), start_pos=0)
    assert cos.shape == (1, 1, seq_len, head_dim)
    assert sin.shape == (1, 1, seq_len, head_dim)

    # Q, K shape [B, n_heads, seq_len, head_dim]
    B, H = 2, 4
    q = torch.randn(B, H, seq_len, head_dim)
    k = torch.randn(B, H, seq_len, head_dim)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape

    # Kiểm tra tính chất bảo toàn độ dài vector (norm) trong không gian 2D xoay
    # Mỗi cặp 2D (x1, x2) khi xoay bảo toàn x1^2 + x2^2
    q_norm_before = torch.norm(q, dim=-1)
    q_norm_after = torch.norm(q_rot, dim=-1)
    assert torch.allclose(q_norm_before, q_norm_after, atol=1e-4), "RoPE must preserve vector L2 norm!"


def test_causal_self_attention_masking():
    """Kiểm tra CausalSelfAttention chặn thông tin tương lai (Causal Mask)."""
    d_model = 64
    n_heads = 4
    seq_len = 6
    attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, max_seq_len=64)
    attn.eval()

    # Tạo 2 chuỗi x1 và x2 giống nhau ở 3 token đầu, khác nhau ở các token sau
    x1 = torch.randn(1, seq_len, d_model)
    x2 = x1.clone()
    x2[:, 3:, :] = torch.randn(1, seq_len - 3, d_model)  # Thay đổi các token tương lai

    with torch.no_grad():
        out1 = attn(x1)
        out2 = attn(x2)

    # Do tính chất Causal (Masked Attention), các token ở vị trí 0, 1, 2 của x1 và x2
    # KHÔNG được phép bị ảnh hưởng bởi sự thay đổi của token 3, 4, 5
    assert torch.allclose(out1[:, :3, :], out2[:, :3, :], atol=1e-5), (
        "Causal Attention leaked future tokens! Out at t <= 2 should be identical."
    )


def test_feedforward_swiglu():
    """Kiểm tra khối SwiGLU FeedForward tính toán đúng shape."""
    d_model = 64
    d_ffn = 128
    ffn = FeedForward(d_model=d_model, d_ffn=d_ffn, use_swiglu=True)

    x = torch.randn(2, 5, d_model)
    out = ffn(x)
    assert out.shape == (2, 5, d_model), f"FFN output shape mismatch: {out.shape}"


def test_weight_tying():
    """Kiểm tra tính năng Weight Tying chia sẻ cùng địa chỉ bộ nhớ giữa Embedding và LM Head."""
    cfg = ModelConfig(
        vocab_size=100,
        d_model=32,
        n_layers=1,
        n_heads=2,
        block_size=16,
        tie_weights=True,
    )
    model = GPTModel(cfg)

    # 1. Kiểm tra tham chiếu cùng đối tượng Parameter
    assert model.lm_head.weight is model.tok_embeddings.weight, (
        "Weight Tying failed: lm_head.weight and tok_embeddings.weight are not identical!"
    )
    # 2. Kiểm tra data_ptr trùng nhau
    assert model.lm_head.weight.data_ptr() == model.tok_embeddings.weight.data_ptr(), (
        "Weight Tying failed: data pointers differ!"
    )


def test_gpt_model_forward_and_init():
    """Kiểm tra toàn bộ forward pass của GPTModel và độ lệch chuẩn khởi tạo trọng số."""
    cfg = ModelConfig(
        vocab_size=256,
        d_model=64,
        n_layers=4,
        n_heads=4,
        block_size=32,
        tie_weights=True,
    )
    model = GPTModel(cfg)
    model.eval()

    B, T = 2, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T))

    with torch.no_grad():
        logits = model(idx)

    assert logits.shape == (B, T, cfg.vocab_size), (
        f"Expected logits shape {(B, T, cfg.vocab_size)}, got {logits.shape}"
    )

    # Kiểm tra khởi tạo trọng số residual projections có std xấp xỉ 0.02 / sqrt(2 * n_layers)
    expected_residual_std = 0.02 / math.sqrt(2 * cfg.n_layers)
    out_proj_std = model.blocks[0].attn.out_proj.weight.std().item()
    w_down_std = model.blocks[0].ffn.net.w_down.weight.std().item()

    assert abs(out_proj_std - expected_residual_std) < 0.02, (
        f"out_proj std ({out_proj_std}) deviated too much from expected ({expected_residual_std})"
    )
    assert abs(w_down_std - expected_residual_std) < 0.02, (
        f"w_down std ({w_down_std}) deviated too much from expected ({expected_residual_std})"
    )
