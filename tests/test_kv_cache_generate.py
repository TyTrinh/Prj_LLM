"""
==============================================================================
FILE: tests/test_kv_cache_generate.py
RESPONSIBILITY:
    - Kiểm thử KV Cache và cơ chế sinh tự hồi quy:
      1. KVCache update & reset: Kiểm tra cập nhật tensor kích thước và reset bộ nhớ.
      2. Attention Prefill vs Decode: Kiểm tra tương thích kết quả giữa dùng KV Cache và full attention.
      3. GPTModel generate:
         - Greedy Search (deterministic)
         - Temperature / Top-k / Top-p Sampling
         - Kết quả sinh từ giữa có KV Cache và không có KV Cache phải giống nhau hoàn toàn trong Greedy Search!
==============================================================================
"""

import pytest
import torch
from src.config import ModelConfig
from src.model import GPTModel, KVCache


def test_kv_cache_update_and_reset():
    """Kiểm tra cập nhật và reset tensor trong KVCache."""
    n_layers = 4
    cache = KVCache(n_layers=n_layers)

    B, H, head_dim = 2, 4, 32

    # Prefill 5 tokens
    k1 = torch.randn(B, H, 5, head_dim)
    v1 = torch.randn(B, H, 5, head_dim)
    out_k, out_v = cache.update(k1, v1, layer_idx=0)
    assert out_k.shape == (B, H, 5, head_dim)
    assert out_v.shape == (B, H, 5, head_dim)
    assert cache.get_seq_length(0) == 5

    # Decode 1 token
    k2 = torch.randn(B, H, 1, head_dim)
    v2 = torch.randn(B, H, 1, head_dim)
    out_k, out_v = cache.update(k2, v2, layer_idx=0)
    assert out_k.shape == (B, H, 6, head_dim)
    assert out_v.shape == (B, H, 6, head_dim)
    assert cache.get_seq_length(0) == 6

    # Reset
    cache.reset()
    assert cache.get_seq_length(0) == 0
    assert cache.k_cache[0] is None
    assert cache.v_cache[0] is None


def test_kv_cache_logits_equivalence_with_naive():
    """Kiểm tra forward pass với KV Cache và full sequence forward cho ra logits tương đương tại token tiếp theo."""
    torch.manual_seed(42)
    cfg = ModelConfig(
        vocab_size=128,
        block_size=64,
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ffn=64,
    )
    model = GPTModel(cfg)
    model.eval()

    prompt_len = 5
    prompt = torch.randint(0, cfg.vocab_size, (1, prompt_len))
    next_tok = torch.tensor([[42]])
    full_seq = torch.cat([prompt, next_tok], dim=1)

    # 1. Full sequence forward không cache
    with torch.no_grad():
        full_logits = model(full_seq)  # [1, 6, vocab_size]
        expected_next_logits = full_logits[:, -1, :]  # Logits tại vị trí token 42

    # 2. KV Cache: Prefill prompt rồi decode next_tok
    cache = KVCache(n_layers=cfg.n_layers)
    with torch.no_grad():
        _ = model(prompt, kv_cache=cache, start_pos=0)
        cache_logits = model(next_tok, kv_cache=cache, start_pos=prompt_len)  # [1, 1, vocab_size]
        actual_next_logits = cache_logits[:, 0, :]

    # Logits của token mới khi dùng KV Cache phải khớp với Full Attention
    assert torch.allclose(expected_next_logits, actual_next_logits, atol=1e-5), (
        f"Logits mismatch! Max diff: {(expected_next_logits - actual_next_logits).abs().max().item()}"
    )


def test_gpt_model_generate_greedy_equivalence():
    """Kiểm tra generate() với Greedy search (temp=0): có KV Cache và không có KV Cache phải ra output y hệt nhau."""
    torch.manual_seed(42)
    cfg = ModelConfig(
        vocab_size=128,
        block_size=64,
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ffn=64,
    )
    model = GPTModel(cfg)
    model.eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    max_new_tokens = 6

    # Sinh bằng KV Cache
    gen_cached = model.generate(
        prompt.clone(),
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        use_kv_cache=True,
    )

    # Sinh không có KV Cache
    gen_naive = model.generate(
        prompt.clone(),
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        use_kv_cache=False,
    )

    assert gen_cached.shape == (1, 4 + max_new_tokens)
    assert gen_naive.shape == (1, 4 + max_new_tokens)
    assert torch.equal(gen_cached, gen_naive), (
        f"Greedy outputs mismatch!\nWith cache: {gen_cached}\nWithout cache: {gen_naive}"
    )


def test_gpt_model_generate_sampling():
    """Kiểm tra generate() hoạt động trơn tru với các chế độ sampling (temperature, top_k, top_p)."""
    torch.manual_seed(42)
    cfg = ModelConfig(
        vocab_size=128,
        block_size=64,
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ffn=64,
    )
    model = GPTModel(cfg)
    model.eval()

    prompt = torch.randint(0, cfg.vocab_size, (2, 4))  # Batch size 2
    max_new_tokens = 5

    # 1. Temperature sampling
    out_temp = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        use_kv_cache=True,
    )
    assert out_temp.shape == (2, 4 + max_new_tokens)

    # 2. Top-k + Top-p sampling
    out_topk_topp = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_k=20,
        top_p=0.9,
        use_kv_cache=True,
    )
    assert out_topk_topp.shape == (2, 4 + max_new_tokens)

    # 3. Early stopping with eos_token_id
    eos_id = 99
    # Giả lập model sinh ra eos_id hoặc chạy test không crash
    out_eos = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        eos_token_id=eos_id,
        use_kv_cache=True,
    )
    assert out_eos.size(1) <= 4 + max_new_tokens
