"""
==============================================================================
FILE: tests/test_profiler.py
RESPONSIBILITY:
    - Kiểm thử unit test cho KVCacheProfiler trên CPU để đảm bảo logic tính toán
      các chỉ số (Tokens/s, TTFT, TPOT, VRAM) không bị lỗi runtime.
==============================================================================
"""

import torch
from src.config import ModelConfig
from src.model import GPTModel
from src.utils.profiler import KVCacheProfiler, ProfilingResult


def test_profiler_measure_cpu():
    cfg = ModelConfig(
        vocab_size=64,
        block_size=64,
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ffn=64,
    )
    model = GPTModel(cfg)

    # Đo với KV Cache
    res_kv = KVCacheProfiler.measure(
        model=model,
        prompt_length=16,
        generate_tokens=4,
        use_kv_cache=True,
        device="cpu",
        warmup=1,
    )
    assert isinstance(res_kv, ProfilingResult)
    assert res_kv.context_length == 16
    assert res_kv.use_kv_cache is True
    assert res_kv.tokens_per_sec > 0
    assert res_kv.ttft_ms >= 0
    assert res_kv.tpot_ms >= 0

    # Đo không dùng KV Cache
    res_no_kv = KVCacheProfiler.measure(
        model=model,
        prompt_length=16,
        generate_tokens=4,
        use_kv_cache=False,
        device="cpu",
        warmup=1,
    )
    assert isinstance(res_no_kv, ProfilingResult)
    assert res_no_kv.use_kv_cache is False
