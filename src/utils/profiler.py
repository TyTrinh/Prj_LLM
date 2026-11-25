"""
==============================================================================
FILE: src/utils/profiler.py
MODULE: src.utils.profiler
RESPONSIBILITY:
    - Đơn trách nhiệm: Đo đạc thực nghiệm định lượng hiệu năng KV Cache (Profiling).
    - Đo chính xác 4 chỉ số bắt buộc:
      1. Tốc độ sinh từ (Tokens/giây)
      2. First Token Latency (TTFT - Time To First Token)
      3. Next Token Latency (TPOT - Time Per Output Token)
      4. Mức chiếm dụng VRAM đỉnh (Peak VRAM qua `torch.cuda.max_memory_allocated()`)
    - Hỗ trợ đo ở cả 2 trạng thái: Bật KV Cache vs Tắt KV Cache.
    - Đảm bảo tính chuẩn xác qua `torch.cuda.synchronize()` và warm-up loop.
INPUT:
    - model: GPTModel
    - prompt_length (L): int ∈ {128, 512, 1024, 2048}
    - generate_tokens: int (mặc định 64)
    - use_kv_cache: bool
    - device: str ("cuda" hoặc "cpu")
OUTPUT:
    - ProfilingResult dataclass chứa 4 chỉ số đo đạc thực tế.
==============================================================================
"""

import time
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class ProfilingResult:
    context_length: int
    use_kv_cache: bool
    tokens_per_sec: float
    ttft_ms: float
    tpot_ms: float
    peak_vram_mb: float


class KVCacheProfiler:
    """Đơn trách nhiệm: Benchmark định lượng hiệu năng và bộ nhớ VRAM."""

    @staticmethod
    def measure(
        model,
        prompt_length: int,
        generate_tokens: int = 64,
        use_kv_cache: bool = True,
        device: str = "cpu",
        warmup: int = 2,
    ) -> ProfilingResult:
        """Đo đạc chính xác với synchronization và memory tracking."""
        model.eval()
        is_cuda = device.startswith("cuda") and torch.cuda.is_available()

        # Tạo prompt ngẫu nhiên với độ dài L
        vocab_size = getattr(model.config, "vocab_size", 1000)
        prompt = torch.randint(0, vocab_size, (1, prompt_length), device=device)

        # 1. Warm-up runs để ổn định GPU kernel / CUDA context
        with torch.no_grad():
            for _ in range(warmup):
                _ = model.generate(
                    prompt[:, :min(16, prompt_length)],
                    max_new_tokens=4,
                    use_kv_cache=use_kv_cache,
                    temperature=0.0,
                )
        if is_cuda:
            torch.cuda.synchronize()

        # 2. Đo VRAM và thời gian
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # Đo TTFT (Time To First Token) - thời gian sinh token đầu tiên
        t_start = time.perf_counter()
        with torch.no_grad():
            first_out = model.generate(
                prompt,
                max_new_tokens=1,
                use_kv_cache=use_kv_cache,
                temperature=0.0,
            )
        if is_cuda:
            torch.cuda.synchronize()
        ttft = (time.perf_counter() - t_start) * 1000.0  # ms

        # Đo toàn bộ quá trình sinh N tokens (để tính TPOT và Tokens/s)
        if is_cuda:
            torch.cuda.synchronize()
        t_gen_start = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(
                prompt,
                max_new_tokens=generate_tokens,
                use_kv_cache=use_kv_cache,
                temperature=0.0,
            )
        if is_cuda:
            torch.cuda.synchronize()
            peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        else:
            peak_vram = 0.0

        total_time = time.perf_counter() - t_gen_start
        tokens_per_sec = generate_tokens / max(total_time, 1e-6)
        # TPOT (Time Per Output Token): thời gian trung bình cho mỗi token sau token đầu
        tpot = (total_time * 1000.0) / max(generate_tokens, 1)  # ms/token

        return ProfilingResult(
            context_length=prompt_length,
            use_kv_cache=use_kv_cache,
            tokens_per_sec=round(tokens_per_sec, 2),
            ttft_ms=round(ttft, 2),
            tpot_ms=round(tpot, 2),
            peak_vram_mb=round(peak_vram, 2),
        )
