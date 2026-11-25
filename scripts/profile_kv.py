"""
==============================================================================
FILE: scripts/profile_kv.py
MODULE: scripts.profile_kv (CLI Runner & Benchmark Generator)
RESPONSIBILITY:
    - Đơn trách nhiệm: CLI Script đo đạc thực nghiệm định lượng KV Cache theo yêu cầu bắt buộc:
      Đo tốc độ (Tokens/s), TTFT, TPOT, và Peak VRAM khi Bật vs Tắt KV Cache tại 4 mốc:
      $L \in \{128, 512, 1024, 2048\}$.
    - Tự động xuất file kết quả JSON và vẽ biểu đồ so sánh PNG lưu vào thư mục `docs/figures/`.
==============================================================================
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import torch
from src.config import ModelConfig
from src.model import GPTModel
from src.utils.profiler import KVCacheProfiler


def parse_args():
    parser = argparse.ArgumentParser(description="Đo đạc thực nghiệm KV Cache")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_fig", type=str, default="docs/figures/kv_cache_benchmark.png")
    parser.add_argument("--output_json", type=str, default="docs/figures/kv_cache_metrics.json")
    parser.add_argument("--generate_tokens", type=int, default=64)
    return parser.parse_args()


def plot_benchmark_results(results, output_path: str):
    """Vẽ 4 biểu đồ so sánh Bật KV Cache vs Tắt KV Cache."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    lengths = sorted(list({r["context_length"] for r in results}))
    metrics = [
        ("tokens_per_sec", "Tốc độ sinh (Tokens/s) - Càng cao càng tốt", "Tokens/s"),
        ("ttft_ms", "First Token Latency - TTFT (ms) - Càng thấp càng tốt", "ms"),
        ("tpot_ms", "Next Token Latency - TPOT (ms) - Càng thấp càng tốt", "ms"),
        ("peak_vram_mb", "Peak VRAM (MB)", "MB"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric_key, title, ylabel) in enumerate(metrics):
        ax = axes[idx]
        with_kv = [next(r[metric_key] for r in results if r["context_length"] == L and r["use_kv_cache"]) for L in lengths]
        without_kv = [next(r[metric_key] for r in results if r["context_length"] == L and not r["use_kv_cache"]) for L in lengths]

        x_indices = range(len(lengths))
        width = 0.35

        ax.bar([x - width / 2 for x in x_indices], with_kv, width=width, label="Bật KV Cache", color="#2ca02c")
        ax.bar([x + width / 2 for x in x_indices], without_kv, width=width, label="Tắt KV Cache", color="#d62728")

        ax.set_xticks(x_indices)
        ax.set_xticklabels([f"L={L}" for L in lengths])
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.suptitle("ĐO ĐẠC THỰC NGHIỆM ĐỊNH LƯỢNG HIỆU NĂNG KV CACHE", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"📊 Đã lưu biểu đồ so sánh tại: {output_path}")


def main():
    args = parse_args()
    print("=" * 70)
    print("📊 BẮT ĐẦU ĐO ĐẠC ĐỊNH LƯỢNG KV CACHE (PROFILING)")
    print(f"Device: {args.device} | Sinh {args.generate_tokens} tokens")
    print("=" * 70)

    cfg = ModelConfig(block_size=2048)
    model = GPTModel(cfg).to(args.device)

    context_lengths = [128, 512, 1024, 2048]
    all_results = []

    for L in context_lengths:
        print(f"\n--- Đang đo mốc Context Length L = {L} ---")
        # 1. Bật KV Cache
        res_on = KVCacheProfiler.measure(
            model=model,
            prompt_length=L,
            generate_tokens=args.generate_tokens,
            use_kv_cache=True,
            device=args.device,
        )
        print(f"  [BẬT KV CACHE]  Tokens/s: {res_on.tokens_per_sec} | TTFT: {res_on.ttft_ms}ms | TPOT: {res_on.tpot_ms}ms | VRAM: {res_on.peak_vram_mb}MB")
        all_results.append(vars(res_on))

        # 2. Tắt KV Cache
        res_off = KVCacheProfiler.measure(
            model=model,
            prompt_length=L,
            generate_tokens=args.generate_tokens,
            use_kv_cache=False,
            device=args.device,
        )
        print(f"  [TẮT KV CACHE]  Tokens/s: {res_off.tokens_per_sec} | TTFT: {res_off.ttft_ms}ms | TPOT: {res_off.tpot_ms}ms | VRAM: {res_off.peak_vram_mb}MB")
        all_results.append(vars(res_off))

    # Lưu JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n💾 Đã lưu số liệu JSON tại: {args.output_json}")

    # Vẽ biểu đồ
    plot_benchmark_results(all_results, args.output_fig)


if __name__ == "__main__":
    main()
