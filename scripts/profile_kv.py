"""
==============================================================================
FILE: scripts/profile_kv.py
MODULE: scripts.profile_kv (CLI Runner & Benchmark Generator)
RESPONSIBILITY:
    - Đơn trách nhiệm: CLI Script đo đạc thực nghiệm định lượng KV Cache theo yêu cầu bắt buộc:
      Đo tốc độ (Tokens/s), TTFT, TPOT, và Peak VRAM khi Bật vs Tắt KV Cache tại các mốc:
      $L \in \{128, 512, 1024, 2048\}$.
    - Tự động xuất file kết quả JSON và vẽ biểu đồ so sánh PNG lưu vào thư mục `docs/figures/`.
INPUT:
    - CLI Flags: --device, --output_fig, --output_json
OUTPUT:
    - File biểu đồ `docs/figures/kv_cache_benchmark.png` và file số liệu JSON.
DEPENDENCY:
    - Phụ thuộc `src.utils.profiler.KVCacheProfiler`, `src.model.transformer.GPTModel`.
==============================================================================
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Đo đạc thực nghiệm KV Cache")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị đo (cuda/cpu)")
    parser.add_argument("--output_fig", type=str, default="docs/figures/kv_cache_benchmark.png")
    parser.add_argument("--output_json", type=str, default="docs/figures/kv_cache_metrics.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 70)
    print("📊 BẮT ĐẦU ĐO ĐẠC ĐỊNH LƯỢNG KV CACHE (PROFILING)")
    print("Ngữ cảnh đo lường: L in [128, 512, 1024, 2048]")
    print(f"Lưu biểu đồ tại: {args.output_fig}")
    print("=" * 70)
