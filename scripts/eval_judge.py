"""
==============================================================================
FILE: scripts/eval_judge.py
MODULE: scripts.eval_judge (CLI Runner & Evaluation)
RESPONSIBILITY:
    - Đơn trách nhiệm: CLI Script đánh giá định lượng bằng phương pháp LLM-as-a-Judge.
    - Gọi API (GPT-4o-mini hoặc Claude-3.5-Haiku) để làm trọng tài chấm đối đầu:
      So sánh cặp phản hồi (SFT Model vs DPO Model) trên subset benchmark MT-Bench
      theo tiêu chí 3H (Helpful, Honest, Harmless).
    - Tính toán tỷ lệ thắng (**Win Rate %**) và xuất báo cáo kết quả tự động.
INPUT:
    - CLI Flags: --benchmark, --judge_model, --api_key
OUTPUT:
    - File báo cáo markdown `docs/eval_results.md`.
DEPENDENCY:
    - Phụ thuộc vào thư viện `openai` hoặc SDK gọi API trọng tài.
==============================================================================
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Pairwise Evaluation")
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--benchmark", type=str, default="mt_bench_subset.jsonl")
    parser.add_argument("--output_report", type=str, default="docs/eval_results.md")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 70)
    print("⚖️ BẮT ĐẦU ĐÁNH GIÁ CHẤM CẶP LLM-AS-A-JUDGE")
    print(f"Trọng tài: {args.judge_model} | Benchmark: {args.benchmark}")
    print("=" * 70)
