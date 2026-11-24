"""
==============================================================================
FILE: scripts/train_sft.py
MODULE: scripts.train_sft (CLI Runner)
RESPONSIBILITY:
    - Đơn trách nhiệm: CLI Script thực thi tinh chỉnh chỉ dẫn (Supervised Fine-Tuning - SFT).
    - Nạp `base_model.pt`, tiêm adapter LoRA vào các ma trận chiếu `q_proj, v_proj`,
      đóng băng trọng số gốc và huấn luyện trên tập `Alpaca-Cleaned` có mặt nạ bỏ qua prompt (-100).
    - Lưu adapter LoRA: `checkpoints/sft_lora.pt`.
INPUT:
    - CLI Flags: --base_model, --dataset, --r, --alpha, --lr
OUTPUT:
    - Adapter trọng số `checkpoints/sft_lora.pt`.
DEPENDENCY:
    - Phụ thuộc `src.peft.lora_manager.LoRAManager`, `src.model.transformer.GPTModel`.
==============================================================================
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện SFT LoRA từ số 0")
    parser.add_argument("--base_model", type=str, default="checkpoints/base_model.pt")
    parser.add_argument("--r", type=int, default=8, help="Hạng rank của LoRA")
    parser.add_argument("--alpha", type=float, default=16.0, help="Hệ số alpha LoRA")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate cho adapter")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 70)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN SFT LORA (ALPACA-CLEANED)")
    print(f"Base: {args.base_model} | LoRA Rank={args.r}, Alpha={args.alpha}")
    print("=" * 70)
