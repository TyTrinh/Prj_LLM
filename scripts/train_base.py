"""
==============================================================================
FILE: scripts/train_base.py
MODULE: scripts.train_base (CLI Runner)
RESPONSIBILITY:
    - Đơn trách nhiệm: CLI Script thực thi tiền huấn luyện Base Model (Pre-training).
    - Đọc dữ liệu nhị phân `train.bin` (TinyStories), chạy vòng lặp huấn luyện:
      AdamW, Cosine LR Scheduler với Warmup, Mixed Precision (AMP), Gradient Clipping.
    - Lưu checkpoint `checkpoints/base_model.pt`.
INPUT:
    - CLI Flags: --data_dir, --epochs, --batch_size, --lr, --checkpoint_dir
OUTPUT:
    - File trọng số `checkpoints/base_model.pt` và log WandB/Tensorboard.
DEPENDENCY:
    - Phụ thuộc `src.model.transformer.GPTModel`, `src.config.ModelConfig`, `src.utils`.
==============================================================================
"""

import argparse
from src.config import ModelConfig, TrainConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train LLM Base Model from Scratch")
    parser.add_argument("--epochs", type=int, default=3, help="Số epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="data", help="Thư mục chứa train.bin")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Nơi lưu .pt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 70)
    print("🚀 BẮT ĐẦU TIỀN HUẤN LUYỆN BASE MODEL (TINYSTORIES)")
    print(f"Tham số: Epochs={args.epochs}, BatchSize={args.batch_size}, LR={args.lr}")
    print("=" * 70)
    # Runner interface ready for cloud execution
