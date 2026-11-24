"""
==============================================================================
FILE: scripts/train_dpo.py
MODULE: scripts.train_dpo (CLI Runner)
RESPONSIBILITY:
    - Đơn trách nhiệm: CLI Script thực thi căn chỉnh hành vi không cần Reward Model (DPO).
    - Nạp mô hình đã gắn `sft_lora.pt`.
    - Sử dụng kỹ thuật tiết kiệm 50% VRAM: Tắt LoRA dưới `torch.no_grad()` để lấy logps của
      mô hình tham chiếu $\pi_{\text{ref}}$, bật LoRA để lấy logps của policy $\pi_\theta$.
    - Tối ưu hàm mất mát DPO trên tập `Anthropic/hh-rlhf` và lưu `checkpoints/dpo_lora.pt`.
INPUT:
    - CLI Flags: --sft_model, --beta, --length_norm, --lr
OUTPUT:
    - Adapter trọng số `checkpoints/dpo_lora.pt`.
DEPENDENCY:
    - Phụ thuộc `src.alignment.dpo_loss`, `src.peft.lora_manager`.
==============================================================================
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện DPO Alignment từ số 0")
    parser.add_argument("--sft_model", type=str, default="checkpoints/sft_lora.pt")
    parser.add_argument("--beta", type=float, default=0.1, help="Hệ số nhiệt độ DPO beta")
    parser.add_argument("--length_norm", action="store_true", help="Kích hoạt Length-Normalized DPO")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate cực nhỏ cho DPO")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 70)
    print("🚀 BẮT ĐẦU CĂN CHỈNH HÀNH VI DPO (ANTHROPIC HH-RLHF)")
    print(f"Beta={args.beta}, LengthNormalization={args.length_norm}, LR={args.lr}")
    print("=" * 70)
