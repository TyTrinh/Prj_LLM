"""
==============================================================================
FILE: scripts/train_base.py
MODULE: scripts.train_base (CLI Runner)
RESPONSIBILITY:
    - Đơn trách nhiệm: CLI Script thực thi tiền huấn luyện Base Model (Pre-training).
    - Hỗ trợ cơ chế:
      1. Gradient Accumulation
      2. Mixed Precision AMP (float16 / bfloat16)
      3. Cosine Learning Rate Scheduler with Warmup
      4. Auto-Resume từ checkpoint gần nhất (không bị mất tiến trình)
      5. Lưu `base_model.pt` và `last_checkpoint.pt`
==============================================================================
"""

import argparse
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
from src.config import ModelConfig, TrainConfig
from src.model import GPTModel
from src.utils.checkpoint import CheckpointManager


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train LLM Base Model from Scratch")
    parser.add_argument("--epochs", type=int, default=3, help="Số epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate cực đại")
    parser.add_argument("--min_lr", type=float, default=5e-5, help="Learning rate tối thiểu")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Số bước warmup")
    parser.add_argument("--grad_accum", type=int, default=4, help="Số bước tích lũy gradient")
    parser.add_argument("--data_dir", type=str, default="data", help="Thư mục chứa train.bin, val.bin")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Nơi lưu .pt")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục train từ checkpoint gần nhất")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Tính learning rate theo Cosine Decay Scheduler with Linear Warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / max(warmup_steps, 1)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def get_batch(data: np.memmap, batch_size: int, block_size: int, device: str):
    """Lấy ngẫu nhiên batch [B, T] từ binary memmap."""
    ix = np.random.randint(0, len(data) - block_size, size=batch_size)
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def main():
    args = parse_args()
    print("=" * 70)
    print("🚀 BẮT ĐẦU TIỀN HUẤN LUYỆN BASE MODEL (TINYSTORIES)")
    print(f"Device: {args.device} | BatchSize: {args.batch_size} | GradAccum: {args.grad_accum}")
    print("=" * 70)

    model_cfg = ModelConfig()
    model = GPTModel(model_cfg).to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Tổng số tham số mô hình: {total_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    use_amp = args.device.startswith("cuda") and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_step = 0
    start_epoch = 0
    best_val_loss = float("inf")

    last_ckpt_path = os.path.join(args.checkpoint_dir, "last_checkpoint.pt")
    if args.resume and os.path.exists(last_ckpt_path):
        print(f"🔄 Nạp checkpoint để Resume: {last_ckpt_path}")
        meta = CheckpointManager.load_checkpoint(
            last_ckpt_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler if use_amp else None,
            device=args.device,
        )
        start_step = meta.get("step", 0)
        start_epoch = meta.get("epoch", 0)
        best_val_loss = meta.get("best_val_loss", float("inf"))
        print(f"✅ Đã phục hồi tại Step: {start_step}, Epoch: {start_epoch}, Best Loss: {best_val_loss:.4f}")

    train_bin = os.path.join(args.data_dir, "train.bin")
    val_bin = os.path.join(args.data_dir, "val.bin")

    if not os.path.exists(train_bin):
        print(f"⚠️ Chưa tìm thấy {train_bin}. Vui lòng tạo dữ liệu trước.")
        return

    train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_bin, dtype=np.uint16, mode="r") if os.path.exists(val_bin) else None

    tokens_per_step = args.batch_size * args.grad_accum * model_cfg.block_size
    steps_per_epoch = max(len(train_data) // tokens_per_step, 1)
    total_steps = args.epochs * steps_per_epoch

    print(f"Tokens/step: {tokens_per_step:,} | Total Steps: {total_steps:,}")

    step = start_step
    criterion = nn.CrossEntropyLoss()
    model.train()

    while step < total_steps:
        optimizer.zero_grad()
        loss_accum = 0.0

        lr = get_lr(step, args.warmup_steps, total_steps, args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        for micro_step in range(args.grad_accum):
            x, y = get_batch(train_data, args.batch_size, model_cfg.block_size, args.device)
            with torch.amp.autocast(device_type="cuda" if "cuda" in args.device else "cpu", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / args.grad_accum

            loss_accum += loss.item()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        step += 1

        if step % 50 == 0:
            print(f"Step {step}/{total_steps} | Loss: {loss_accum:.4f} | LR: {lr:.6f}")

        # Lưu checkpoint định kỳ mỗi 500 steps để sẵn sàng Resume
        if step % 500 == 0:
            CheckpointManager.save_checkpoint(
                last_ckpt_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler if use_amp else None,
                step=step,
                epoch=step // steps_per_epoch,
                best_val_loss=best_val_loss,
                stage="pretrain",
            )

    # Lưu base_model.pt cuối cùng
    final_path = os.path.join(args.checkpoint_dir, "base_model.pt")
    CheckpointManager.save_base_model(model, final_path)
    print(f"🎉 Huấn luyện hoàn tất! Trọng số đã lưu tại: {final_path}")


if __name__ == "__main__":
    main()
