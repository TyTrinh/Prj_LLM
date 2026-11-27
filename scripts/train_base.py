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
from src.utils.checkpoint import CheckpointManager, CheckpointRollbackManager
from src.utils.scheduler import get_wsd_lr


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train LLM Base Model with WSD Scheduler")
    parser.add_argument("--epochs", type=int, default=3, help="Số epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate cực đại (eta_max)")
    parser.add_argument("--min_lr", type=float, default=5e-5, help="Learning rate tối thiểu (0.1 eta_max)")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Tỷ lệ warmup (3% tổng số bước)")
    parser.add_argument("--decay_ratio", type=float, default=0.10, help="Tỷ lệ decay (10% cuối cùng)")
    parser.add_argument("--decay_type", type=str, default="cosine", choices=["cosine", "linear", "exponential"])
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

    # Khởi tạo CheckpointRollbackManager giám sát ứng cứu tự động
    rollback_manager = CheckpointRollbackManager(
        checkpoint_dir=args.checkpoint_dir,
        threshold_factor=3.0,
        lr_reduction_ratio=0.20,
        recovery_steps=50,
    )
    if candidate_ckpt:
        rollback_manager.register_safe_checkpoint(candidate_ckpt)

    print(f"Tokens/step: {tokens_per_step:,} | Total Steps: {total_steps:,}")

    step = start_step
    criterion = nn.CrossEntropyLoss()
    model.train()

    while step < total_steps:
        optimizer.zero_grad()
        loss_accum = 0.0

        # Cập nhật Learning Rate theo WSD Scheduler (có tính hệ số hồi phục nếu đang rollback)
        base_lr = get_wsd_lr(
            step=step,
            max_steps=total_steps,
            max_lr=args.lr,
            min_lr=args.min_lr,
            warmup_ratio=args.warmup_ratio,
            decay_ratio=args.decay_ratio,
            decay_type=args.decay_type,
        )
        effective_lr = base_lr * rollback_manager.current_lr_multiplier
        for param_group in optimizer.param_groups:
            param_group["lr"] = effective_lr

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

        # Unscale để tính chính xác norm đạo hàm
        if use_amp:
            scaler.unscale_(optimizer)

        # 1. Đo lường norm gradient để phát hiện bất thường
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0 * rollback_manager.current_clip_multiplier,
        ).item()

        # 2. Kiểm tra ngưỡng bất thường: nếu nổ gradient norm hoặc NaN/Inf
        if rollback_manager.check_gradient_anomaly(grad_norm) or math.isnan(loss_accum):
            print(f"⚠️ Phát hiện bất thường tại Step {step}: GradNorm={grad_norm:.4f}, Loss={loss_accum:.4f}")
            # Kích hoạt Rollback về checkpoint an toàn gần nhất & bỏ qua batch này
            rollback_manager.trigger_rollback(
                model=model,
                optimizer=optimizer,
                scaler=scaler if use_amp else None,
                device=args.device,
                failed_step=step,
                reason=f"Gradient norm anomaly ({grad_norm:.2f})",
            )
            optimizer.zero_grad()
            continue

        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Cập nhật tiến trình hồi phục sau sự cố
        rollback_manager.step_recovery(optimizer)
        step += 1

        if step % 50 == 0:
            print(f"Step {step}/{total_steps} | Loss: {loss_accum:.4f} | LR: {effective_lr:.6f} | GradNorm: {grad_norm:.2f}")

        # Lưu checkpoint định kỳ mỗi 500 steps để sẵn sàng Resume / Rollback
        if step % 500 == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step}.pt")
            CheckpointManager.save_checkpoint(
                filepath=ckpt_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler if use_amp else None,
                step=step,
                best_val_loss=loss_accum,
            )
            rollback_manager.register_safe_checkpoint(ckpt_path)
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
