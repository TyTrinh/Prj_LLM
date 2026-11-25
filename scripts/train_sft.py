"""
==============================================================================
FILE: scripts/train_sft.py
MODULE: scripts.train_sft (CLI Runner)
RESPONSIBILITY:
    - Đơn trách nhiệm: CLI Script thực thi tinh chỉnh chỉ dẫn (Supervised Fine-Tuning - SFT).
    - Nạp `base_model.pt`, tiêm adapter LoRA vào các ma trận chiếu `q_proj, v_proj`,
      đóng băng trọng số gốc và huấn luyện trên tập `Alpaca-Cleaned` có mặt nạ bỏ qua prompt (-100).
    - Áp dụng chuẩn xác cơ chế Shift Logits [..., :-1, :] và Labels [..., 1:] cho Next-Token Prediction.
    - Hỗ trợ Train/Val split, Validation loop để phát hiện Overfitting, cố định random seed.
    - Lưu adapter LoRA tốt nhất: `checkpoints/sft_lora.pt`.
INPUT:
    - CLI Flags: --base_model, --dataset, --r, --alpha, --lr, --batch_size, --epochs, --max_seq_len, --val_ratio, --seed
OUTPUT:
    - Adapter trọng số `checkpoints/sft_lora.pt`.
DEPENDENCY:
    - Phụ thuộc `src.peft.lora_manager.LoRAManager`, `src.model.transformer.GPTModel`,
      `src.alignment.sft_dataset`, `src.tokenizer.bpe.BytePairTokenizer`, `src.utils.seed`.
==============================================================================
"""

import argparse
import math
import os
import torch
from torch.utils.data import DataLoader, random_split

from src.config import ModelConfig
from src.model.transformer import GPTModel
from src.peft.lora_manager import LoRAManager
from src.alignment.sft_dataset import (
    AlpacaSFTDataset,
    SFTDataCollator,
    compute_sft_loss,
)
from src.tokenizer.bpe import BytePairTokenizer
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện SFT LoRA từ số 0")
    parser.add_argument("--base_model", type=str, default="checkpoints/base_model.pt", help="Đường dẫn base model checkpoint")
    parser.add_argument("--output_lora", type=str, default="checkpoints/sft_lora.pt", help="Đường dẫn lưu adapter LoRA")
    parser.add_argument("--data_file", type=str, default="data/alpaca_cleaned.json", help="File dữ liệu Alpaca JSON")
    parser.add_argument("--vocab_file", type=str, default="checkpoints/vocab.json", help="Đường dẫn vocab.json")
    parser.add_argument("--merges_file", type=str, default="checkpoints/merges.txt", help="Đường dẫn merges.txt")
    parser.add_argument("--r", type=int, default=8, help="Hạng rank của LoRA")
    parser.add_argument("--alpha", type=float, default=16.0, help="Hệ số alpha LoRA")
    parser.add_argument("--dropout", type=float, default=0.05, help="Tỷ lệ dropout của LoRA")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate cho adapter")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size huấn luyện")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=2, help="Số epochs huấn luyện SFT (khuyến nghị 1-2 epochs)")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Độ dài ngữ cảnh tối đa")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Tỷ lệ tập validation (mặc định 5%)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed tái lập kết quả")
    parser.add_argument("--max_samples", type=int, default=None, help="Giới hạn số mẫu nạp (None = toàn bộ)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Thiết bị tính toán")
    return parser.parse_args()


def evaluate(model, val_loader, device, use_amp=False):
    """Vòng lặp validation tính validation loss dưới torch.no_grad()."""
    model.eval()
    total_val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=use_amp):
                logits = model(input_ids)
                loss = compute_sft_loss(logits, labels, ignore_index=-100)

            total_val_loss += loss.item()
            val_batches += 1

    model.train()
    return total_val_loss / max(1, val_batches)


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 70)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN SFT LORA (ALPACA-CLEANED)")
    print(f"Device: {args.device} | Base: {args.base_model} | LoRA Rank={args.r}, Alpha={args.alpha}, Dropout={args.dropout}")
    print(f"Batch Size: {args.batch_size} | Grad Accum: {args.grad_accum} | Epochs: {args.epochs} | Seed: {args.seed}")
    print("=" * 70)

    # 1. Khởi tạo Tokenizer
    tokenizer = BytePairTokenizer()
    if os.path.exists(args.vocab_file) and os.path.exists(args.merges_file):
        tokenizer.load(args.vocab_file, args.merges_file)
        print(f"✅ Đã nạp Tokenizer từ {args.vocab_file} (Vocab size: {len(tokenizer.vocab)})")
    else:
        print(f"⚠️ Không tìm thấy file tokenizer tại {args.vocab_file}, sử dụng tokenizer mặc định.")

    # 2. Khởi tạo GPTModel
    model_cfg = ModelConfig(vocab_size=max(tokenizer.vocab_size, len(tokenizer.vocab)))
    model = GPTModel(model_cfg)

    # Nạp base model weights nếu có
    if os.path.exists(args.base_model):
        ckpt = torch.load(args.base_model, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Đã nạp thành công Base Model weights từ {args.base_model}")
    else:
        print(f"⚠️ Base model {args.base_model} không tồn tại! Khởi tạo ngẫu nhiên.")

    # 3. Tiêm LoRA và Đóng băng trọng số gốc
    LoRAManager.inject_lora(
        model,
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "v_proj"],
    )
    LoRAManager.freeze_base_model(model)
    trainable_params, total_params, trainable_ratio = LoRAManager.get_trainable_parameters(model)
    print(f"📊 Thống kê tham số LoRA: Trainable = {trainable_params:,} / {total_params:,} ({trainable_ratio * 100:.2f}%)")

    model.to(args.device)

    # 4. Nạp Dataset và Collator (Train/Val split)
    if not os.path.exists(args.data_file):
        print(f"⚠️ Cảnh báo: File dữ liệu {args.data_file} chưa tồn tại!")
        print("💡 Gợi ý: Hãy chuẩn bị file alpaca_cleaned.json trước khi chạy huấn luyện thực tế.")
        return

    full_dataset = AlpacaSFTDataset(file_path=args.data_file, max_samples=args.max_samples)
    total_len = len(full_dataset)
    val_len = int(total_len * args.val_ratio)
    train_len = total_len - val_len

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len], generator=generator)
    print(f"📚 Dữ liệu nạp: {total_len:,} mẫu | Train: {train_len:,} | Validation: {val_len:,}")

    collator = SFTDataCollator(tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator) if val_len > 0 else None

    # 5. Optimizer (chỉ tối ưu tham số LoRA trainable)
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params_list, lr=args.lr, weight_decay=0.01)

    use_amp = args.device.startswith("cuda") and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    global_step = 0
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)

            with torch.amp.autocast(device_type="cuda" if "cuda" in args.device else "cpu", enabled=use_amp):
                # Forward pass qua GPTModel
                logits = model(input_ids)
                # Tính SFT Loss với cơ chế Shift Logits & Labels chuẩn xác
                loss = compute_sft_loss(logits, labels, ignore_index=-100)
                loss = loss / args.grad_accum

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params_list, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params_list, max_norm=1.0)
                    optimizer.step()

                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    avg_loss = epoch_loss / (step + 1)
                    print(f"Epoch {epoch+1}/{args.epochs} | Step {global_step}/{total_steps} | Train Loss: {avg_loss:.4f}")

        # Validation cuối mỗi epoch
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, args.device, use_amp=use_amp)
            print(f"📊 Kết thúc Epoch {epoch+1} | Train Loss: {epoch_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(args.output_lora), exist_ok=True)
                LoRAManager.save_lora_weights(
                    model,
                    args.output_lora,
                    metadata={"r": args.r, "alpha": args.alpha, "epoch": epoch + 1, "val_loss": val_loss, "lr": args.lr},
                )
                print(f"⭐ Đã lưu Best Adapter LoRA tại Epoch {epoch+1} (Val Loss: {val_loss:.4f}) -> {args.output_lora}")

    # Nếu không có val_loader, lưu ở bước cuối
    if val_loader is None:
        os.makedirs(os.path.dirname(args.output_lora), exist_ok=True)
        LoRAManager.save_lora_weights(
            model,
            args.output_lora,
            metadata={"r": args.r, "alpha": args.alpha, "epochs": args.epochs, "lr": args.lr},
        )
        print(f"🎉 HUẤN LUYỆN SFT HOÀN TẤT! Adapter LoRA đã được lưu tại: {args.output_lora}")


if __name__ == "__main__":
    main()
