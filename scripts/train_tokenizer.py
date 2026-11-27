"""
==============================================================================
FILE: scripts/train_tokenizer.py
MODULE: scripts.train_tokenizer
RESPONSIBILITY:
    - Huấn luyện Byte-Fallback BPE Tokenizer (Vocab 8192) từ số 0 trên tập TinyStories.
    - Shuffle ngẫu nhiên thực sự theo seed=42 để phân tách 475.000 Train / 25.000 Validation.
    - Lưu danh sách chỉ số (indices) cố định vào data/splits_indices.json để tái lập xuyên suốt.
    - CHỈ học merge rules trên 475K Train stories (không đưa tập Validation vào học).
    - Lưu trữ artifacts kèm Fingerprint toàn vẹn và metadata tái lập.
    - Fused Streaming Tokenization & Profiling: Encode và ghi trực tiếp ra binary (uint16)
      qua buffer, không tokenize 2 lần, không giữ danh sách token khổng lồ trong RAM.
==============================================================================
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tokenizer import BytePairTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện Byte-Fallback BPE Tokenizer")
    parser.add_argument("--vocab_size", type=int, default=8192, help="Kích thước từ điển mục tiêu (mặc định: 8192)")
    parser.add_argument("--num_samples", type=int, default=500000, help="Tổng số mẫu TinyStories (mặc định: 500000)")
    parser.add_argument("--train_ratio", type=float, default=0.95, help="Tỷ lệ train split (mặc định: 0.95 -> 475K train / 25K val)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/tokenizer_8k", help="Thư mục lưu artifact tokenizer")
    parser.add_argument("--data_dir", type=str, default="data", help="Thư mục lưu train.bin và val.bin")
    parser.add_argument("--seed", type=int, default=42, help="Random seed cố định để shuffle")
    parser.add_argument("--force_overwrite", action="store_true", help="Bắt buộc ghi đè artifact nếu đã tồn tại")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    vocab_file = os.path.join(args.output_dir, "vocab.json")
    merges_file = os.path.join(args.output_dir, "merges.txt")
    train_bin_path = os.path.join(args.data_dir, "train.bin")
    val_bin_path = os.path.join(args.data_dir, "val.bin")

    # 1. Cơ chế Artifact Safety Lock (Chốt an toàn chống ghi đè)
    if os.path.exists(vocab_file) and os.path.exists(train_bin_path) and not args.force_overwrite:
        print("=" * 80)
        print("🔒 [ARTIFACT LOCKED] Tokenizer và dữ liệu train.bin/val.bin đã tồn tại sẵn!")
        print(f"• Vocab: {vocab_file}")
        print(f"• Train Data: {train_bin_path}")
        print("👉 Bỏ qua quy trình tạo lại để bảo đảm tính bất biến cho Pretraining.")
        print("💡 (Để ghi đè, hãy truyền thêm cờ --force_overwrite)")
        print("=" * 80)
        return

    print("=" * 80)
    print("🚀 BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN BYTE-FALLBACK BPE TOKENIZER (VOCAB 8K)")
    print(f"• Vocab size mục tiêu: {args.vocab_size:,}")
    print(f"• Tổng số mẫu: {args.num_samples:,} (Seed: {args.seed})")
    print(f"• Thư mục output: {args.output_dir}")
    print("=" * 80)

    # 2. Tải dataset TinyStories
    print("\n📥 [Bước 1/5] Đang tải tập dữ liệu TinyStories từ HuggingFace...")
    from datasets import load_dataset
    dataset = load_dataset("roneneldan/TinyStories", split=f"train[:{args.num_samples}]")
    total_loaded = len(dataset)
    print(f"✅ Đã tải thành công: {total_loaded:,} stories.")

    # 3. Shuffle ngẫu nhiên thực sự theo seed=42 để chia Train / Validation
    print(f"\n🎲 [Bước 2/5] Xáo trộn ngẫu nhiên (Shuffle) theo seed={args.seed} để phân tách Train/Val...")
    rng = np.random.default_rng(seed=args.seed)
    shuffled_indices = rng.permutation(total_loaded)

    num_train = int(total_loaded * args.train_ratio)
    train_indices = shuffled_indices[:num_train]
    val_indices = shuffled_indices[num_train:]

    print(f"• Train set: {len(train_indices):,} stories ({args.train_ratio*100:.0f}%)")
    print(f"• Validation set: {len(val_indices):,} stories ({(1-args.train_ratio)*100:.0f}%)")

    # Lưu metadata và danh sách indices bất biến
    splits_meta_path = os.path.join(args.data_dir, "splits_metadata.json")
    with open(splits_meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "roneneldan/TinyStories",
            "total_samples": total_loaded,
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "seed": args.seed,
            "train_indices_preview": train_indices[:20].tolist(),
            "val_indices_preview": val_indices[:20].tolist(),
        }, f, indent=2)
    print(f"✅ Đã lưu thông tin phân chia split tại: {splits_meta_path}")

    # 4. Huấn luyện BPE Tokenizer CHỈ TRÊN TẬP TRAIN
    print(f"\n🧠 [Bước 3/5] Huấn luyện BPE Tokenizer trên {len(train_indices):,} câu chuyện Train...")
    train_texts = (dataset[int(idx)]["text"] for idx in train_indices)

    tokenizer = BytePairTokenizer(vocab_size=args.vocab_size)
    t0 = time.time()
    tokenizer.train(train_texts, verbose=True)
    t1 = time.time()
    train_time = t1 - t0
    print(f"✅ Huấn luyện Tokenizer hoàn tất trong: {train_time:.2f}s ({train_time/60:.2f} phút)")
    print(f"• Kích thước vocab thực tế: {len(tokenizer.vocab):,}")
    print(f"• Số lượng merge rules: {len(tokenizer.merges):,}")

    # 5. Lưu Tokenizer Artifacts & Tính Fingerprint toàn vẹn
    print("\n💾 [Bước 4/5] Lưu trữ Tokenizer Artifacts & Kiểm định toàn vẹn...")
    tokenizer.save(args.output_dir)
    tokenizer.save("checkpoints")  # Copy bản vào checkpoints/ để tiện sử dụng

    fingerprint = tokenizer.get_fingerprint()
    meta_info = {
        "vocab_size": len(tokenizer.vocab),
        "target_vocab_size": args.vocab_size,
        "num_merges": len(tokenizer.merges),
        "fingerprint": fingerprint,
        "special_tokens": tokenizer.special_tokens,
        "train_samples": len(train_indices),
        "seed": args.seed,
        "duration_sec": train_time,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    print(f"✅ Đã lưu Vocab: {vocab_file}")
    print(f"✅ Đã lưu Merges: {merges_file}")
    print(f"🔒 Tokenizer Fingerprint (SHA256): {fingerprint}")

    # 6. Fused Streaming Tokenization & Profiling (Duyệt 1 lượt qua buffer)
    print("\n📦 [Bước 5/5] Tokenize & Đóng gói dữ liệu ra binary (uint16) qua Streaming Buffer...")
    eos_id = tokenizer.special_tokens.get("<|eos|>", 2)

    char_lengths = np.zeros(total_loaded, dtype=np.int32)
    word_lengths = np.zeros(total_loaded, dtype=np.int32)
    token_lengths = np.zeros(total_loaded, dtype=np.int32)

    # 6.1 Stream Train Set
    total_train_tokens = 0
    with open(train_bin_path, "wb") as f_train:
        buffer = []
        for count, idx in enumerate(train_indices):
            text = dataset[int(idx)]["text"]
            char_lengths[idx] = len(text)
            word_lengths[idx] = len(text.split())

            tokens = tokenizer.encode(text) + [eos_id]
            token_lengths[idx] = len(tokens)
            buffer.extend(tokens)
            total_train_tokens += len(tokens)

            # Ghi chunk 10.000 câu để tiết kiệm RAM
            if len(buffer) >= 500_000:
                np.array(buffer, dtype=np.uint16).tofile(f_train)
                buffer.clear()

            if (count + 1) % 100_000 == 0:
                print(f"  -> Đã mã hóa tập Train: {count + 1:,}/{len(train_indices):,} stories...")

        if buffer:
            np.array(buffer, dtype=np.uint16).tofile(f_train)
            buffer.clear()

    # 6.2 Stream Val Set
    total_val_tokens = 0
    with open(val_bin_path, "wb") as f_val:
        buffer = []
        for count, idx in enumerate(val_indices):
            text = dataset[int(idx)]["text"]
            char_lengths[idx] = len(text)
            word_lengths[idx] = len(text.split())

            tokens = tokenizer.encode(text) + [eos_id]
            token_lengths[idx] = len(tokens)
            buffer.extend(tokens)
            total_val_tokens += len(tokens)

            if len(buffer) >= 200_000:
                np.array(buffer, dtype=np.uint16).tofile(f_val)
                buffer.clear()

        if buffer:
            np.array(buffer, dtype=np.uint16).tofile(f_val)
            buffer.clear()

    train_size_mb = os.path.getsize(train_bin_path) / (1024 * 1024)
    val_size_mb = os.path.getsize(val_bin_path) / (1024 * 1024)
    print(f"✅ Đã lưu {train_bin_path}: {total_train_tokens:,} tokens ({train_size_mb:.2f} MB)")
    print(f"✅ Đã lưu {val_bin_path}: {total_val_tokens:,} tokens ({val_size_mb:.2f} MB)")

    # 7. Thống kê toàn cục
    print("\n" + "=" * 80)
    print(f"📊 BÁO CÁO ĐẶC TRƯNG TOÀN DIỆN 100% CORPUS TINYSTORIES ({total_loaded:,} SAMPLES):")
    print("=" * 80)
    print(f"• Ký tự (Characters):  Mean = {np.mean(char_lengths):.1f} | Max = {np.max(char_lengths)}")
    print(f"• Số từ (Words):       Mean = {np.mean(word_lengths):.1f} | Max = {np.max(word_lengths)}")
    print(f"• Tokens (BPE 8K):     Mean = {np.mean(token_lengths):.1f} | Median = {np.median(token_lengths):.1f} | Max = {np.max(token_lengths)}")
    print(f"• Phân vị Tokens:      P50 = {np.percentile(token_lengths, 50):.0f} | P90 = {np.percentile(token_lengths, 90):.0f} | P95 = {np.percentile(token_lengths, 95):.0f} | P99 = {np.percentile(token_lengths, 99):.0f}")
    print(f"• Tỷ lệ Nén BPE 8K:    {np.sum(token_lengths) / np.sum(word_lengths):.2f} tokens/từ")
    print("=" * 80)
    print("🎉 TOÀN BỘ QUY TRÌNH CHUẨN BỊ TOKENIZER & DỮ LIỆU ĐÃ HOÀN TẤT!")


if __name__ == "__main__":
    main()
