"""
==============================================================================
FILE: tests/test_tokenizer.py
MODULE: tests.test_tokenizer
RESPONSIBILITY:
    - Kiểm thử đơn vị toàn diện cho BytePairTokenizer:
      1. Khóa cứng Special tokens (ID 0..4) và Raw bytes (ID 5..260).
      2. Huấn luyện BPE với GPT-4 Regex tách từ và tối ưu word_counts.
      3. Round-trip test: decode(encode(s)) == s.
      4. Kiểm tra các ca biên:
         - Chuỗi rỗng "" và chuỗi chỉ có khoảng trắng.
         - Special tokens xen kẽ văn bản thường.
         - Tiếng Việt có dấu, Emoji 4-byte UTF-8, Code Python đa dòng.
      5. Lưu và tải tokenizer (save / load).
==============================================================================
"""

import os
import shutil
import tempfile

from src.tokenizer.bpe import BytePairTokenizer, BYTE_OFFSET
from src.tokenizer.special_tokens import (
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    PAD_ID,
    BOS_ID,
    EOS_ID,
    USER_ID,
    ASSISTANT_ID,
)


def test_special_tokens_fixed_ids():
    """Kiểm tra khóa cứng ID của Special Tokens."""
    tok = BytePairTokenizer(vocab_size=32000)

    assert tok.special_tokens[PAD_TOKEN] == 0
    assert tok.special_tokens[BOS_TOKEN] == 1
    assert tok.special_tokens[EOS_TOKEN] == 2
    assert tok.special_tokens[USER_TOKEN] == 3
    assert tok.special_tokens[ASSISTANT_TOKEN] == 4

    assert PAD_ID == 0
    assert BOS_ID == 1
    assert EOS_ID == 2
    assert USER_ID == 3
    assert ASSISTANT_ID == 4


def test_raw_bytes_offset():
    """Kiểm tra 256 raw bytes được gán từ ID 5 đến 260."""
    tok = BytePairTokenizer(vocab_size=32000)

    # 256 bytes tương ứng ID 5..260
    for b in range(256):
        expected_id = BYTE_OFFSET + b
        assert tok.vocab[expected_id] == bytes([b])


def test_train_and_merges():
    """Kiểm tra huấn luyện BPE trên mẫu dữ liệu văn bản nhỏ."""
    corpus = [
        "học máy và trí tuệ nhân tạo.",
        "học sâu là một phần của học máy.",
        "mô hình ngôn ngữ lớn huấn luyện trên dữ liệu lớn.",
    ]

    tok = BytePairTokenizer(vocab_size=300)
    tok.train(corpus)

    assert len(tok.merges) > 0
    assert len(tok.vocab) <= 300


def test_edge_cases_round_trip():
    """Kiểm tra các trường hợp biên và round-trip decode(encode(s)) == s."""
    corpus = [
        "Trường Đại học Bách Khoa - Đại học Đà Nẵng.",
        "Hello world! Python 3.10 and PyTorch are awesome.",
        "<|user|>Hỏi gì đó?<|assistant|>Trả lời đây.<|eos|>",
    ]

    tok = BytePairTokenizer(vocab_size=320)
    tok.train(corpus)

    test_cases = [
        # 1. Chuỗi rỗng & khoảng trắng
        "",
        "   ",
        "\n\t\r\n",

        # 2. Special tokens xen kẽ văn bản thường
        f"{BOS_TOKEN} Xin chào! {EOS_TOKEN}",
        f"{USER_TOKEN}Viết hàm tính giai thừa bằng Python.{ASSISTANT_TOKEN}def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1){EOS_TOKEN}",

        # 3. Tiếng Việt có dấu phức tạp
        "Trường Đại học Bách Khoa, Đại học Đà Nẵng - Khoa Công nghệ Thông tin.",
        "Nguyễn Du đã viết Truyện Kiều: Trăm năm trong cõi người ta, chữ tài chữ mệnh khéo là ghét nhau.",

        # 4. Emoji và ký tự đa byte UTF-8
        "🤖 Mô hình AI thông minh 🇻🇳 Chúc mừng năm mới 🎉🚀🔥",
        "Ký tự toán học: ∀x ∈ ℝ, ∃y: y > x; ∑_{i=1}^n x_i",

        # 5. Mã nguồn Python với thụt lề, dấu câu dính liền
        """def train_step(model, batch):\n    optimizer.zero_grad()\n    loss = model(batch)\n    loss.backward()\n    optimizer.step()\n    return loss.item()\n""",

        # 6. Chuỗi số và phân biệt dấu câu
        "cat. dog! cat? Số pi xấp xỉ 3.1415926535.",
    ]

    for text in test_cases:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text, f"Lỗi Round-trip thất bại với chuỗi: {repr(text)}\nGot: {repr(decoded)}"


def test_save_and_load():
    """Kiểm tra tính toàn vẹn khi lưu (save) và nạp lại (load) tokenizer."""
    corpus = [
        "Xây dựng tokenizer byte-fallback bpe cho mô hình ngôn ngữ.",
        "Kiểm tra hàm save và load.",
    ]

    tok = BytePairTokenizer(vocab_size=300)
    tok.train(corpus)

    test_dir = os.path.join(os.path.dirname(__file__), "tmp_save_load")
    os.makedirs(test_dir, exist_ok=True)
    vocab_file = os.path.join(test_dir, "vocab.json")
    merges_file = os.path.join(test_dir, "merges.txt")

    try:
        tok.save(vocab_file, merges_file)
        assert os.path.exists(vocab_file)
        assert os.path.exists(merges_file)

        loaded_tok = BytePairTokenizer()
        loaded_tok.load(vocab_file, merges_file)

        assert loaded_tok.vocab_size == tok.vocab_size
        assert len(loaded_tok.merges) == len(tok.merges)
        assert loaded_tok.vocab == tok.vocab

        sample_text = f"{USER_TOKEN}Kiểm tra tokenizer{ASSISTANT_TOKEN}Hoàn hảo!{EOS_TOKEN}"
        assert loaded_tok.encode(sample_text) == tok.encode(sample_text)
        assert loaded_tok.decode(loaded_tok.encode(sample_text)) == sample_text
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    print("Running test_special_tokens_fixed_ids()...")
    test_special_tokens_fixed_ids()
    print("PASS: test_special_tokens_fixed_ids")

    print("Running test_raw_bytes_offset()...")
    test_raw_bytes_offset()
    print("PASS: test_raw_bytes_offset")

    print("Running test_train_and_merges()...")
    test_train_and_merges()
    print("PASS: test_train_and_merges")

    print("Running test_edge_cases_round_trip()...")
    test_edge_cases_round_trip()
    print("PASS: test_edge_cases_round_trip")

    print("Running test_save_and_load()...")
    test_save_and_load()
    print("PASS: test_save_and_load")

    print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")

