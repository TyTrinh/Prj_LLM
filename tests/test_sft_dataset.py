"""
==============================================================================
FILE: tests/test_sft_dataset.py
RESPONSIBILITY:
    - Kiểm thử module SFT Dataset và cơ chế Target Loss Masking.
    - Test case 1: Chat Template chuẩn hóa đúng với và không có input.
    - Test case 2: Target Loss Masking: 100% token thuộc prompt được gán nhãn -100.
    - Test case 3: Token response và token <|eos|> giữ nguyên ID thực tế (không bị mask).
    - Test case 4: Padding tokens nhận nhãn -100 và attention_mask = 0.
    - Test case 5: Mô phỏng CrossEntropyLoss(ignore_index=-100): gradient tại prompt bị triệt tiêu (=0.0),
      gradient tại response và <|eos|> kích hoạt học (!= 0.0).
==============================================================================
"""

import pytest
import torch
import torch.nn as nn

from src.alignment.sft_dataset import (
    AlpacaSFTDataset,
    SFTDataCollator,
    compute_sft_loss,
    format_alpaca_prompt,
    format_sft_text,
)
from src.tokenizer.bpe import BytePairTokenizer
from src.tokenizer.special_tokens import (
    USER_TOKEN,
    ASSISTANT_TOKEN,
    EOS_TOKEN,
    PAD_ID,
    USER_ID,
    ASSISTANT_ID,
    EOS_ID,
)


@pytest.fixture
def dummy_tokenizer():
    tokenizer = BytePairTokenizer(vocab_size=300)
    return tokenizer


def test_format_alpaca_prompt():
    # Trường hợp không có input
    p1 = format_alpaca_prompt("Dịch sang tiếng Anh", "")
    assert p1 == "Dịch sang tiếng Anh"

    p2 = format_alpaca_prompt("Dịch sang tiếng Anh", None)
    assert p2 == "Dịch sang tiếng Anh"

    # Trường hợp có input
    p3 = format_alpaca_prompt("Dịch sang tiếng Anh", "Xin chào")
    assert p3 == "Dịch sang tiếng Anh\n\nXin chào"


def test_format_sft_text():
    prompt = "Thủ đô của Việt Nam là gì?"
    response = "Thủ đô của Việt Nam là Hà Nội."
    text = format_sft_text(prompt, response)
    expected = f"<|user|>Thủ đô của Việt Nam là gì?<|assistant|>Thủ đô của Việt Nam là Hà Nội.<|eos|>"
    assert text == expected


def test_alpaca_sft_dataset():
    raw_data = [
        {"instruction": "Câu hỏi 1", "input": "", "output": "Trả lời 1"},
        {"instruction": "Câu hỏi 2", "input": "Ngữ cảnh", "output": "Trả lời 2"},
    ]
    dataset = AlpacaSFTDataset(data=raw_data)
    assert len(dataset) == 2
    assert dataset[0]["prompt"] == "Câu hỏi 1"
    assert dataset[0]["response"] == "Trả lời 1"
    assert dataset[1]["prompt"] == "Câu hỏi 2\n\nNgữ cảnh"
    assert dataset[1]["response"] == "Trả lời 2"


def test_target_loss_masking_values(dummy_tokenizer):
    raw_data = [
        {"instruction": "Xin chào!", "input": "", "output": "Chào bạn!"}
    ]
    dataset = AlpacaSFTDataset(data=raw_data)
    collator = SFTDataCollator(tokenizer=dummy_tokenizer, max_seq_len=64, ignore_index=-100)
    batch = collator([dataset[0]])

    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    attention_mask = batch["attention_mask"][0]

    # Kiểm tra kích thước tensor
    assert input_ids.shape == (64,)
    assert labels.shape == (64,)
    assert attention_mask.shape == (64,)

    # Xác định ranh giới
    prompt_str = f"{USER_TOKEN}Xin chào!{ASSISTANT_TOKEN}"
    prompt_len = len(dummy_tokenizer.encode(prompt_str))

    response_str = f"Chào bạn!{EOS_TOKEN}"
    response_ids = dummy_tokenizer.encode(response_str)
    response_len = len(response_ids)

    total_len = prompt_len + response_len

    # 1. Toàn bộ prompt tokens phải có label == -100
    for idx in range(prompt_len):
        assert labels[idx].item() == -100, f"Token tại vị trí prompt {idx} không được mask -100"

    # 2. Toàn bộ response tokens và token <|eos|> phải giữ nguyên ID (label != -100)
    for idx in range(prompt_len, total_len):
        assert labels[idx].item() == input_ids[idx].item()
        assert labels[idx].item() != -100

    # Token cuối của response phải là EOS_ID
    assert input_ids[total_len - 1].item() == EOS_ID
    assert labels[total_len - 1].item() == EOS_ID

    # 3. Các vị trí padding còn lại phải là PAD_ID, labels == -100, attention_mask == 0
    for idx in range(total_len, 64):
        assert input_ids[idx].item() == PAD_ID
        assert labels[idx].item() == -100
        assert attention_mask[idx].item() == 0


def test_compute_sft_loss_gradient_elimination(dummy_tokenizer):
    """
    CRIT-02 Verification:
    Kiểm tra cơ chế Target Loss Masking khi tính toán bằng chính hàm production `compute_sft_loss`:
    - Do có cơ chế Shift: shift_logits[:, :-1] dự đoán shift_labels[:, 1:]
    - Token prompt tại vị trí t dự đoán prompt token t+1 (label = -100) -> gradient tại logit t phải = 0.
    - Token prompt cuối cùng dự đoán token đầu tiên của response (label != -100) -> gradient != 0.
    - Token cuối cùng của sequence (index -1) bị shift loại bỏ -> gradient tại logit -1 phải = 0.
    """
    collator = SFTDataCollator(tokenizer=dummy_tokenizer, max_seq_len=32, ignore_index=-100)
    batch = collator([{"prompt": "Hi", "response": "Hello"}])

    labels = batch["labels"]  # (1, 32)
    seq_len = 32
    vocab_size = dummy_tokenizer.vocab_size

    # Tạo logits giả lập có requires_grad=True
    logits = torch.randn(1, seq_len, vocab_size, requires_grad=True)

    # Tính loss bằng chính hàm production compute_sft_loss (có shift)
    loss = compute_sft_loss(logits, labels, ignore_index=-100)
    loss.backward()

    grad = logits.grad  # shape: (1, seq_len, vocab_size)

    # Vị trí prompt_len:
    prompt_str = f"{USER_TOKEN}Hi{ASSISTANT_TOKEN}"
    prompt_len = len(dummy_tokenizer.encode(prompt_str))

    # Với shift: logit tại t dự đoán label tại t+1
    for t in range(seq_len - 1):
        target_label = labels[0, t + 1].item()
        if target_label == -100:
            assert torch.all(grad[0, t] == 0.0), f"Gradient tại vị trí logit {t} dự đoán masked token khác 0!"
        else:
            assert not torch.all(grad[0, t] == 0.0), f"Gradient tại vị trí logit {t} dự đoán target bị triệt tiêu!"

    # Logit cuối cùng (seq_len - 1) không có token kế tiếp để dự đoán -> gradient bắt buộc = 0
    assert torch.all(grad[0, -1] == 0.0), "Gradient tại logit cuối cùng phải bằng 0 do bị shift loại bỏ!"



def test_tokenizer_apply_chat_template(dummy_tokenizer):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    text = dummy_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    expected = "<|user|>Hello<|assistant|>Hi there<|eos|>"
    assert text == expected

    # Test add_generation_prompt=True (dùng khi inference)
    infer_text = dummy_tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        add_generation_prompt=True
    )
    assert infer_text == "<|user|>Hello<|assistant|>"

    # Test tokenize=True
    tokens = dummy_tokenizer.apply_chat_template(messages, tokenize=True)
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_safe_truncation_preserves_eos_and_targets(dummy_tokenizer):
    """
    Kiểm thử an toàn khi chuỗi vượt quá max_seq_len:
    - Khi chuỗi dài bị cắt, token cuối cùng của input_ids và labels BẮT BUỘC là EOS_ID.
    - Không bao giờ xảy ra tình trạng toàn bộ labels đều là -100 (tránh NaN loss / chia cho 0).
    """
    collator = SFTDataCollator(
        tokenizer=dummy_tokenizer,
        max_seq_len=20,
        min_response_tokens=4,
        ignore_index=-100,
    )

    long_prompt = "Một câu hỏi rất dài vượt quá độ dài tối đa của cửa sổ ngữ cảnh thử nghiệm"
    long_response = "Một câu trả lời cũng rất dài để kích hoạt cơ chế cắt cụt an toàn"

    batch = collator([{"prompt": long_prompt, "response": long_response}])
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    assert input_ids.shape == (20,)
    assert labels.shape == (20,)

    # Token cuối cùng của chuỗi bị cắt PHẢI là EOS_ID
    assert input_ids[-1].item() == EOS_ID
    assert labels[-1].item() == EOS_ID

    # Đảm bảo có ít nhất min_response_tokens nhãn hợp lệ (!= -100)
    valid_labels = (labels != -100).sum().item()
    assert valid_labels >= 4, f"Số nhãn hợp lệ ({valid_labels}) ít hơn min_response_tokens (4)"


def test_compute_sft_loss_shift_logits(dummy_tokenizer):
    """
    Kiểm thử hàm compute_sft_loss:
    - Xác minh cơ chế Shift Logits [..., :-1, :] và Labels [..., 1:].
    - Đảm bảo hàm tính toán thành công, trả về tensor vô hướng và không bị lỗi NaN.
    """
    batch_size = 2
    seq_len = 16
    vocab_size = dummy_tokenizer.vocab_size

    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    # Giả lập labels với một nửa là -100, một nửa là token ID
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels[:, :8] = -100

    loss = compute_sft_loss(logits, labels, ignore_index=-100)

    assert not torch.isnan(loss)
    assert loss.dim() == 0  # scalar
    assert loss.item() > 0

    loss.backward()
    assert logits.grad is not None
    # Logits tại bước cuối cùng logits[:, -1, :] bị shift bỏ qua nên grad bằng 0
    assert torch.all(logits.grad[:, -1, :] == 0.0)
    # Logits tại các bước dự đoán token target phải có grad != 0
    assert not torch.all(logits.grad[:, 7, :] == 0.0)

