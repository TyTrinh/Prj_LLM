"""
==============================================================================
FILE: src/alignment/sft_dataset.py
MODULE: src.alignment.sft_dataset
RESPONSIBILITY:
    - Đơn trách nhiệm: Xử lý và chuẩn bị dữ liệu tinh chỉnh có giám sát (SFT - Alpaca-Cleaned).
    - Định dạng hội thoại theo Chat Template thống nhất:
        `<|user|>{prompt}<|assistant|>{response}<|eos|>`
    - Target Loss Masking:
        + Toàn bộ token thuộc phân vùng `<|user|>{prompt}<|assistant|>` được gán nhãn `-100`.
        + Chỉ tính loss trên chuỗi token của `{response}` VÀ token `<|eos|>`.
        + Token padding `<|pad|>` được gán nhãn `-100`.
    - Bảo vệ an toàn chống Loss NaN / Mất Stop Token:
        + Lọc/cắt an toàn bảo tồn token dừng `<|eos|>` ở cuối ngay cả khi chuỗi bị cắt cụt.
        + Đảm bảo không bao giờ có batch chứa 100% nhãn `-100`.
    - Hàm mất mát chuẩn hóa Causal LM:
        + `compute_sft_loss`: Tự động Shift Logits [..., :-1, :] và Labels [..., 1:]
          cho bài toán Next-Token Prediction.
==============================================================================
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.tokenizer.special_tokens import (
    USER_TOKEN,
    ASSISTANT_TOKEN,
    EOS_TOKEN,
    PAD_ID,
    EOS_ID,
)

logger = logging.getLogger(__name__)


def format_alpaca_prompt(instruction: str, input_text: Optional[str] = None) -> str:
    """
    Chuẩn hóa prompt từ instruction và input tùy chọn theo chuẩn Alpaca:
    - Nếu có input: f"{instruction}\n\n{input_text}"
    - Nếu không có: instruction
    """
    instruction = instruction.strip()
    if input_text and input_text.strip():
        return f"{instruction}\n\n{input_text.strip()}"
    return instruction


def format_sft_text(prompt: str, response: str) -> str:
    """
    Tạo chuỗi hội thoại hoàn chỉnh theo Chat Template:
    `<|user|>{prompt}<|assistant|>{response}<|eos|>`
    """
    return f"{USER_TOKEN}{prompt.strip()}{ASSISTANT_TOKEN}{response.strip()}{EOS_TOKEN}"


def compute_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Tính hàm mất mát Cross-Entropy chuẩn cho Causal LM (Next-Token Prediction):
    - Logits tại bước t dự đoán token tại bước t+1:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
    - Triệt tiêu hoàn toàn gradient tại các vị trí mang nhãn `ignore_index` (-100).
    """
    # Shift logits và labels để khớp nhịp dự đoán token kế tiếp
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    vocab_size = shift_logits.size(-1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss = loss_fn(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    return loss


class AlpacaSFTDataset(Dataset):
    """
    Dataset cho tập dữ liệu Alpaca-Cleaned phục vụ huấn luyện SFT.
    Mỗi mẫu chứa:
        - prompt: Câu lệnh/ngữ cảnh của User
        - response: Câu trả lời mẫu của Assistant
    """

    def __init__(
        self,
        data: Optional[List[Dict[str, Any]]] = None,
        file_path: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        if data is not None:
            raw_data = data
        elif file_path is not None:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        else:
            raw_data = []

        if max_samples is not None and max_samples > 0:
            raw_data = raw_data[:max_samples]

        self.samples: List[Dict[str, str]] = []
        for item in raw_data:
            instruction = item.get("instruction", "").strip()
            input_text = item.get("input", "").strip()
            output = item.get("output", "").strip()
            # Yêu cầu chặt chẽ: Cả instruction (hoặc prompt) và response đều không được rỗng
            if not instruction or not output:
                continue

            prompt = format_alpaca_prompt(instruction, input_text)
            self.samples.append({
                "prompt": prompt,
                "response": output,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.samples[idx]


class SFTDataCollator:
    """
    Collate function cho SFT với cơ chế Target Loss Masking và Bảo tồn Stop Token:
    - Tokenize prompt riêng và response riêng để phân định ranh giới chuẩn xác.
    - An toàn khi cắt cụt (Safe Truncation):
        + Nếu prompt quá dài vượt quá (max_seq_len - min_response_tokens), cắt bớt phần đầu
          của prompt hoặc bỏ qua để luôn dành tối thiểu `min_response_tokens` cho câu trả lời.
        + Khi toàn bộ chuỗi vượt quá `max_seq_len`, phần thân response bị cắt nhưng
          **luôn ghim token `<|eos|>` ở cuối chuỗi** để mô hình học điểm dừng.
    - Gán nhãn `-100` cho:
        1. Toàn bộ prompt tokens (`<|user|>{prompt}<|assistant|>`)
        2. Các token đệm (`<|pad|>`)
    - Giữ nguyên token ID cho:
        1. Các token thuộc `{response}`
        2. Token dừng `<|eos|>` (BẮT BUỘC để mô hình học cách dừng sinh).
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len: int = 512,
        min_response_tokens: int = 4,
        ignore_index: int = -100,
        pad_id: int = PAD_ID,
        eos_id: int = EOS_ID,
        dynamic_padding: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.min_response_tokens = max(1, min_response_tokens)
        self.ignore_index = ignore_index
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.dynamic_padding = dynamic_padding

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for item in batch:
            prompt = item["prompt"]
            response = item["response"]

            # 1. Prompt header và body: <|user|>{prompt}<|assistant|>
            prompt_str = f"{USER_TOKEN}{prompt.strip()}{ASSISTANT_TOKEN}"
            prompt_ids = self.tokenizer.encode(prompt_str)

            # 2. Response body và EOS: {response}<|eos|>
            response_str = f"{response.strip()}{EOS_TOKEN}"
            response_ids = self.tokenizer.encode(response_str)

            # Đảm bảo response kết thúc bằng eos_id
            if not response_ids or response_ids[-1] != self.eos_id:
                response_ids.append(self.eos_id)

            # 3. Xử lý an toàn khi prompt quá dài
            # Nếu prompt chiếm quá nhiều ngữ cảnh, cắt bớt phần đầu prompt để bảo toàn response
            max_prompt_len = self.max_seq_len - self.min_response_tokens
            if len(prompt_ids) > max_prompt_len:
                logger.warning(
                    f"Prompt quá dài ({len(prompt_ids)} tokens > {max_prompt_len}). "
                    f"Cắt bớt phần đầu prompt để bảo toàn {self.min_response_tokens} tokens response."
                )
                # Xác minh runtime: Special token <|user|> và <|assistant|> có độ dài token hóa chuẩn
                user_token_ids = self.tokenizer.encode(USER_TOKEN)
                assistant_token_ids = self.tokenizer.encode(ASSISTANT_TOKEN)
                len_user = len(user_token_ids)
                len_asst = len(assistant_token_ids)

                prefix = prompt_ids[:len_user]   # <|user|>
                suffix = prompt_ids[-len_asst:]  # <|assistant|>
                middle_allowed = max(0, max_prompt_len - len(prefix) - len(suffix))
                prompt_ids = prefix + prompt_ids[len(prompt_ids) - middle_allowed - len(suffix): -len(suffix)] + suffix

            prompt_len = len(prompt_ids)

            # 4. Xử lý an toàn khi tổng chuỗi vượt quá max_seq_len (Bảo tồn <|eos|>)
            remaining_for_response = self.max_seq_len - prompt_len
            if len(response_ids) > remaining_for_response:
                # Cắt bớt thân response nhưng LUÔN GHIM token cuối là eos_id
                response_ids = response_ids[: remaining_for_response - 1] + [self.eos_id]

            input_ids = prompt_ids + response_ids
            seq_len = len(input_ids)

            # 5. Xây dựng nhãn với Target Loss Masking:
            # - Vùng prompt: gán ignore_index (-100)
            # - Vùng response: giữ nguyên token IDs thật
            labels = [self.ignore_index] * prompt_len + response_ids
            attention_mask = [1] * seq_len

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(attention_mask)

        # 6. Padding theo Dynamic Padding (độ dài lớn nhất trong batch) hoặc Max Seq Len
        if self.dynamic_padding:
            max_batch_len = max(len(seq) for seq in batch_input_ids) if batch_input_ids else self.max_seq_len
            target_pad_len = min(self.max_seq_len, max_batch_len)
        else:
            target_pad_len = self.max_seq_len

        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []

        for input_ids, labels, attention_mask in zip(batch_input_ids, batch_labels, batch_attention_mask):
            padding_len = target_pad_len - len(input_ids)
            if padding_len > 0:
                input_ids = input_ids + [self.pad_id] * padding_len
                labels = labels + [self.ignore_index] * padding_len
                attention_mask = attention_mask + [0] * padding_len
            padded_input_ids.append(input_ids)
            padded_labels.append(labels)
            padded_attention_mask.append(attention_mask)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        }
