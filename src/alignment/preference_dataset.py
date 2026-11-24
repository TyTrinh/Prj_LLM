"""
==============================================================================
FILE: src/alignment/preference_dataset.py
MODULE: src.alignment.preference_dataset
RESPONSIBILITY:
    - Đơn trách nhiệm: Xử lý và chuẩn bị dữ liệu cặp ưu tiên (Preference Pairs).
    - Phân tách mẫu dữ liệu thành: `prompt`, `chosen`, `rejected`.
    - Tạo `response_mask` đánh dấu chính xác vị trí token của phản hồi để hàm DPO
      chỉ tính log-probability trên câu trả lời, loại trừ prompt.
INPUT:
    - Danh sách mẫu dictionary chứa {"prompt", "chosen", "rejected"}.
    - tokenizer: Tokenizer dùng để mã hóa chuỗi.
OUTPUT:
    - Batch PyTorch Tensor:
      `chosen_input_ids`, `chosen_mask`, `rejected_input_ids`, `rejected_mask`.
DEPENDENCY:
    - Phụ thuộc `torch.utils.data.Dataset`.
==============================================================================
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any


class PreferenceDataset(Dataset):
    """Đơn trách nhiệm: Nạp và sinh batch cho quá trình huấn luyện DPO."""

    def __init__(self, data: List[Dict[str, str]], tokenizer=None, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        return {
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        }
