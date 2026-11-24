"""
==============================================================================
FILE: src/utils/checkpoint.py
MODULE: src.utils.checkpoint
RESPONSIBILITY:
    - Đơn trách nhiệm: Lưu trữ và nạp các checkpoint mô hình (Serialization).
    - Hỗ trợ lưu tách biệt:
      1. Base Model Checkpoint: Toàn bộ state_dict của kiến trúc Transformer.
      2. LoRA Adapter Checkpoint: Chỉ lưu các ma trận `lora_A` và `lora_B` (cực nhẹ, vài MB).
INPUT:
    - model: nn.Module
    - save_path: str
OUTPUT:
    - File nhị phân .pt lưu trữ an toàn.
==============================================================================
"""

import os
import torch
import torch.nn as nn


class CheckpointManager:
    """Đơn trách nhiệm: Quản lý nạp và lưu trữ trọng số mô hình."""

    @staticmethod
    def save_base_model(model: nn.Module, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)

    @staticmethod
    def load_base_model(model: nn.Module, path: str, device: str = "cpu") -> nn.Module:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def save_lora_adapter(model: nn.Module, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
        torch.save(lora_state, path)

    @staticmethod
    def load_lora_adapter(model: nn.Module, path: str, device: str = "cpu") -> nn.Module:
        lora_state = torch.load(path, map_location=device)
        model.load_state_dict(lora_state, strict=False)
        return model
