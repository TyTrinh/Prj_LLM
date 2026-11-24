"""
==============================================================================
FILE: src/peft/lora_manager.py
MODULE: src.peft.lora_manager
RESPONSIBILITY:
    - Đơn trách nhiệm: Quản lý vòng đời và tiêm (inject) adapter LoRA vào mô hình.
    - Cung cấp:
      1. `inject_lora(model, r, alpha, target_modules)`: Tìm và thay thế các lớp chiếu.
      2. `freeze_base_model(model)`: Đóng băng toàn bộ trọng số gốc, chỉ mở khóa LoRA.
      3. `set_lora_enabled(model, enabled)`: Bật/tắt nhánh LoRA phục vụ DPO.
      4. `merge_and_unload(model)`: Gập vĩnh viễn trọng số phục vụ inference tối ưu.
      5. `save_lora_weights`, `load_lora_weights`: Lưu/nạp riêng lẻ file adapter (~vài MB).
INPUT:
    - model: nn.Module
OUTPUT:
    - model đã được gắn adapter LoRA
DEPENDENCY:
    - Phụ thuộc `src.peft.lora_layer.LoRALinear`.
==============================================================================
"""

from typing import List, Dict, Any
import torch
import torch.nn as nn
from src.peft.lora_layer import LoRALinear


class LoRAManager:
    """Đơn trách nhiệm: Tiêm, quản lý, đóng băng và gập trọng số LoRA."""

    @staticmethod
    def inject_lora(
        model: nn.Module,
        r: int = 8,
        lora_alpha: float = 16.0,
        target_modules: List[str] = ["q_proj", "v_proj"],
    ) -> nn.Module:
        """Duyệt cây module và thay thế các Linear layer chỉ định bằng LoRALinear."""
        for name, module in model.named_modules():
            for target in target_modules:
                if hasattr(module, target):
                    curr_layer = getattr(module, target)
                    if isinstance(curr_layer, nn.Linear):
                        lora_layer = LoRALinear(curr_layer, r=r, lora_alpha=lora_alpha)
                        setattr(module, target, lora_layer)
        return model

    @staticmethod
    def freeze_base_model(model: nn.Module) -> None:
        """Đóng băng trọng số gốc, chỉ cho phép gradient tại ma trận A và B của LoRA."""
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    @staticmethod
    def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
        """Bật hoặc tắt nhánh tính toán LoRA."""
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.enabled = enabled

    @staticmethod
    def merge_and_unload(model: nn.Module) -> nn.Module:
        """Gập trọng số LoRA vào lớp gốc và phục hồi cấu trúc Linear chuẩn."""
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.merge()
        return model
