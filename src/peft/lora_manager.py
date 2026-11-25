"""
==============================================================================
FILE: src/peft/lora_manager.py
MODULE: src.peft.lora_manager
RESPONSIBILITY:
    - Đơn trách nhiệm: Quản lý vòng đời và tiêm (inject) adapter LoRA vào mô hình.
    - Cung cấp:
      1. `inject_lora(model, r, alpha, target_modules)`: Tìm và thay thế các Linear layer chỉ định bằng LoRALinear.
      2. `freeze_base_model(model)`: Đóng băng toàn bộ trọng số gốc, chỉ mở khóa tham số LoRA (A và B).
      3. `set_lora_enabled(model, enabled)`: Bật/tắt nhánh LoRA linh hoạt (phục vụ DPO/so sánh).
      4. `merge_and_unload(model)`: Gập vĩnh viễn trọng số vào base_layer và dỡ bỏ LoRA wrapper cho zero-overhead inference.
      5. `save_lora_weights(model, path, metadata)`: Lưu chỉ riêng các trọng số của Adapter LoRA (~vài MB).
      6. `load_lora_weights(model, path, strict)`: Nạp các trọng số Adapter đã lưu vào mô hình đã inject LoRA.
      7. `get_trainable_parameters(model)`: Thống kê số lượng và tỷ lệ tham số có thể huấn luyện.
INPUT:
    - model: nn.Module
OUTPUT:
    - model đã được gắn adapter LoRA hoặc đã gập trọng số
DEPENDENCY:
    - Phụ thuộc `src.peft.lora_layer.LoRALinear`.
==============================================================================
"""

import os
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from src.peft.lora_layer import LoRALinear


class LoRAManager:
    """Đơn trách nhiệm: Tiêm, quản lý, đóng băng, gập trọng số và lưu/nạp adapter LoRA."""

    @staticmethod
    def inject_lora(
        model: nn.Module,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        target_modules: List[str] = ["q_proj", "v_proj"],
    ) -> nn.Module:
        """
        Duyệt cây module và thay thế các Linear layer chỉ định bằng LoRALinear.
        Trả về model (ghi nhận logging số lượng injected layers).
        """
        injected_count = 0
        for name, module in model.named_modules():
            for target in target_modules:
                if hasattr(module, target):
                    curr_layer = getattr(module, target)
                    if isinstance(curr_layer, nn.Linear) and not isinstance(curr_layer, LoRALinear):
                        lora_layer = LoRALinear(
                            curr_layer,
                            r=r,
                            lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout,
                        )
                        setattr(module, target, lora_layer)
                        injected_count += 1
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
        """Bật hoặc tắt nhánh tính toán LoRA trên toàn mô hình."""
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.enabled = enabled

    @staticmethod
    def merge_and_unload(model: nn.Module) -> nn.Module:
        """Gập vĩnh viễn trọng số LoRA vào lớp gốc và phục hồi hoàn toàn cấu trúc nn.Linear chuẩn."""
        # Duyệt module cha và thay thế child LoRALinear bằng base_layer đã merge
        for parent_module in model.modules():
            for child_name, child_module in list(parent_module.named_children()):
                if isinstance(child_module, LoRALinear):
                    unloaded_layer = child_module.merge_and_unload()
                    setattr(parent_module, child_name, unloaded_layer)
        return model

    @staticmethod
    def save_lora_weights(
        model: nn.Module,
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Chỉ lưu các trọng số thuộc về adapter LoRA (lora_A, lora_B) và metadata đi kèm."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        lora_state_dict = {}
        for name, param in model.named_parameters():
            if "lora_" in name:
                lora_state_dict[name] = param.data.cpu()

        payload = {
            "lora_state_dict": lora_state_dict,
            "metadata": metadata or {},
        }
        torch.save(payload, path)

    @staticmethod
    def load_lora_weights(
        model: nn.Module,
        path: str,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Nạp trọng số adapter LoRA vào mô hình đã inject LoRA."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"LoRA adapter file not found at: {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        lora_state_dict = checkpoint.get("lora_state_dict", checkpoint)

        model_params = dict(model.named_parameters())
        for name, param_data in lora_state_dict.items():
            if name in model_params:
                with torch.no_grad():
                    model_params[name].copy_(param_data)
            elif strict:
                raise KeyError(f"Parameter '{name}' in adapter not found in model.")

        return checkpoint.get("metadata", {})

    @staticmethod
    def get_trainable_parameters(model: nn.Module) -> Tuple[int, int, float]:
        """Thống kê (trainable_params, total_params, trainable_ratio)."""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_ratio = (trainable_params / total_params) if total_params > 0 else 0.0
        return trainable_params, total_params, trainable_ratio
