"""
==============================================================================
FILE: src/peft/lora_layer.py
MODULE: src.peft.lora_layer
RESPONSIBILITY:
    - Đơn trách nhiệm: Triển khai lớp Low-Rank Adaptation (LoRALinear) từ số 0.
    - Bọc quanh lớp `nn.Linear` có sẵn, thêm nhánh phân rã hạng thấp song song:
      $h = W_0 x + \frac{\alpha}{r} B A x$
    - Hỗ trợ công tắc bật/tắt `enabled: bool` phục vụ cơ chế DPO tham chiếu.
    - Hỗ trợ gập trọng số `merge_weights()` cho suy luận zero-overhead.
INPUT:
    - x: torch.Tensor [..., in_features]
OUTPUT:
    - out: torch.Tensor [..., out_features]
DEPENDENCY:
    - Chỉ phụ thuộc `torch` và `torch.nn`.
==============================================================================
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Đơn trách nhiệm: Module ma trận phân rã hạng thấp LoRA."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.enabled = True
        self.merged = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Ma trận A (khởi tạo Kaiming/Gauss) và B (khởi tạo số 0)
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if not self.enabled or self.merged or self.r == 0:
            return base_out
        lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    def merge(self) -> None:
        """Gập trọng số LoRA trực tiếp vào base_layer (Cải tiến Zero-Overhead)."""
        if not self.merged and self.r > 0:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta_w
            self.merged = True

    def unmerge(self) -> None:
        """Tách trọng số LoRA khỏi base_layer."""
        if self.merged and self.r > 0:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= delta_w
            self.merged = False
