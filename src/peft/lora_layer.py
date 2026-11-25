"""
==============================================================================
FILE: src/peft/lora_layer.py
MODULE: src.peft.lora_layer
RESPONSIBILITY:
    - Đơn trách nhiệm: Triển khai lớp Low-Rank Adaptation (LoRALinear) từ số 0.
    - Bọc quanh lớp `nn.Linear` có sẵn, thêm nhánh phân rã hạng thấp song song:
      $h = W_0 x + \frac{\alpha}{r} B A x$
    - A in R^(r x d_in) khởi tạo phân phối Gauss chuẩn N(0, 1/r), B in R^(d_out x r) khởi tạo bằng 0.
    - Hỗ trợ công tắc bật/tắt `enabled: bool` phục vụ cơ chế DPO tham chiếu.
    - Hỗ trợ gập trọng số `merge()` / `unmerge()` và chuyển đổi hoàn toàn `merge_and_unload()`.
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
    """Module ma trận phân rã hạng thấp LoRA từ số 0."""

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
        self.scaling = (lora_alpha / r) if r > 0 else 1.0
        self.enabled = True
        self.merged = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Khởi tạo ma trận A và B trên cùng device và dtype với base_layer
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        if r > 0:
            self.lora_A = nn.Parameter(
                torch.empty((r, in_features), device=device, dtype=dtype)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, r), device=device, dtype=dtype)
            )
            # Khởi tạo ma trận A theo phân phối chuẩn Gaussian N(0, 1/r) theo chuẩn paper LoRA gốc (Hu et al., 2021)
            nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / math.sqrt(r))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.dropout = (
            nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if not self.enabled or self.merged or self.r == 0:
            return base_out

        # h = W_0 x + (alpha / r) * (dropout(x) @ A^T) @ B^T
        lora_x = self.dropout(x)
        lora_out = (lora_x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    def merge(self) -> None:
        """Gập trọng số LoRA trực tiếp vào base_layer (In-place merge)."""
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

    def merge_and_unload(self) -> nn.Linear:
        """Gập vĩnh viễn trọng số LoRA vào base_layer và trả về nn.Linear thuần cho zero-latency inference."""
        self.merge()
        return self.base_layer
