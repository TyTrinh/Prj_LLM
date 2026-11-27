"""
WSD (Warmup - Stable - Decay) Learning Rate Scheduler.

Đặc tả kỹ thuật theo chuẩn huấn luyện mô hình ngôn ngữ hiện đại (MiniCPM, DeepSeek, Llama-3 data annealing):
1. Warmup Phase (1% - 5% tổng số bước):
   Tăng tuyến tính learning rate từ 0 lên giá trị cực đại eta_max.
2. Stable Phase (85% - 90% thời gian):
   Duy trì learning rate ổn định tuyệt đối ở mức eta_max. Trong giai đoạn này,
   mô hình liên tục hấp thụ tri thức thế giới mà không bị "đóng băng" trọng số.
3. Decay Phase (10% cuối cùng):
   Khi chốt phiên bản phát hành hoặc kích hoạt giai đoạn nạp dữ liệu chất lượng cao (Data Annealing),
   hạ learning rate nhanh chóng (theo hàm cosine hoặc exponential) về 0.1 * eta_max.
"""

import math
from typing import Literal, Union
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def get_wsd_lr(
    step: int,
    max_steps: int,
    max_lr: float,
    min_lr: float | None = None,
    warmup_ratio: float = 0.03,
    decay_ratio: float = 0.10,
    decay_type: Literal["cosine", "linear", "exponential"] = "cosine",
) -> float:
    """
    Tính learning rate tại một bước `step` theo lịch trình WSD (Warmup - Stable - Decay).

    Args:
        step: Bước tối ưu hiện tại (0-indexed).
        max_steps: Tổng số bước huấn luyện dự kiến.
        max_lr: Tốc độ học cực đại (eta_max).
        min_lr: Tốc độ học tối thiểu cuối giai đoạn decay (mặc định = 0.1 * max_lr).
        warmup_ratio: Tỷ lệ số bước Warmup (thường 0.01 -> 0.05, mặc định 0.03 = 3%).
        decay_ratio: Tỷ lệ số bước Decay ở cuối (thường 0.10 -> 0.15, mặc định 0.10 = 10%).
        decay_type: Dạng hàm hạ learning rate ('cosine', 'linear', 'exponential').

    Returns:
        float: Learning rate tương ứng tại `step`.
    """
    if min_lr is None:
        min_lr = 0.1 * max_lr

    warmup_steps = int(max_steps * warmup_ratio)
    decay_steps = int(max_steps * decay_ratio)
    stable_end_step = max_steps - decay_steps

    # 1. Giai đoạn Warmup: 0 -> warmup_steps
    if step < warmup_steps:
        if warmup_steps <= 0:
            return max_lr
        return max_lr * (step + 1) / warmup_steps

    # 2. Giai đoạn Stable: warmup_steps -> stable_end_step
    if step < stable_end_step:
        return max_lr

    # 3. Quá max_steps: Giữ ở min_lr
    if step >= max_steps:
        return min_lr

    # 4. Giai đoạn Decay: stable_end_step -> max_steps
    progress = (step - stable_end_step) / max(decay_steps, 1)

    if decay_type == "cosine":
        # Cosine decay từ max_lr xuống min_lr
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + coeff * (max_lr - min_lr)
    elif decay_type == "linear":
        # Tuyến tính từ max_lr xuống min_lr
        return max_lr - progress * (max_lr - min_lr)
    elif decay_type == "exponential":
        # Lũy thừa dốc
        return max_lr * math.pow(min_lr / max_lr, progress)
    else:
        raise ValueError(f"Không hỗ trợ decay_type: {decay_type}. Chọn 'cosine', 'linear', hoặc 'exponential'.")


class WSDScheduler(_LRScheduler):
    """
    PyTorch-compatible LRScheduler cho WSD (Warmup - Stable - Decay).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_steps: int,
        warmup_ratio: float = 0.03,
        decay_ratio: float = 0.10,
        min_lr_ratio: float = 0.10,
        decay_type: Literal["cosine", "linear", "exponential"] = "cosine",
        last_epoch: int = -1,
    ):
        self.max_steps = max_steps
        self.warmup_ratio = warmup_ratio
        self.decay_ratio = decay_ratio
        self.min_lr_ratio = min_lr_ratio
        self.decay_type = decay_type
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        return [
            get_wsd_lr(
                step=step,
                max_steps=self.max_steps,
                max_lr=base_lr,
                min_lr=base_lr * self.min_lr_ratio,
                warmup_ratio=self.warmup_ratio,
                decay_ratio=self.decay_ratio,
                decay_type=self.decay_type,
            )
            for base_lr in self.base_lrs
        ]
