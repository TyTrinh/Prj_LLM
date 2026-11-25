"""
==============================================================================
FILE: tests/test_checkpoint.py
RESPONSIBILITY:
    - Kiểm thử unit test cho CheckpointManager:
      1. Lưu và nạp Base Model state_dict thuần túy.
      2. Lưu và nạp toàn diện trạng thái huấn luyện (Model, Optimizer, Scheduler, Step, Loss) để Resume.
      3. Đảm bảo các tham số optimizer và weights sau khi resume khớp 100%.
==============================================================================
"""

import os
import shutil
import tempfile
import torch
import torch.nn as nn
from src.utils.checkpoint import CheckpointManager


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


def test_base_model_save_and_load():
    temp_dir = tempfile.mkdtemp()
    try:
        model = SimpleModel()
        save_path = os.path.join(temp_dir, "base_model.pt")

        CheckpointManager.save_base_model(model, save_path)
        assert os.path.exists(save_path)

        new_model = SimpleModel()
        CheckpointManager.load_base_model(new_model, save_path)

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)
    finally:
        shutil.rmtree(temp_dir)


def test_full_training_state_resume():
    temp_dir = tempfile.mkdtemp()
    try:
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        # Chạy 1 bước để cập nhật optimizer state
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

        save_path = os.path.join(temp_dir, "last_checkpoint.pt")
        CheckpointManager.save_checkpoint(
            filepath=save_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=100,
            epoch=2,
            best_val_loss=1.85,
            stage="pretrain",
        )

        assert os.path.exists(save_path)

        # Mô hình mới và optimizer mới
        resumed_model = SimpleModel()
        resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
        resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, step_size=1, gamma=0.5)

        meta = CheckpointManager.load_checkpoint(
            filepath=save_path,
            model=resumed_model,
            optimizer=resumed_optimizer,
            scheduler=resumed_scheduler,
            device="cpu",
        )

        assert meta["step"] == 100
        assert meta["epoch"] == 2
        assert abs(meta["best_val_loss"] - 1.85) < 1e-5
        assert meta["stage"] == "pretrain"

        # Kiểm tra trọng số mô hình khớp
        for p1, p2 in zip(model.parameters(), resumed_model.parameters()):
            assert torch.allclose(p1, p2)

        # Kiểm tra learning rate trong scheduler khớp
        assert resumed_scheduler.get_last_lr() == scheduler.get_last_lr()
    finally:
        shutil.rmtree(temp_dir)
