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
import pytest
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


def test_checkpoint_rollback_strategy():
    temp_dir = tempfile.mkdtemp()
    try:
        from src.utils.checkpoint import CheckpointRollbackManager

        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Lưu checkpoint an toàn ban đầu
        safe_ckpt = os.path.join(temp_dir, "step_500.pt")
        CheckpointManager.save_checkpoint(
            path=safe_ckpt,
            model=model,
            optimizer=optimizer,
            step=500,
            best_val_loss=1.5,
        )

        manager = CheckpointRollbackManager(
            checkpoint_dir=temp_dir,
            threshold_factor=3.0,
            min_grad_norm_threshold=3.0,
            lr_reduction_ratio=0.20,
            recovery_steps=10,
        )
        manager.register_safe_checkpoint(safe_ckpt)

        # Bước 1: Gradient norm bình thường (EMA khởi tạo)
        assert manager.check_gradient_anomaly(1.0) is False
        assert manager.check_gradient_anomaly(1.2) is False

        # Bước 2: Phát hiện gradient spike (bùng nổ > 3x EMA)
        spike_norm = 10.0
        assert manager.check_gradient_anomaly(spike_norm) is True

        # Bước 3: Thay đổi trọng số mô hình thành rác (giả lập gradient nổ làm hỏng weights)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)

        # Giả lập GradScaler unscaled state để kiểm tra reset an toàn
        scaler = torch.amp.GradScaler("cpu")
        dummy_loss = (sum(p.sum() for p in model.parameters()))
        scaler.scale(dummy_loss).backward()
        scaler.unscale_(optimizer)

        # Bước 4: Kích hoạt Rollback tự động
        meta = manager.trigger_rollback(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device="cpu",
            failed_step=523,
            reason="Gradient norm spike: 10.0 > 3.0 * EMA",
        )

        # Kiểm tra mô hình đã được phục hồi về step 500
        assert meta["step"] == 500
        for p in model.parameters():
            assert not torch.allclose(p, torch.tensor([999.0]))

        # Kiểm tra scaler đã được reset an toàn: gọi unscale_ tiếp không bị crash
        dummy_loss2 = (sum(p.sum() for p in model.parameters()))
        scaler.scale(dummy_loss2).backward()
        scaler.unscale_(optimizer)  # Phải thành công, không ném RuntimeError!

        # Kiểm tra siêu tham số được điều chỉnh: LR giảm 20%
        current_lr = optimizer.param_groups[0]["lr"]
        assert current_lr == pytest.approx(1e-3 * 0.8)
        assert manager.in_recovery is True

        # Bước 5: Qua chu kỳ phục hồi 10 steps
        for _ in range(10):
            manager.step_recovery(optimizer)
        assert manager.in_recovery is False
    finally:
        shutil.rmtree(temp_dir)

