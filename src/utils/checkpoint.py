"""
==============================================================================
FILE: src/utils/checkpoint.py
MODULE: src.utils.checkpoint
RESPONSIBILITY:
    - Đơn trách nhiệm: Quản lý lưu trữ và nạp trạng thái huấn luyện (Checkpointing & Serialization).
    - Hỗ trợ đầy đủ cơ chế Resume Training:
      1. Lưu toàn diện State Dict: Model, Optimizer, LR Scheduler, GradScaler, Metadata (step, epoch, best_loss, config).
      2. Tự động phục hồi trạng thái để huấn luyện tiếp tục mà không bị gián đoạn hay reset learning rate/momentum.
      3. Lưu và nạp riêng rẽ trọng số sạch (Inference weights): Base Model (.pt) và LoRA Adapters (.pt).
INPUT:
    - model: nn.Module
    - optimizer, scheduler, scaler (tùy chọn)
    - filepath: str
OUTPUT:
    - File nhị phân .pt an toàn, tương thích PyTorch.
==============================================================================
"""

import math
import os
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn


class CheckpointManager:
    """Quản lý nạp và lưu trữ trạng thái huấn luyện toàn diện hỗ trợ Auto-Resume."""

    @staticmethod
    def save_checkpoint(
        filepath: Optional[str] = None,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        step: int = 0,
        epoch: int = 0,
        best_val_loss: float = float("inf"),
        config: Optional[Any] = None,
        stage: str = "pretrain",
        path: Optional[str] = None,
    ) -> None:
        """Lưu toàn bộ trạng thái huấn luyện để phục vụ Resume (chấp nhận cả `filepath` hoặc `path`)."""
        target_path = filepath or path
        if target_path is None:
            raise ValueError("Cần cung cấp 'filepath' hoặc 'path' để lưu checkpoint.")
        if model is None:
            raise ValueError("Cần cung cấp 'model' để lưu checkpoint.")

        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        checkpoint_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": config,
            "stage": stage,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        torch.save(checkpoint_dict, target_path)

    @staticmethod
    def load_checkpoint(
        filepath: Optional[str] = None,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        device: str = "cpu",
        path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Nạp toàn bộ trạng thái huấn luyện từ checkpoint file để resume.

        Trả về metadata chứa: step, epoch, best_val_loss, config, stage.
        """
        target_path = filepath or path
        if target_path is None or not os.path.exists(target_path):
            raise FileNotFoundError(f"Không tìm thấy file checkpoint tại: {target_path}")

        checkpoint = torch.load(target_path, map_location=device)

        # Nạp trạng thái mô hình
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Hỗ trợ trường hợp file chỉ chứa state_dict thuần
            model.load_state_dict(checkpoint)

        # Nạp trạng thái optimizer nếu có
        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Nạp trạng thái lr scheduler nếu có
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Nạp trạng thái GradScaler nếu có
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Khôi phục trạng thái ngẫu nhiên an toàn (chuyển sang ByteTensor và CPU)
        try:
            if "rng_state" in checkpoint and checkpoint["rng_state"] is not None:
                rng = checkpoint["rng_state"]
                if isinstance(rng, torch.Tensor):
                    torch.set_rng_state(rng.to(dtype=torch.uint8, device="cpu"))
            if (
                torch.cuda.is_available()
                and device.startswith("cuda")
                and checkpoint.get("cuda_rng_state") is not None
            ):
                cuda_rng = checkpoint["cuda_rng_state"]
                if isinstance(cuda_rng, torch.Tensor):
                    torch.cuda.set_rng_state(cuda_rng.to(dtype=torch.uint8, device="cpu"))
        except Exception as e:
            # Nếu RNG state không tương thích giữa các phiên bản PyTorch, bỏ qua mà không làm crash training
            print(f"[Checkpoint Info] Bỏ qua khôi phục RNG state do khác biệt môi trường ({e}).")

        return {
            "step": checkpoint.get("step", 0),
            "epoch": checkpoint.get("epoch", 0),
            "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
            "config": checkpoint.get("config", None),
            "stage": checkpoint.get("stage", "pretrain"),
        }

    @staticmethod
    def save_base_model(model: nn.Module, path: str) -> None:
        """Lưu file trọng số sạch của Base Model cho inference."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(model.state_dict(), path)

    @staticmethod
    def load_base_model(model: nn.Module, path: str, device: str = "cpu") -> nn.Module:
        """Nạp file trọng số sạch của Base Model."""
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def save_lora_adapter(model: nn.Module, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Lưu ma trận LoRA Adapter theo định dạng chuẩn thống nhất với LoRAManager."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        lora_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k}
        payload = {"lora_state_dict": lora_state, "metadata": metadata or {}}
        torch.save(payload, path)

    @staticmethod
    def load_lora_adapter(model: nn.Module, path: str, device: str = "cpu") -> nn.Module:
        """Nạp các ma trận LoRA Adapter vào mô hình tương thích cả 2 định dạng."""
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        lora_state = checkpoint.get("lora_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(lora_state, strict=False)
        return model


class CheckpointRollbackManager:
    """
    Hệ thống giám sát và ứng cứu Checkpoint Rollback tự động.

    Chiến lược 4 bước:
    1. Phát hiện ngưỡng bất thường: Nếu norm gradient ||\nabla_\theta L|| > Threshold * MovingAverage,
       hoặc loss/gradient bị NaN/Inf, kích hoạt còi báo động.
    2. Kích hoạt Rollback: Khôi phục trọng số mô hình và optimizer về checkpoint an toàn gần nhất.
    3. Cô lập và loại bỏ batch lỗi: Bỏ qua (skip) batch dữ liệu gây bất ổn.
    4. Điều chỉnh siêu tham số: Giảm learning rate tạm thời (ví dụ 20%) và siết chặt Gradient Clipping
       để bảo vệ quá trình huấn luyện trước khi tăng dần trở lại.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        threshold_factor: float = 5.0,
        min_grad_norm_threshold: float = 5.0,
        ema_alpha: float = 0.95,
        lr_reduction_ratio: float = 0.20,
        tight_clip_factor: float = 0.50,
        recovery_steps: int = 50,
    ):
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.threshold_factor = threshold_factor
        self.min_grad_norm_threshold = min_grad_norm_threshold
        self.ema_alpha = ema_alpha
        self.lr_reduction_ratio = lr_reduction_ratio
        self.tight_clip_factor = tight_clip_factor
        self.recovery_steps = recovery_steps

        # Trạng thái theo dõi động
        self.grad_norm_ema: Optional[float] = None
        self.last_safe_checkpoint_path: Optional[str] = None
        self.in_recovery: bool = False
        self.recovery_remaining: int = 0
        self.current_clip_multiplier: float = 1.0
        self.current_lr_multiplier: float = 1.0
        self.rollback_history: list = []

    def register_safe_checkpoint(self, checkpoint_path: str) -> None:
        """Đăng ký checkpoint an toàn vừa được lưu."""
        if os.path.exists(checkpoint_path):
            self.last_safe_checkpoint_path = os.path.abspath(checkpoint_path)

    def check_gradient_anomaly(self, grad_norm: float) -> bool:
        """
        Kiểm tra tính bất thường của gradient norm:
        Trả về True nếu phát hiện bất thường thực sự (NaN, Inf, hoặc vượt quá threshold_factor * EMA
        VÀ vượt qua ngưỡng an toàn tối thiểu min_grad_norm_threshold).
        """
        if math.isnan(grad_norm) or math.isinf(grad_norm):
            return True

        if self.grad_norm_ema is None:
            self.grad_norm_ema = grad_norm
            return False

        # Kiểm tra ngưỡng bất thường so với đường trung bình động (EMA)
        threshold = max(self.threshold_factor * self.grad_norm_ema, self.min_grad_norm_threshold)
        if grad_norm > threshold:
            return True

        # Cập nhật EMA khi gradient ở ngưỡng an toàn
        self.grad_norm_ema = self.ema_alpha * self.grad_norm_ema + (1.0 - self.ema_alpha) * grad_norm
        return False

    def find_latest_safe_checkpoint(self) -> Optional[str]:
        """Tìm file checkpoint an toàn mới nhất (ưu tiên last_safe_checkpoint, sau đó step_*.pt mới nhất, rồi last_checkpoint.pt)."""
        if self.last_safe_checkpoint_path and os.path.exists(self.last_safe_checkpoint_path):
            return self.last_safe_checkpoint_path

        import glob
        step_ckpts = glob.glob(os.path.join(self.checkpoint_dir, "step_*.pt"))
        if step_ckpts:
            return max(step_ckpts, key=os.path.getmtime)

        candidates = [
            os.path.join(self.checkpoint_dir, "last_checkpoint.pt"),
            os.path.join(self.checkpoint_dir, "base_model.pt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def trigger_rollback(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
        failed_step: int = 0,
        reason: str = "Gradient norm spike anomaly",
    ) -> Dict[str, Any]:
        """
        Thực thi quy trình Rollback về checkpoint an toàn gần nhất và reset trạng thái an toàn cho GradScaler.
        """
        target_ckpt = self.find_latest_safe_checkpoint()

        if target_ckpt is None or not os.path.exists(target_ckpt):
            raise FileNotFoundError(
                f"[Rollback Manager] Không tìm thấy bất kỳ checkpoint an toàn nào tại {self.checkpoint_dir} để rollback!"
            )

        print("=" * 70)
        print(f"🚨 [CẢNH BÁO ỨNG CỨU] Phát hiện sự cố tại Step {failed_step}: {reason}!")
        print(f"🔄 Đang kích hoạt Rollback tự động về Checkpoint an toàn: {target_ckpt}...")

        # 1. Rollback trọng số mô hình và trạng thái optimizer
        restored_meta = CheckpointManager.load_checkpoint(
            path=target_ckpt,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            device=device,
        )

        # 2. Xóa sạch gradients trên optimizer để loại bỏ hoàn toàn batch lỗi
        optimizer.zero_grad(set_to_none=True)

        # 3. QUAN TRỌNG: Reset trạng thái GradScaler để không bị lỗi "unscale_() has already been called" ở step kế tiếp
        if scaler is not None:
            scaler.update()

        # 4. Điều chỉnh siêu tham số: Giảm LR 20% và siết chặt Gradient Clipping
        self.in_recovery = True
        self.recovery_remaining = self.recovery_steps
        self.current_lr_multiplier = 1.0 - self.lr_reduction_ratio
        self.current_clip_multiplier = self.tight_clip_factor

        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * self.current_lr_multiplier

        rollback_record = {
            "failed_step": failed_step,
            "restored_step": restored_meta.get("step", 0),
            "restored_from": target_ckpt,
            "reason": reason,
            "lr_multiplier": self.current_lr_multiplier,
            "clip_multiplier": self.current_clip_multiplier,
        }
        self.rollback_history.append(rollback_record)

        print(f"✅ Rollback hoàn tất! Khôi phục về Step {restored_meta.get('step', 0)}.")
        print(f"⚡ Điều chỉnh an toàn: Giảm tạm thời 20% LR (x{self.current_lr_multiplier:.2f}) "
              f"và siết Gradient Clipping (x{self.tight_clip_factor:.2f}) trong {self.recovery_steps} steps tiếp theo.")
        print("=" * 70)

        return restored_meta
        self.rollback_history.append(rollback_record)

        print(f"✅ Rollback hoàn tất! Khôi phục về Step {restored_meta.get('step', 0)}.")
        print(f"⚡ Điều chỉnh an toàn: Giảm tạm thời 20% LR (x{self.current_lr_multiplier:.2f}) "
              f"và siết Gradient Clipping (x{self.tight_clip_factor:.2f}) trong {self.recovery_steps} steps tiếp theo.")
        print("=" * 70)

        return restored_meta

    def step_recovery(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Theo dõi và nới lỏng dần các hạn chế siêu tham số sau khi qua vùng sự cố.
        """
        if not self.in_recovery:
            return

        self.recovery_remaining -= 1
        if self.recovery_remaining <= 0:
            self.in_recovery = False
            self.current_lr_multiplier = 1.0
            self.current_clip_multiplier = 1.0
            print("🌱 [Rollback Manager] Đã qua vùng rủi ro, khôi phục lại tỷ lệ LR và Clipping bình thường.")

