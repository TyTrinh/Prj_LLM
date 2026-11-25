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

import os
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn


class CheckpointManager:
    """Quản lý nạp và lưu trữ trạng thái huấn luyện toàn diện hỗ trợ Auto-Resume."""

    @staticmethod
    def save_checkpoint(
        filepath: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        step: int = 0,
        epoch: int = 0,
        best_val_loss: float = float("inf"),
        config: Optional[Any] = None,
        stage: str = "pretrain",
    ) -> None:
        """Lưu toàn bộ trạng thái huấn luyện để phục vụ Resume."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
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
        torch.save(checkpoint_dict, filepath)

    @staticmethod
    def load_checkpoint(
        filepath: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Nạp toàn bộ trạng thái huấn luyện từ checkpoint file để resume.

        Trả về metadata chứa: step, epoch, best_val_loss, config, stage.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file checkpoint tại: {filepath}")

        checkpoint = torch.load(filepath, map_location=device)

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
