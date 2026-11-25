"""
==============================================================================
FILE: src/utils/tracker.py
MODULE: src.utils.tracker
RESPONSIBILITY:
    - Quản lý quá trình theo dõi huấn luyện (Experiment Tracking) qua Weights & Biases (W&B).
    - Hỗ trợ log metrics (train/val loss, lr, perplexity, grad_norm, tokens/s, VRAM).
    - Hỗ trợ log text generation sample sau mỗi N steps/epochs.
    - Chống crash khi mất mạng hoặc không có API key (hỗ trợ chế độ offline/disabled an toàn).
==============================================================================
"""

from typing import Any, Dict, List, Optional
import wandb


class WandbTracker:
    """Wrapper quản lý Weights & Biases tracker cho LLM Training."""

    def __init__(
        self,
        project_name: str = "LLM-DecoderOnly-Scratch",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        mode: str = "online",  # "online", "offline", hoặc "disabled"
    ):
        self.enabled = enabled
        self.run = None
        self.wandb = None

        if not self.enabled:
            return

        try:
            self.wandb = wandb
            self.run = self.wandb.init(
                project=project_name,
                name=run_name,
                config=config,
                mode=mode,
                reinit=True,
            )
        except Exception as e:
            print(f"[Tracker Warning] Không thể khởi tạo W&B ({e}). Đang chuyển về chế độ vô hiệu hóa (disabled).")
            self.enabled = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Ghi nhận dict các chỉ số số học (loss, lr, perplexity, vram,...)."""
        if not self.enabled or self.run is None:
            return
        self.wandb.log(metrics, step=step)

    def log_text_samples(
        self,
        table_name: str,
        samples: List[Dict[str, Any]],
        step: Optional[int] = None,
    ) -> None:
        """
        Log bảng kết quả sinh văn bản thử nghiệm của mô hình.
        samples: danh sách dict, ví dụ: [{"prompt": "...", "generated": "..."}]
        """
        if not self.enabled or self.run is None or not samples:
            return

        columns = list(samples[0].keys())
        data = [[s.get(col, "") for col in columns] for s in samples]
        table = self.wandb.Table(columns=columns, data=data)
        self.wandb.log({table_name: table}, step=step)

    def finish(self) -> None:
        """Đóng session W&B khi kết thúc huấn luyện."""
        if self.enabled and self.run is not None:
            self.wandb.finish()
            self.run = None
