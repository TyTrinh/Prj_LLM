"""
==============================================================================
FILE: src/alignment/__init__.py
MODULE: src.alignment
RESPONSIBILITY:
    - Đơn trách nhiệm: Expose các module tính DPO loss và Preference Dataset.
==============================================================================
"""

from src.alignment.dpo_loss import dpo_loss, get_batch_logps
from src.alignment.preference_dataset import PreferenceDataset
from src.alignment.sft_dataset import (
    AlpacaSFTDataset,
    SFTDataCollator,
    compute_sft_loss,
    format_alpaca_prompt,
    format_sft_text,
)

__all__ = [
    "dpo_loss",
    "get_batch_logps",
    "PreferenceDataset",
    "AlpacaSFTDataset",
    "SFTDataCollator",
    "compute_sft_loss",
    "format_alpaca_prompt",
    "format_sft_text",
]
