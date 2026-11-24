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

__all__ = ["dpo_loss", "get_batch_logps", "PreferenceDataset"]
