"""
==============================================================================
FILE: src/utils/seed.py
MODULE: src.utils.seed
RESPONSIBILITY:
    - Đơn trách nhiệm: Thiết lập Random Seed toàn cục để đảm bảo khả năng tái lập kết quả (Reproducibility).
    - Cố định seed cho: Python random, NumPy, PyTorch CPU và PyTorch CUDA.
INPUT:
    - seed: int (mặc định: 42)
OUTPUT:
    - None
==============================================================================
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Cố định seed trên tất cả các thư viện ngẫu nhiên."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
