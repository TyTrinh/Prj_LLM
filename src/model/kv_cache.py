"""
==============================================================================
FILE: src/model/kv_cache.py
MODULE: src.model.kv_cache
RESPONSIBILITY:
    - Đơn trách nhiệm: Quản lý bộ đệm tensor động lưu trữ Key và Value (KV Cache).
    - Phục vụ quá trình giải mã tự hồi quy (Autoregressive Decoding):
      Lưu trữ tensor [B, n_heads, seq_len, head_dim] cho từng tầng Transformer,
      cho phép ghép nối (concat) token mới mà không phải tính lại toàn bộ quá khứ.
INPUT:
    - new_k, new_v: torch.Tensor của token hiện tại [B, n_heads, 1, head_dim]
    - layer_idx: int
OUTPUT:
    - full_k, full_v: torch.Tensor tích lũy [B, n_heads, total_len, head_dim]
DEPENDENCY:
    - Chỉ phụ thuộc `torch`. Độc lập với cơ chế tính Attention.
==============================================================================
"""

import torch
from typing import Optional, Tuple, List


class KVCache:
    """Đơn trách nhiệm: Bộ đệm tensor Key-Value phục vụ tối ưu hóa suy luận."""

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.k_cache: List[Optional[torch.Tensor]] = [None] * n_layers
        self.v_cache: List[Optional[torch.Tensor]] = [None] * n_layers

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cập nhật key và value mới vào cache và trả về toàn bộ chuỗi lịch sử.
        
        Args:
            key: Tensor shape [B, n_heads, T, head_dim]
            value: Tensor shape [B, n_heads, T, head_dim]
            layer_idx: Chỉ số tầng Transformer
            
        Returns:
            Tuple[Tensor, Tensor]: (full_key, full_value) shape [B, n_heads, total_len, head_dim]
        """
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = key
            self.v_cache[layer_idx] = value
        else:
            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], key], dim=2)
            self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], value], dim=2)

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Lấy độ dài chuỗi hiện tại đã lưu trữ trong cache."""
        if self.k_cache[layer_idx] is None:
            return 0
        return self.k_cache[layer_idx].shape[2]

    def reset(self) -> None:
        """Xóa toàn bộ cache sau khi kết thúc 1 phiên sinh từ."""
        self.k_cache = [None] * self.n_layers
        self.v_cache = [None] * self.n_layers
