"""Centralized Configuration Module using Python dataclasses.

Clean Architecture principle:
All hyperparameters, architecture dimensions, and training settings
are strongly typed, centralized, and immutable across the codebase.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Hyperparameters for Decoder-Only Transformer."""
    vocab_size: int = 8192
    block_size: int = 512            # Maximum context window (L)
    n_layers: int = 8                # Number of Transformer blocks
    n_heads: int = 8                 # Number of attention heads
    d_model: int = 512               # Embedding dimension
    d_ffn: Optional[int] = None      # SwiGLU hidden dim (default: int(2/3 * 4 * d_model))
    rope_base: float = 10000.0       # Base frequency for RoPE
    norm_eps: float = 1e-5           # RMSNorm epsilon
    tie_weights: bool = True         # Weight tying: W_head = W_embed^T
    dropout: float = 0.0             # Dropout rate (0.0 for pre-training stability)

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        if self.d_ffn is None:
            # Standard SwiGLU formula: ~2/3 * 4 * d_model rounded to multiple of 64
            hidden = int(2 * (4 * self.d_model) / 3)
            self.d_ffn = ((hidden + 63) // 64) * 64


@dataclass
class TrainConfig:
    """Hyperparameters for Base Model Pre-training."""
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-4
    min_learning_rate: float = 5e-5
    warmup_steps: int = 500
    max_steps: int = 10000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    device: str = "cuda"
    mixed_precision: bool = True     # Use fp16 / bf16 autocast
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"


@dataclass
class LoRAConfig:
    """Configuration for Low-Rank Adaptation (PEFT)."""
    r: int = 8                       # LoRA rank
    lora_alpha: float = 16.0         # Scaling factor alpha
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    bias: str = "none"


@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization (Alignment)."""
    beta: float = 0.1                # DPO temperature parameter
    learning_rate: float = 1e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    max_prompt_length: int = 256
    max_response_length: int = 256
    length_normalization: bool = True  # Improvement: Prevent length bias
    epochs: int = 3


@dataclass
class GuardrailConfig:
    """Configuration for Input & Output Guardrails (Safety)."""
    enable_input_guard: bool = True
    enable_output_guard: bool = True
    max_input_length: int = 1024
    refusal_response: str = (
        "Xin lỗi, tôi không thể thực hiện yêu cầu này nhằm đảm bảo tính an toàn và chuẩn mực."
    )
    toxic_threshold: float = 0.5
