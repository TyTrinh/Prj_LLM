"""
==============================================================================
FILE: src/tokenizer/__init__.py
MODULE: src.tokenizer
RESPONSIBILITY:
    - Đơn trách nhiệm: Expose BPE Tokenizer và Special Tokens.
==============================================================================
"""

from src.tokenizer.bpe import BytePairTokenizer
from src.tokenizer.special_tokens import (
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    DEFAULT_SPECIAL_TOKENS,
)

__all__ = [
    "BytePairTokenizer",
    "PAD_TOKEN",
    "BOS_TOKEN",
    "EOS_TOKEN",
    "USER_TOKEN",
    "ASSISTANT_TOKEN",
    "DEFAULT_SPECIAL_TOKENS",
]
