"""
==============================================================================
FILE: src/tokenizer/special_tokens.py
MODULE: src.tokenizer.special_tokens
RESPONSIBILITY:
    - Đơn trách nhiệm: Định nghĩa và quản lý hằng số Special Tokens.
    - Đảm bảo tính nhất quán của ID và ký hiệu token điều khiển hệ thống:
      <|pad|>, <|bos|>, <|eos|>, <|user|>, <|assistant|>
==============================================================================
"""

PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"

DEFAULT_SPECIAL_TOKENS = [
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
]
