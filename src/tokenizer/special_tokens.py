"""
==============================================================================
FILE: src/tokenizer/special_tokens.py
MODULE: src.tokenizer.special_tokens
RESPONSIBILITY:
    - Định nghĩa và quản lý hằng số Special Tokens.
    - Khóa cứng ID cố định (0..4) ngay từ đầu để đảm bảo tính nhất quán
      cho Loss Masking (ignore_index=0 hoặc -100) và Chat Template.
==============================================================================
"""

from typing import Dict, List

PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"

# Khóa cứng ID cố định cho Special Tokens
SPECIAL_TOKENS_MAP: Dict[str, int] = {
    PAD_TOKEN: 0,
    BOS_TOKEN: 1,
    EOS_TOKEN: 2,
    USER_TOKEN: 3,
    ASSISTANT_TOKEN: 4,
}

# Danh sách token theo thứ tự ID
DEFAULT_SPECIAL_TOKENS: List[str] = [
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
]

PAD_ID = SPECIAL_TOKENS_MAP[PAD_TOKEN]
BOS_ID = SPECIAL_TOKENS_MAP[BOS_TOKEN]
EOS_ID = SPECIAL_TOKENS_MAP[EOS_TOKEN]
USER_ID = SPECIAL_TOKENS_MAP[USER_TOKEN]
ASSISTANT_ID = SPECIAL_TOKENS_MAP[ASSISTANT_TOKEN]
