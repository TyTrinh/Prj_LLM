"""
==============================================================================
FILE: src/tokenizer/bpe.py
MODULE: src.tokenizer.bpe
RESPONSIBILITY:
    - Thuật toán Byte-Pair Encoding (BPE) với cơ chế Byte-Fallback viết từ số 0.
    - Cấu trúc không gian từ điển (Vocab 32K):
        + IDs 0..4: Special Tokens (<|pad|>=0, <|bos|>=1, <|eos|>=2, <|user|>=3, <|assistant|>=4)
        + IDs 5..260: 256 Raw Bytes (0x00..0xFF) để đảm bảo Byte-Fallback không bao giờ OOV/UNK
        + IDs 261..31999: Các token ghép cặp (Merges)
    - Tối ưu hóa huấn luyện BPE:
        + Tiền phân tách bằng GPT-4 Regex với thư viện `regex` (Unicode properties \\p{L}, \\p{N})
        + Duyệt theo bảng tần suất từ (word_counts) kết hợp Inverted Index để tránh bottleneck
    - Đầy đủ hàm encode, decode, save, load và round-trip test.
==============================================================================
"""

import json
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple
import regex as re

from src.tokenizer.special_tokens import (
    SPECIAL_TOKENS_MAP,
    DEFAULT_SPECIAL_TOKENS,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
)

# GPT-4 / LLaMA-3 standard splitting regex with unicode properties
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
COMPILED_PATTERN = re.compile(GPT4_SPLIT_PATTERN)

# Offset bắt đầu của 256 raw bytes ngay sau Special tokens
NUM_SPECIAL = len(SPECIAL_TOKENS_MAP)  # 5
BYTE_OFFSET = NUM_SPECIAL               # ID 5 đến 260 cho 256 bytes


class BytePairTokenizer:
    """
    Tokenizer BPE với Byte-fallback và không gian Vocab 32K.
    """

    def __init__(self, vocab_size: int = 32000):
        if vocab_size < BYTE_OFFSET + 256:
            raise ValueError(f"vocab_size phải lớn hơn ít nhất {BYTE_OFFSET + 256} để chứa Special tokens và 256 bytes.")
        self.vocab_size = vocab_size

        # ID -> bytes
        self.vocab: Dict[int, bytes] = {}
        # bytes -> ID
        self.inverse_vocab: Dict[bytes, int] = {}
        # (id1, id2) -> merged_id
        self.merges: Dict[Tuple[int, int], int] = {}
        # special token str -> ID
        self.special_tokens: Dict[str, int] = dict(SPECIAL_TOKENS_MAP)
        self.inverse_special_tokens: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}

        self._init_base_vocab()

    def _init_base_vocab(self) -> None:
        """Khởi tạo 256 byte cơ bản và special tokens trong vocab."""
        self.vocab.clear()
        self.inverse_vocab.clear()

        # 1. Đăng ký Special tokens vào vocab (lưu dạng UTF-8 bytes)
        for token_str, token_id in self.special_tokens.items():
            token_bytes = token_str.encode("utf-8")
            self.vocab[token_id] = token_bytes
            self.inverse_vocab[token_bytes] = token_id

        # 2. Đăng ký 256 raw bytes (ID: 5 .. 260)
        for b in range(256):
            token_id = BYTE_OFFSET + b
            token_bytes = bytes([b])
            self.vocab[token_id] = token_bytes
            self.inverse_vocab[token_bytes] = token_id

    def _get_special_tokens_regex(self) -> Optional[re.Pattern]:
        """Tạo regex pattern để bóc tách special tokens mà không bị tách vụn."""
        if not self.special_tokens:
            return None
        escaped_tokens = [re.escape(tok) for tok in self.special_tokens.keys()]
        pattern = "(" + "|".join(escaped_tokens) + ")"
        return re.compile(pattern)

    def train(self, texts: Iterable[str], verbose: bool = False) -> None:
        """
        Huấn luyện BPE trên tập văn bản thô:
        - Áp dụng GPT-4 Regex tách từ
        - Đếm tần suất từ (word_counts: Tuple[int, ...] -> frequency)
        - Inverted index gom các cặp (pair -> set of words) để cập nhật cục bộ cực nhanh.
        """
        self._init_base_vocab()
        self.merges.clear()

        # 1. Thu thập tần suất các từ/chunk (word_counts)
        word_counts: Dict[Tuple[int, ...], int] = defaultdict(int)
        special_regex = self._get_special_tokens_regex()

        for text in texts:
            if not text:
                continue

            # Tách riêng special tokens ra nếu có trong text
            if special_regex:
                chunks = special_regex.split(text)
            else:
                chunks = [text]

            for chunk in chunks:
                if not chunk or chunk in self.special_tokens:
                    continue
                # Dùng GPT-4 regex tách các phân đoạn ngôn ngữ
                for match in COMPILED_PATTERN.finditer(chunk):
                    word_bytes = match.group(0).encode("utf-8")
                    if not word_bytes:
                        continue
                    # Biểu diễn từ thành tuple các byte-token IDs
                    token_tuple = tuple(BYTE_OFFSET + b for b in word_bytes)
                    word_counts[token_tuple] += 1

        # 2. Xây dựng Inverted Index: pair -> Set[word_tuple] và pair_counts: pair -> int
        pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        where_pair: Dict[Tuple[int, int], Set[Tuple[int, ...]]] = defaultdict(set)

        for word, count in word_counts.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += count
                where_pair[pair].add(word)

        num_merges = self.vocab_size - len(self.vocab)
        next_id = BYTE_OFFSET + 256  # 261

        if verbose:
            print(f"Bắt đầu huấn luyện BPE: Cần thực hiện {num_merges} lượt merge để đạt vocab_size={self.vocab_size}...")

        # 3. Vòng lặp merge tối ưu
        for step in range(num_merges):
            if not pair_counts:
                break

            # Lấy cặp có tần suất xuất hiện cao nhất
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] <= 0:
                break

            # Gán ID mới
            new_id = next_id
            next_id += 1
            self.merges[best_pair] = new_id

            # Tạo byte representation cho token mới
            merged_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_id] = merged_bytes
            self.inverse_vocab[merged_bytes] = new_id

            # Cập nhật cục bộ chỉ trên các word chứa best_pair
            words_to_update = list(where_pair[best_pair])
            p0, p1 = best_pair

            for old_word in words_to_update:
                count = word_counts.pop(old_word)

                # Giảm thống kê của các cặp cũ trong old_word
                for i in range(len(old_word) - 1):
                    pair = (old_word[i], old_word[i + 1])
                    pair_counts[pair] -= count
                    if pair_counts[pair] <= 0:
                        pair_counts.pop(pair, None)
                    where_pair[pair].discard(old_word)
                    if not where_pair[pair]:
                        where_pair.pop(pair, None)

                # Tạo new_word bằng cách thay thế cặp (p0, p1) bằng new_id
                new_word_list = []
                i = 0
                while i < len(old_word):
                    if i < len(old_word) - 1 and old_word[i] == p0 and old_word[i + 1] == p1:
                        new_word_list.append(new_id)
                        i += 2
                    else:
                        new_word_list.append(old_word[i])
                        i += 1
                new_word = tuple(new_word_list)

                # Cập nhật word_counts với new_word
                word_counts[new_word] += count

                # Thêm thống kê của các cặp mới trong new_word
                for i in range(len(new_word) - 1):
                    pair = (new_word[i], new_word[i + 1])
                    pair_counts[pair] += count
                    where_pair[pair].add(new_word)

            if verbose and (step + 1) % 1000 == 0:
                print(f"Đã merge {step + 1}/{num_merges} cặp. Vocab hiện tại: {len(self.vocab)}")

        if verbose:
            print(f"Huấn luyện hoàn tất! Kích thước Vocab cuối cùng: {len(self.vocab)}, Số lượt merge: {len(self.merges)}")

    def _encode_chunk(self, chunk_bytes: bytes) -> List[int]:
        """Mã hóa một đoạn byte bằng các quy tắc merges đã học."""
        if not chunk_bytes:
            return []

        # Chuyển từng byte sang raw byte id (offset 5)
        tokens = [BYTE_OFFSET + b for b in chunk_bytes]

        while len(tokens) >= 2:
            # Tìm cặp liền kề có rank thấp nhất (được merge sớm nhất)
            stats = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges:
                    stats[pair] = self.merges[pair]

            if not stats:
                break

            best_pair = min(stats, key=stats.get)
            new_id = self.merges[best_pair]

            # Merge cặp đó
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str, allowed_special: Optional[Set[str]] = None) -> List[int]:
        """
        Mã hóa chuỗi văn bản thành danh sách token IDs:
        - Xử lý các Special tokens theo ID cố định (0..4).
        - Đoạn văn bản thường được phân rã theo Regex GPT-4 và encode qua BPE.
        - Mọi ký tự chưa thấy sẽ tự động fallback về Raw Bytes (5..260).
        """
        if not text:
            return []

        if allowed_special is None:
            # Mặc định cho phép tất cả các special tokens đã đăng ký
            allowed_special = set(self.special_tokens.keys())

        special_regex = self._get_special_tokens_regex()
        if not special_regex:
            parts = [text]
        else:
            parts = special_regex.split(text)

        ids: List[int] = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                if part in allowed_special:
                    ids.append(self.special_tokens[part])
                else:
                    # Nếu không cho phép special token, encode như text thô
                    for match in COMPILED_PATTERN.finditer(part):
                        ids.extend(self._encode_chunk(match.group(0).encode("utf-8")))
            else:
                for match in COMPILED_PATTERN.finditer(part):
                    ids.extend(self._encode_chunk(match.group(0).encode("utf-8")))

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Giải mã danh sách token IDs trở lại thành chuỗi UTF-8:
        - Nối toàn bộ byte của các token IDs
        - Special tokens được giữ nguyên dạng chuỗi
        - Sử dụng errors="replace" để đảm bảo tính an toàn nếu gặp byte lẻ.
        """
        if not ids:
            return ""

        parts: List[str] = []
        byte_buffer = bytearray()

        for token_id in ids:
            if token_id in self.inverse_special_tokens:
                if byte_buffer:
                    parts.append(byte_buffer.decode("utf-8", errors="replace"))
                    byte_buffer.clear()
                parts.append(self.inverse_special_tokens[token_id])
            elif token_id in self.vocab:
                byte_buffer.extend(self.vocab[token_id])
            else:
                # Trường hợp ID ngoài vocab (không hợp lệ)
                pass

        if byte_buffer:
            parts.append(byte_buffer.decode("utf-8", errors="replace"))

        return "".join(parts)

    def save(self, vocab_file: str, merges_file: str) -> None:
        """Lưu vocab.json và merges.txt."""
        # Lưu vocab: token_id -> hex-string để biểu diễn an toàn mọi bytes
        serializable_vocab = {
            token_id: token_bytes.hex()
            for token_id, token_bytes in self.vocab.items()
        }
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "vocab": serializable_vocab,
                "special_tokens": self.special_tokens,
            }, f, indent=2, ensure_ascii=False)

        # Lưu merges: (id1, id2) -> new_id
        with open(merges_file, "w", encoding="utf-8") as f:
            for (p0, p1), new_id in self.merges.items():
                f.write(f"{p0} {p1} {new_id}\n")

    def load(self, vocab_file: str, merges_file: str) -> None:
        """Tải vocab.json và merges.txt."""
        with open(vocab_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]
        self.special_tokens = data.get("special_tokens", dict(SPECIAL_TOKENS_MAP))
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

        self.vocab = {
            int(token_id): bytes.fromhex(hex_str)
            for token_id, hex_str in data["vocab"].items()
        }
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        self.merges.clear()
        with open(merges_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 3:
                    p0, p1, new_id = int(parts[0]), int(parts[1]), int(parts[2])
                    self.merges[(p0, p1)] = new_id
