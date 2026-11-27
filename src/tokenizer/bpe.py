"""
==============================================================================
FILE: src/tokenizer/bpe.py
MODULE: src.tokenizer.bpe
RESPONSIBILITY:
    - Thuật toán Byte-Pair Encoding (BPE) với cơ chế Byte-Fallback viết từ số 0.
    - Cấu trúc không gian từ điển (Vocab 8K - 8192 tokens):
        + IDs 0..4: Special Tokens (<|pad|>=0, <|bos|>=1, <|eos|>=2, <|user|>=3, <|assistant|>=4)
        + IDs 5..260: 256 Raw Bytes (0x00..0xFF) để đảm bảo Byte-Fallback không bao giờ OOV/UNK
        + IDs 261..8191: 7931 token ghép cặp (Merges)
    - Tiền phân tách bằng GPT-4 Regex với thư viện `regex` (Unicode properties \\p{L}, \\p{N})
    - Duyệt theo bảng tần suất từ (word_counts) kết hợp Inverted Index để tối ưu tốc độ huấn luyện.
    - Đầy đủ hàm encode, decode, save, load với kiểm tra toàn vẹn chặt chẽ (Integrity Validation).
    - get_fingerprint() băm toàn vẹn: regex pattern, toàn bộ vocab, toàn bộ merges, special tokens.
==============================================================================
"""

import hashlib
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
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
    Tokenizer BPE với Byte-fallback và không gian Vocab 8K (8192 tokens).
    """

    def __init__(self, vocab_size: int = 8192):
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
                    token_tuple = tuple(BYTE_OFFSET + b for b in word_bytes)
                    word_counts[token_tuple] += 1

        # 2. Xây dựng Inverted Index
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

            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] <= 0:
                break

            new_id = next_id
            next_id += 1
            self.merges[best_pair] = new_id

            merged_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_id] = merged_bytes
            self.inverse_vocab[merged_bytes] = new_id

            words_to_update = list(where_pair[best_pair])
            p0, p1 = best_pair

            for old_word in words_to_update:
                count = word_counts.pop(old_word)

                for i in range(len(old_word) - 1):
                    pair = (old_word[i], old_word[i + 1])
                    pair_counts[pair] -= count
                    if pair_counts[pair] <= 0:
                        pair_counts.pop(pair, None)
                    where_pair[pair].discard(old_word)
                    if not where_pair[pair]:
                        where_pair.pop(pair, None)

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

                word_counts[new_word] += count

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

        tokens = [BYTE_OFFSET + b for b in chunk_bytes]

        while len(tokens) >= 2:
            stats = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges:
                    stats[pair] = self.merges[pair]

            if not stats:
                break

            best_pair = min(stats, key=stats.get)
            new_id = self.merges[best_pair]

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
        - Kiểm tra nghiêm ngặt tính hợp lệ của token ID: Ném lỗi nếu gặp ID ngoài vocab.
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
                # Tuân thủ Constitution Điều 8: Không bao giờ nuốt lỗi âm thầm (pass)
                raise ValueError(
                    f"Token ID không hợp lệ: {token_id}. "
                    f"ID này nằm ngoài phạm vi không gian từ điển [0, {self.vocab_size - 1}]."
                )

        if byte_buffer:
            parts.append(byte_buffer.decode("utf-8", errors="replace"))

        return "".join(parts)

    def _resolve_paths(self, path_or_vocab: str, merges_file: Optional[str] = None) -> Tuple[str, str]:
        """Hỗ trợ cả 2 dạng: truyền 1 thư mục (folder) hoặc truyền 2 file riêng rẽ."""
        if merges_file is None:
            if os.path.isdir(path_or_vocab) or not path_or_vocab.endswith(".json"):
                vocab_file = os.path.join(path_or_vocab, "vocab.json")
                merges_path = os.path.join(path_or_vocab, "merges.txt")
                return vocab_file, merges_path
            else:
                raise ValueError("Cần cung cấp cả vocab_file và merges_file hoặc 1 đường dẫn thư mục chứa chúng.")
        return path_or_vocab, merges_file

    def save(self, path_or_vocab: str, merges_file: Optional[str] = None) -> None:
        """Lưu vocab.json và merges.txt."""
        vocab_path, merges_path = self._resolve_paths(path_or_vocab, merges_file)
        os.makedirs(os.path.dirname(os.path.abspath(vocab_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(merges_path)), exist_ok=True)

        serializable_vocab = {
            token_id: token_bytes.hex()
            for token_id, token_bytes in self.vocab.items()
        }
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "split_pattern": GPT4_SPLIT_PATTERN,
                "special_tokens": self.special_tokens,
                "vocab": serializable_vocab,
            }, f, indent=2, ensure_ascii=False)

        with open(merges_path, "w", encoding="utf-8") as f:
            for (p0, p1), new_id in sorted(self.merges.items(), key=lambda x: x[1]):
                f.write(f"{p0} {p1} {new_id}\n")

    def save_vocab(self, path_or_vocab: str, merges_file: Optional[str] = None) -> None:
        """Alias cho phương thức save()."""
        self.save(path_or_vocab, merges_file)

    def load(self, path_or_vocab: str, merges_file: Optional[str] = None) -> None:
        """
        Tải và kiểm định tính toàn vẹn tuyệt đối của vocab.json và merges.txt.
        Ném ValueError nếu phát hiện dữ liệu bị rác, khuyết thiếu hoặc không nhất quán.
        """
        vocab_path, merges_path = self._resolve_paths(path_or_vocab, merges_file)

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Không tìm thấy file từ điển tại: {vocab_path}")
        if not os.path.exists(merges_path):
            raise FileNotFoundError(f"Không tìm thấy file merges tại: {merges_path}")

        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]
        self.special_tokens = data.get("special_tokens", dict(SPECIAL_TOKENS_MAP))
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

        raw_vocab = data["vocab"]
        self.vocab = {
            int(token_id): bytes.fromhex(hex_str)
            for token_id, hex_str in raw_vocab.items()
        }
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        # 1. Kiểm tra tính liên tục tuyệt đối của vocab: từ 0 đến vocab_size - 1
        expected_ids = set(range(self.vocab_size))
        actual_ids = set(self.vocab.keys())
        if actual_ids != expected_ids:
            missing = expected_ids - actual_ids
            extra = actual_ids - expected_ids
            raise ValueError(
                f"Lỗi toàn vẹn Vocab! Kích thước kỳ vọng: {self.vocab_size}. "
                f"Thiếu {len(missing)} IDs, dư {len(extra)} IDs."
            )

        # 2. Tải và kiểm tra tính toàn vẹn của merges.txt
        self.merges.clear()
        seen_merge_ids = set()

        with open(merges_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"Dòng {line_idx} trong {merges_path} sai định dạng: '{line}'. Phải có đúng 3 thành phần.")

                try:
                    p0, p1, new_id = int(parts[0]), int(parts[1]), int(parts[2])
                except ValueError:
                    raise ValueError(f"Dòng {line_idx} chứa ID không phải số nguyên: '{line}'")

                # Kiểm tra tham chiếu ID hợp lệ
                if p0 not in self.vocab or p1 not in self.vocab:
                    raise ValueError(f"Merge rule ({p0}, {p1}) tham chiếu đến ID không tồn tại trong vocab!")
                if new_id not in self.vocab:
                    raise ValueError(f"Merged ID {new_id} không tồn tại trong vocab!")
                if new_id in seen_merge_ids:
                    raise ValueError(f"Merged ID {new_id} bị trùng lặp trong merges.txt!")

                seen_merge_ids.add(new_id)
                self.merges[(p0, p1)] = new_id

        # 3. Kiểm tra tính nhất quán giữa số merge và vocab
        expected_merges = self.vocab_size - (BYTE_OFFSET + 256)
        if len(self.merges) != expected_merges:
            raise ValueError(
                f"Số merge rules trong merges.txt ({len(self.merges)}) "
                f"không khớp với số token merge dự kiến ({expected_merges})!"
            )

    def load_vocab(self, path_or_vocab: str, merges_file: Optional[str] = None) -> None:
        """Alias cho phương thức load()."""
        self.load(path_or_vocab, merges_file)

    def get_fingerprint(self) -> str:
        """
        Tính SHA256 Fingerprint toàn diện bảo đảm 100% tính toàn vẹn của artifact:
        - Chuỗi regex splitting pattern
        - Ánh xạ Special tokens
        - Toàn bộ từ điển vocab (được sắp xếp theo ID từ 0 đến vocab_size-1)
        - Toàn bộ các quy tắc merge (theo đúng thứ tự rank)
        """
        h = hashlib.sha256()

        # 1. Băm regex split pattern
        h.update(f"regex:{GPT4_SPLIT_PATTERN};".encode("utf-8"))

        # 2. Băm special tokens
        for k, v in sorted(self.special_tokens.items()):
            h.update(f"special:{k}->{v};".encode("utf-8"))

        # 3. Băm toàn bộ vocab theo ID tăng dần
        for token_id in range(self.vocab_size):
            if token_id in self.vocab:
                h.update(f"id:{token_id}=".encode("utf-8"))
                h.update(self.vocab[token_id])
                h.update(b";")

        # 4. Băm toàn bộ merges theo rank
        for pair, mid in sorted(self.merges.items(), key=lambda x: x[1]):
            h.update(f"merge:{pair[0]}+{pair[1]}->{mid};".encode("utf-8"))

        return h.hexdigest()

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> Union[str, List[int]]:
        """
        Định dạng danh sách hội thoại theo Chat Template chuẩn:
        - Mỗi lượt thoại:
            + role == 'user': `<|user|>{content}`
            + role == 'assistant': `<|assistant|>{content}<|eos|>`
        - add_generation_prompt=True: tự động thêm `<|assistant|>` ở cuối để mồi sinh.
        - tokenize=True: trả về list token IDs thay vì chuỗi string.
        """
        formatted = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            if role == "user":
                formatted += f"{USER_TOKEN}{content}"
            elif role == "assistant":
                formatted += f"{ASSISTANT_TOKEN}{content}{EOS_TOKEN}"

        if add_generation_prompt:
            formatted += ASSISTANT_TOKEN

        if tokenize:
            return self.encode(formatted)
        return formatted
