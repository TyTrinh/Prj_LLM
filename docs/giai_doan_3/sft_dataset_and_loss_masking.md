# TÀI LIỆU KỸ THUẬT: PIPELINE DỮ LIỆU SFT & TARGET LOSS MASKING

> **Giai đoạn:** Giai đoạn 3 — Lập trình LoRA từ số 0 & Huấn luyện SFT  
> **Module liên quan:** `src.alignment`, `src.tokenizer`  
> **Source Files:**
> - [`src/alignment/sft_dataset.py`](../../src/alignment/sft_dataset.py)
> - [`src/alignment/__init__.py`](../../src/alignment/__init__.py)
> - [`src/tokenizer/bpe.py`](../../src/tokenizer/bpe.py)
> - [`src/tokenizer/special_tokens.py`](../../src/tokenizer/special_tokens.py)  
> **Test File:** [`tests/test_sft_dataset.py`](../../tests/test_sft_dataset.py)  
> **Tài liệu lý thuyết tham chiếu:**
> - Taori et al. (2023) *"Stanford Alpaca: An Instruction-following LLaMA model"*
> - Zhou et al. (2023) *"LIMA: Less Is More for Alignment"* (Meta AI & CMU)

---

## 1. Tổng quan & Bản chất Kỹ thuật

### 1.1. Từ Base Model sang Instruction-Following Model
Base Model sau quá trình Pre-training là một bộ ước lượng xác suất phân phối văn bản $P(x_t \mid x_{<t})$. Nó có xu hướng sinh tiếp văn bản tương đồng với dữ liệu web thay vì trả lời câu hỏi của người dùng (ví dụ: khi gặp prompt hỏi "Thủ đô của Việt Nam là gì?", Base Model có thể sinh tiếp "Thủ đô của Pháp là gì? Thủ đô của Nhật Bản là gì?...").

Giai đoạn **Supervised Fine-Tuning (SFT)** chuyển hóa mô hình thành một trợ lý đàm thoại có cấu trúc bằng cách:
1. Đưa dữ liệu về dạng hội thoại có phân định vai trò rõ ràng qua các Special Tokens.
2. Áp dụng kỹ thuật **Target Loss Masking** để chỉ cập nhật trọng số trên phần phản hồi của Assistant, không ép mô hình học thuộc lòng câu hỏi của User.

---

## 2. Chuẩn Hóa Định Dạng Hội Thoại (Chat Template)

### 2.1. Cấu trúc Template Dự án
Dự án sử dụng các Special Tokens có ID cố định trong `src/tokenizer/special_tokens.py` (`<|pad|>=0`, `<|bos|>=1`, `<|eos|>=2`, `<|user|>=3`, `<|assistant|>=4`):

```text
<|user|>{prompt}<|assistant|>{response}<|eos|>
```

- Thẻ `<|user|>`: Đóng vai trò mở đầu lượt hỏi của người dùng.
- Thẻ `<|assistant|>`: Đóng vai trò mồi sinh (generation prompt) và báo hiệu kết thúc prompt.
- Thẻ `<|eos|>`: Đóng vai trò **Stop Token**. Mô hình bắt buộc phải học cách sinh token này để kết thúc lượt phát ngôn.

### 2.2. Chuẩn hóa dữ liệu Alpaca-Cleaned
Dữ liệu Alpaca gồm 3 trường: `instruction`, `input`, và `output`.
- Nếu có ngữ cảnh `input`: `prompt = f"{instruction}\n\n{input}"`
- Nếu không có `input`: `prompt = instruction`
- Chuỗi hoàn chỉnh:
  ```python
  text = f"<|user|>{prompt}<|assistant|>{output}<|eos|>"
  ```

### 2.3. Đồng bộ hóa với Tokenizer (`apply_chat_template`)
Tương tự cơ chế Jinja2 Chat Template trong Hugging Face Tokenizers, lớp `BytePairTokenizer` tại [`src/tokenizer/bpe.py`](../../src/tokenizer/bpe.py) được bổ sung phương thức:

```python
def apply_chat_template(
    self,
    messages: List[Dict[str, str]],
    tokenize: bool = False,
    add_generation_prompt: bool = False,
) -> Union[str, List[int]]:
```
- Khi huấn luyện SFT: Ghép đầy đủ chuỗi hội thoại.
- Khi triển khai suy luận (Inference): Đặt cờ `add_generation_prompt=True` để tự động chèn thêm `<|assistant|>` ở cuối chuỗi làm mồi cho mô hình bắt đầu sinh phản hồi.

---

## 3. Cơ Chế Target Loss Masking

### 3.1. Cơ sở Toán học
Chuỗi huấn luyện gồm 2 phân vùng:
$$\mathbf{X} = [\underbrace{x_1, \dots, x_P}_{\text{Prompt (User + Assistant Header)}}, \underbrace{x_{P+1}, \dots, x_T}_{\text{Completion (Response + EOS)}}]$$

Nếu tính Cross-Entropy trên toàn chuỗi, mô hình sẽ lãng phí biên độ gradient vào việc ghi nhớ các biến thể ngẫu nhiên của câu hỏi (User Prompt), dẫn đến hiện tượng suy thoái hành vi (học vẹt, tự hỏi tự trả lời).

Hàm mất mát SFT chuẩn chỉ tính trên tập token phản hồi $T_{\text{assistant}}$:
$$\mathcal{L}_{\text{SFT}} = - \frac{1}{|T_{\text{assistant}}|} \sum_{t \in T_{\text{assistant}}} \log P(x_t \mid x_{<t})$$

### 3.2. Cơ chế Triệt tiêu Gradient của PyTorch (`ignore_index = -100`)
Trong `torch.nn.CrossEntropyLoss(ignore_index=-100)`:
1. **Forward Pass:** Vị trí có nhãn `-100` bị loại trừ khỏi phép cộng loss và mẫu số chuẩn hóa: $\ell_t = 0$.
2. **Backward Pass:** Đạo hàm riêng tại vị trí mang nhãn `-100` bằng `0.0` tuyệt đối:
   $$\frac{\partial \mathcal{L}_{\text{SFT}}}{\partial z_{t, i}} = 0.0, \quad \forall i \in \{1, \dots, V\}$$

### 3.3. Bảng Phân Bổ Tensor Nhãn Huấn Luyện

| Vị trí | Token Chuỗi | Phân Vùng | `input_ids` | `labels` | Trạng thái Gradient |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `<|user|>` | Special Token mở đầu | `3` | **-100** | $0.0$ (Triệt tiêu) |
| $1 \dots P-2$ | `{prompt}` | Nội dung câu hỏi | Token IDs | **-100** | $0.0$ (Triệt tiêu) |
| $P-1$ | `<|assistant|>` | Header chuyển giao | `4` | **-100** | $0.0$ (Triệt tiêu) |
| $P \dots T-2$ | `{response}` | **Phản hồi của Assistant** | Token IDs | **Token IDs** | $P - 1$ (Kích hoạt học) |
| $T-1$ | `<|eos|>` | **Stop Token bắt buộc** | `2` | **2** | $P - 1$ (Kích hoạt học) |
| $T \dots \text{max\_seq\_len}$ | `<|pad|>` | Đệm cố định độ dài | `0` | **-100** | $0.0$ (Triệt tiêu) |

> [!CAUTION]
> Token `<|eos|>` của Assistant **BẮT BUỘC PHẢI TÍNH LOSS** (giữ nguyên token ID thay vì gán -100). Nếu mask luôn token này, mô hình sẽ không bao giờ học được cách kết thúc câu trả lời và sẽ sinh chữ lan man không điểm dừng khi đưa vào thực tế!

---

## 4. Chi Tiết Cài Đặt Trong Source Code

### 4.1. Lớp `AlpacaSFTDataset` (`src/alignment/sft_dataset.py`)
- Kế thừa `torch.utils.data.Dataset`.
- Nhận danh sách raw JSON hoặc file path, chuẩn hóa trường dữ liệu thành `{"prompt": ..., "response": ...}`.
- Hỗ trợ tham số `max_samples` để lấy tập con nhỏ (ví dụ 1.000–5.000 mẫu theo triết lý LIMA nhằm tăng tốc và tránh Catastrophic Forgetting).

### 4.2. Lớp `SFTDataCollator` (`src/alignment/sft_dataset.py`)
- Nhận một mini-batch mẫu dictionary `{"prompt", "response"}`.
- Mã hóa `prompt_str = f"<|user|>{prompt}<|assistant|>"` để lấy chính xác độ dài `prompt_len`.
- Mã hóa `response_str = f"{response}<|eos|>"`.
- **Cơ chế Safe Truncation (Bảo tồn Token Dừng & Tránh NaN Loss):**
  - Đảm bảo dành tối thiểu `min_response_tokens` cho câu trả lời. Nếu prompt quá dài (`prompt_len > max_seq_len - min_response_tokens`), phần đầu của prompt sẽ được cắt bớt an toàn, bảo vệ prefix `<|user|>` và suffix `<|assistant|>`.
  - Khi tổng chuỗi vượt quá `max_seq_len`, phần thân `response` được cắt ngắn nhưng **LUÔN GHIM token cuối cùng là `<|eos|>`**. Nhờ đó, mô hình luôn học được điểm kết thúc câu trả lời và không bao giờ gặp tình trạng toàn bộ labels mang giá trị `-100` (loại bỏ triệt để lỗi chia cho 0 hoặc NaN loss).
- Gán nhãn:
  ```python
  labels = [self.ignore_index] * prompt_len + response_ids
  ```
- Tự động chèn padding `<|pad|>` cho `input_ids` và `-100` cho `labels`.

### 4.3. Cơ chế Shift Logits & Labels trong Causal LM (`compute_sft_loss`)
Trong bài toán Causal Language Modeling, logits tại bước thời gian $t$ mang nhiệm vụ dự đoán token tiếp theo tại $t+1$:
```python
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
loss = loss_fn(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
```
- Nếu không thực hiện shift mà truyền trực tiếp vào CrossEntropyLoss, mô hình sẽ bị lệch 1 nhịp thời gian (ép học đoán từ hiện tại thay vì từ tiếp theo).
- Hàm `compute_sft_loss` được tích hợp vào `src/alignment/sft_dataset.py` và script huấn luyện `scripts/train_sft.py`.

---

## 5. Kết Quả Kiểm Thử Toàn Diện (Verification)

Module đã được kiểm thử nghiêm ngặt tại [`tests/test_sft_dataset.py`](../../tests/test_sft_dataset.py) với 8/8 test cases đạt chuẩn:

| Test Case | Mục tiêu kiểm chứng | Trạng thái |
| :--- | :--- | :---: |
| `test_format_alpaca_prompt` | Kiểm tra chuẩn hóa prompt có `input` và không có `input` | **PASSED** |
| `test_format_sft_text` | Kiểm tra định dạng Chat Template chuỗi | **PASSED** |
| `test_alpaca_sft_dataset` | Kiểm tra load và parsing của `AlpacaSFTDataset` | **PASSED** |
| `test_target_loss_masking_values` | Kiểm tra giá trị tensor: 100% prompt là -100, response và EOS giữ nguyên ID, padding là -100 | **PASSED** |
| `test_compute_sft_loss_gradient_elimination` | Kiểm tra Autograd với compute_sft_loss (có shift): gradient tại prompt và pad = 0.0, gradient tại completion != 0.0, logit cuối = 0.0 | **PASSED** |
| `test_tokenizer_apply_chat_template` | Kiểm tra hàm `apply_chat_template` đồng bộ format và generation prompt | **PASSED** |
| `test_safe_truncation_preserves_eos_and_targets` | Kiểm tra cắt cụt an toàn: luôn bảo tồn token `<|eos|>` ở cuối và nhãn hợp lệ >= min_response_tokens, không bị NaN | **PASSED** |
| `test_compute_sft_loss_shift_logits` | Kiểm tra cơ chế Shift Logits và Labels $t \to t+1$, gradient khớp chuẩn Autograd | **PASSED** |

```bash
============================= test session starts =============================
platform win32 -- Python 3.12.13, pytest-9.2.0, pluggy-1.7.0
rootdir: E:\Python\DUT_AI\A-Module LLM\Prj_LLM
collected 8 items

tests/test_sft_dataset.py::test_format_alpaca_prompt PASSED              [ 12%]
tests/test_sft_dataset.py::test_format_sft_text PASSED                   [ 25%]
tests/test_sft_dataset.py::test_alpaca_sft_dataset PASSED                [ 37%]
tests/test_sft_dataset.py::test_target_loss_masking_values PASSED        [ 50%]
tests/test_sft_dataset.py::test_compute_sft_loss_gradient_elimination PASSED [ 62%]
tests/test_sft_dataset.py::test_tokenizer_apply_chat_template PASSED     [ 75%]
tests/test_sft_dataset.py::test_safe_truncation_preserves_eos_and_targets PASSED [ 87%]
tests/test_sft_dataset.py::test_compute_sft_loss_shift_logits PASSED     [100%]

============================== 8 passed in 3.01s ==============================
```

---

## 6. Nhật Ký Sự Cố & Phân Tích Khắc Phục (Post-Mortem & Fix Log)

Tuân thủ nghiêm ngặt nguyên tắc **Constitution** ("Verify before reporting. Document root causes and fixes transparently"), bảng dưới đây tổng hợp chi tiết các lỗi phát hiện qua audit, nguyên nhân gốc rễ, hậu quả và giải pháp kỹ thuật đã thực hiện:

### 6.1. Chi tiết các lỗi phát hiện và giải pháp kỹ thuật

#### Sự cố 1: Lỗi Crash `train_sft.py` do sai kiểu dữ liệu (CRIT-01)
- **File vi phạm:** [`scripts/train_sft.py`](../../scripts/train_sft.py#L125)
- **Vi phạm gì:** Truy cập key dictionary `stats['trainable_params']` trên đối tượng `Tuple[int, int, float]`.
- **Lỗi trước vi phạm & Hậu quả:**
  - `LoRAManager.get_trainable_parameters()` trả về tuple 3 phần tử `(trainable_params, total_params, trainable_ratio)`.
  - Script `train_sft.py` truy cập `stats['trainable_params']` dẫn đến ngoại lệ `TypeError: tuple indices must be integers or slices`.
  - Script SFT crash ngay tại thời điểm khởi động, không thể thực hiện bất kỳ bước huấn luyện nào.
- **Cách khắc phục & Lý do:**
  - Chuyển sang unpacking tuple trực tiếp:
    ```python
    trainable_params, total_params, trainable_ratio = LoRAManager.get_trainable_parameters(model)
    ```
  - Lý do: Đồng bộ 100% với chữ ký hàm và bảo tồn tương thích với các test case hiện có trong `tests/test_lora_freeze.py`.

#### Sự cố 2: Lỗ hổng kiểm thử Gradient Elimination không qua Shift Logits (CRIT-02)
- **File vi phạm:** [`tests/test_sft_dataset.py`](../../tests/test_sft_dataset.py#L125)
- **Vi phạm gì:** Unit test kiểm tra gradient elimination bằng `nn.CrossEntropyLoss` không shift, trong khi mã nguồn production lại dùng `compute_sft_loss` (có shift).
- **Lỗi trước vi phạm & Hậu quả:**
  - Hai luồng thực thi khác nhau: Test kiểm tra một đằng, code chạy thực tế một nẻo.
  - Nếu cơ chế shift của `compute_sft_loss` có lỗi logic gán nhãn, test suite vẫn báo PASS, tạo ra ảo giác an toàn giả tạo (False Positive).
- **Cách khắc phục & Lý do:**
  - Viết lại test `test_compute_sft_loss_gradient_elimination`: gọi trực tiếp `compute_sft_loss(logits, labels)` $\to$ `loss.backward()` $\to$ xác minh gradient của logit tại bước $t$ tương ứng với label tại bước $t+1$.
  - Kiểm tra thêm điều kiện biên: logit tại vị trí cuối cùng sequence ($t = \text{seq\_len}-1$) bị shift loại bỏ nên gradient bắt buộc bằng 0.

#### Sự cố 3: Truncation cắt mất vùng học và làm mất Stop Token `<|eos|>`
- **File vi phạm:** [`src/alignment/sft_dataset.py`](../../src/alignment/sft_dataset.py#L182-L205)
- **Vi phạm gì:** Cắt cụt ngây thơ `input_ids[:max_seq_len]` mà không bảo vệ ranh giới response.
- **Lỗi trước vi phạm & Hậu quả:**
  - Khi prompt dài $\ge$ `max_seq_len`, toàn bộ response và `<|eos|>` bị xóa sổ $\to$ `labels` chỉ toàn `-100`.
  - Hàm `CrossEntropyLoss(ignore_index=-100)` không tìm thấy nhãn hợp lệ nào $\to$ sinh lỗi chia cho 0 hoặc trả về `NaN loss`, làm phân rã toàn bộ trọng số mạng thần kinh.
  - Khi response bị cắt ngang, token `<|eos|>` ở cuối bị văng ra ngoài khiến mô hình mất khả năng học điều kiện dừng.
- **Cách khắc phục & Lý do:**
  - Thêm tham số `min_response_tokens = 4`. Nếu prompt quá dài, cắt bớt phần đầu prompt để bảo toàn không gian cho response.
  - Khi tổng độ dài vượt quá `max_seq_len`, cắt phần giữa response nhưng **LUÔN GHIM token cuối là `<|eos|>`**.

#### Sự cố 4: `train_sft.py` thiếu Validation Loop và Random Seed (HIGH-02, LOW-02)
- **File vi phạm:** [`scripts/train_sft.py`](../../scripts/train_sft.py#L45-L180)
- **Vi phạm gì:** Không có Train/Val split, không tính validation loss, không cố định random seed.
- **Lỗi trước vi phạm & Hậu quả:**
  - Không thể phát hiện hiện tượng Overfitting khi train 2 epochs. Không có thước đo khách quan để lưu checkpoint LoRA tốt nhất.
  - Kết quả huấn luyện không thể tái lập (non-reproducible).
- **Cách khắc phục & Lý do:**
  - Thêm `--val_ratio` (5%), dùng `random_split` chia tập Train/Val.
  - Viết hàm `evaluate(model, val_loader)` tính `val_loss` dưới `torch.no_grad()`.
  - Lưu checkpoint LoRA kèm metadata khi đạt `best_val_loss`.
  - Tích hợp `set_seed(args.seed)` cố định toàn bộ các nguồn ngẫu nhiên.

#### Sự cố 5: `InferenceEngine` là Mock Stub, không kết nối Model/LoRA (HIGH-03)
- **File vi phạm:** [`src/inference/engine.py`](../../src/inference/engine.py#L1-L120)
- **Vi phạm gì:** Hàm `generate` trả về chuỗi hardcode và số liệu ảo `tokens_per_sec = 85.4`.
- **Lỗi trước vi phạm & Hậu quả:**
  - Training-Inference Mismatch hoàn toàn. Không thể dùng engine để kiểm thử câu trả lời đối chứng trước và sau khi fine-tune SFT.
- **Cách khắc phục & Lý do:**
  - Viết lại toàn bộ `InferenceEngine.generate()`: định dạng prompt qua `apply_chat_template(add_generation_prompt=True)`, gọi `model.generate()`, hỗ trợ KV Cache, Sampler và đo đạc latency/throughput thực tế.

#### Sự cố 6: Lọc dữ liệu Alpaca không đủ chặt (MED-03)
- **File vi phạm:** [`src/alignment/sft_dataset.py`](../../src/alignment/sft_dataset.py#L110-L115)
- **Vi phạm gì:** Chỉ bỏ qua khi `not instruction and not output`.
- **Lỗi trước vi phạm & Hậu quả:**
  - Mẫu có instruction hợp lệ nhưng output rỗng `""` vẫn lọt qua $\to$ tạo ra mẫu hội thoại không có câu trả lời, gây nhiễu phân phối loss.
- **Cách khắc phục & Lý do:**
  - Đổi thành `if not instruction or not output: continue`, đảm bảo cả câu hỏi và câu trả lời đều có nội dung thực sự.

