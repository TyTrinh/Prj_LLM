# PHÂN TÍCH BÀI TOÁN

1. **Xây dựng nền móng (Pre-training/Base Model):** Tạo ra cốt lõi LLM có khả năng Next-Token Prediction dựa trên kiến trúc Decoder-Only Transformer (Dùng Masked Self-Attention) và Weight Typing để tiết kiệm tham số cho mô hình nhỏ
2. **Dạy định dạng giao tiếp (SFT với LoRA tự viết):** Biến mô hình từ chỉ biết nối tiếp văn bản thành trợ lý biết nghe lệnh và trả lời theo cấu trúc hỏi - đáp. Tự code LoRA đóng băng trọng số gốc → giảm tham số cần cập nhập Gradient
3. **Uốn nắn hành vi theo 3H (DPO tự viết):** Tinh chỉnh xác suất sinh từ mà không cần huấn luyện 1 Reward Model riêng biệt → Ép mô hình ưu tiên câu trả lời đáp ứng 3H
4. **Thiết lập rào chắn bảo vệ (Input/Output Guardrails):** Bọc mô hình bằng lớp kiểm duyệt 2 đầu để chặn đứng các Prompt Injection, Jailbreak hoặc nội dung độc hại trước khi người dùng nhìn thấy

# CÁC GIAI ĐOẠN THỰC HIỆN

## Giai đoạn 1: Tokenizer & Kiến trúc Transfomer lõi

- **Thiết lập môi trường & Đường ống dữ liệu**
    - [x]  Tạo cấu trúc thư mục dự án (`src/model`, `src/tokenizer`, `src/alignment`, `src/guardrails`, `scripts/`, `notebooks/`)
    - [x]  Thiết lập bộ theo dõi huấn luyện (Weight & Biases/TensorBoard)
- **Tự xây dựng Tokenizer (Thuật toán Byte-Fallback BPE - Chiếm 10% điểm)**
    - [x]  Cài đặt thuật toán Byte-Pair Encoding từ số 0 (hoặc huấn luyện từ điển BPE thuần túy trên dữ liệu thô, không dùng Tokenizer có sẵn của mô hình khác)
    - [x]  Khai báo và đăng ký bộ cứng Special Token
    - [x]  Viết 2 hàm cốt lõi: encode(text) → List[int] và decode(ids) → str
    - [x]  Viết hàm `save_vocab(path)` và `load_vocab(path)` ra file `vocab.json` & `merges.txt`
    - [x]  Kiểm thử tính toàn vẹn (Round-trip test): Đảm bảo `decode(encode(s)) == s` với các chuỗi chứa ký tự đặc biệt, dấu câu và số
    - [ ]  Chạy token hóa toàn bộ tập Pre-train và lưu ra file nhị phân (`train.bin`, `val.bin` kiểu `uint16`) để nạp siêu tốc lên GPU.
- **Lập trình Kiến trúc Decoder-Only Transformer bằng PyTorch thuần (Chiếm 10% điểm)**
    - [x]  Viết lớp `RMSNorm(nn.Module)` (hoặc `LayerNorm` chuẩn hóa trước - Pre-Norm để ổn định gradient)
    - [x]  Viết cơ chế mã hóa vị trí **RoPE (Rotary Positional Embedding)** áp dụng trực tiếp lên Q & K
    - [x]  Viết lớp `CausalSelfAttention(nn.Module)` có ma trận mặt nạ tam giác dưới (Causal Mask) chặn nhìn trộm tương lai và hỗ trợ Multi-Head Attention
    - [x]  Viết lớp `FeedForward(nn.Module)` (sử dụng kiến trúc Linear-GELU-Linear hoặc SwiGLU)
    - [x]  Gom thành `TransformerBlock(nn.Module)` có kết nối thặng dư (Residual Connections)
    - [x]  Lắp ráp toàn bộ thành `GPTModel(nn.Module)`, áp dụng **Weight Tying** ($W_{\text{head}} = W_E^T$) để tiết kiệm tham số cho mô hình nhỏ
    - [x]  Khởi tạo trọng số chuẩn (Gaussian $\mathcal{N}(0, 0.02)$, chia $\sqrt{2 \times n_{\text{layers}}}$ tại residual projection)
    - [x]  **Chốt Checkpoint 1:** Chạy Unit Test kiểm tra tensor đầu vào `[B, T]` đi qua `GPTModel` trả ra Logits `[B, T, |V|]` không lỗi shape + Hoàn thành bản Đề cương Kỹ thuật (2 trang)

## Giai đoạn 2: KV Cache & Pre-training base model

- **Tự xây dựng KV Cache động (Chiếm 10% điểm)**
    - [x]  Thiết kế cấu trúc lưu trữ `KVCache` quản lý danh sách các tensor `(key_cache, value_cache)` cho từng tầng Transformer
    - [x]  Cập nhật hàm `forward` của `CausalSelfAttention` để hỗ trợ 2 chế độ:
        1. *Prefill (Xử lý prompt đầu vào):* Tính toán toàn bộ chuỗi ban đầu và nạp vào cache.
        2. *Decoding (Sinh từng token tiếp theo):* Chỉ tính toán trên 1 token mới nhất ($T=1$), nối (concatenate) $K, V$ mới vào cache cũ để truy xuất lại toàn bộ lịch sử
    - [x]  Viết hàm sinh văn bản tự hồi quy `generate()` hỗ trợ Greedy Search, Temperature Sampling và Top-k / Top-p Sampling
- **Tiền huấn luyện Base Model trên Kaggle**
    - [ ]  Cấu hình siêu tham số mô hình vừa vặn 16GB VRAM (gợi ý: ~40M–60M tham số, `d_model=512`, `n_layers=8`, `n_heads=8`, `block_size=512`)
    - [ ]  Viết vòng lặp huấn luyện sử dụng `torch.cuda.amp.autocast(dtype=torch.float16)`, Gradient Accumulation và Cosine Learning Rate Scheduler.
    - [ ]  Chạy huấn luyện trên Kaggle GPU cho đến khi Validation Loss hội tụ ổn định, lưu checkpoint `base_model.pt`
- **Đo đạc thực nghiệm KV Cache (Profiling)**
    - [ ]  Viết script benchmark đo đạc hiệu năng ở 2 trạng thái **Bật KV Cache** vs. **Tắt KV Cache** tại 4 mốc độ dài ngữ cảnh: $L \in \{128, 512, 1024, 2048\}$
    - [ ]  Ghi nhận đủ 4 chỉ số: Tốc độ sinh từ (`Tokens/giây`), `First Token Latency` (TTFT), `Next Token Latency` (TPOT), và `Peak VRAM` (`torch.cuda.max_memory_allocated()`)
    - [ ]  **Chốt Checkpoint 2:** Xuất biểu đồ so sánh Profiling + Quay video/chụp ảnh demo `base_model.pt` tự sinh văn bản mượt mà

## Giai đoạn 3: Tự code LoRA & Huấn luyện SFT

- **Lập trình module LoRA từ số 0**
    - [x]  Tạo lớp `LoRALinear(nn.Module)` bọc lấy lớp `nn.Linear` gốc, chứa 2 ma trận hạng thấp $A \in \mathbb{R}^{r \times d_{\text{in}}}$ (khởi tạo Kaiming/Gaussian) và $B \in \mathbb{R}^{d_{\text{out}} \times r}$ (khởi tạo toàn số `0`)
    - [x]  Cài đặt công thức tính đầu ra: $h = W_0 x + \frac{\alpha}{r} B A x$
    - [x]  Viết hàm tiện ích `inject_lora(model, r, alpha)` tự động duyệt qua tất cả các tầng Attention và thay thế ma trận chiếu $W_q, W_v$ bằng `LoRALinear`
    - [x]  Viết hàm `set_lora_enabled(model, enabled: bool)` để bật/tắt nhánh LoRA linh hoạt (phục vụ giai đoạn DPO sau này)
    - [x]  **[CẢI TIẾN 1 - 10% điểm]** Hàm `merge_and_unload()` gập trọng số cho zero-latency inference
    - [x]  Đóng băng toàn bộ trọng số gốc (`requires_grad = False`), chỉ mở khóa tham số của `LoRALinear` và kiểm tra tỷ lệ tham số trainable (thường < 3%)
- **Huấn luyện SFT (Supervised Fine-Tuning)**
    - [ ]  Tải tập `Alpaca-Cleaned`, định dạng lại theo Chat Template thống nhất: `<|user|>{prompt}<|assistant|>{response}<|eos|>`
    - [ ]  **Tạo Loss Mask (Quan trọng):** Gán nhãn `-100` cho toàn bộ phần `<|user|>{prompt}` để hàm Cross-Entropy Loss chỉ tính lỗi trên câu trả lời của Assistant (không ép mô hình học thuộc lòng câu hỏi)
    - [ ]  Chạy huấn luyện SFT trên Kaggle trong ~30–60 phút, lưu trọng số nhánh LoRA thành `sft_lora.pt`
    - [ ]  **Chốt Checkpoint 3A:** Xuất biểu đồ SFT Loss + Chạy thử nghiệm đối chứng câu trả lời trước SFT (nói lan man) và sau SFT (trả lời đúng câu hỏi và biết dừng ở `<|eos|>`)

## Giai đoạn 4: Code DPO Loss & Behavior Alignment

- **Xử lý dữ liệu cặp ưu tiên (Preference Dataset)**
    - [ ]  Tải tập `Anthropic/hh-rlhf` (lấy subset khoảng 10.000–20.000 mẫu chất lượng cao để khớp thời gian chạy GPU miễn phí)
    - [ ]  Tách dữ liệu thành 3 trường rõ ràng: `prompt`, `chosen` (câu trả lời tốt/an toàn), và `rejected` (câu trả lời xấu/độc hại)
    - [ ]  Token hóa song song chuỗi `prompt + chosen` và `prompt + rejected`, tạo mask đánh dấu chính xác vị trí bắt đầu của phần phản hồi để tính tổng log-xác suất
- **Lập trình DPO Loss & Vòng lặp huấn luyện tối ưu VRAM**
    - [ ]  Viết hàm `get_batch_logps(logits, labels, mask)` tính tổng log-probability $\log P(y \mid x)$ của từng chuỗi phản hồi.
    - [ ]  ****[CẢI TIẾN 2]**** Triển khai cờ `length_normalization=True` để tránh hiện tượng Length Bias của DPO
    - [ ]  Tự viết hàm `dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1)`
    - [ ]  **Áp dụng mẹo tiết kiệm 50% VRAM:** Trên cùng 1 mô hình duy nhất đã nạp `sft_lora.pt`:
        1. Gọi `set_lora_enabled(model, False)` dưới `torch.no_grad()` để lấy `ref_logps` ($\pi_{\text{ref}}$)
        2. Gọi `set_lora_enabled(model, True)` để tính `policy_logps` ($\pi_\theta$) và cập nhật đạo hàm
    - [ ]  Theo dõi và log metrics: DPO Loss, Implicit Reward Margin $(r_w - r_l)$, Chosen/Rejected Rewards, Win-rate nội bộ
    - [ ]  Chạy huấn luyện DPO (dùng Learning Rate rất nhỏ, khoảng `1e-5` đến `5e-5` để tránh sập mô hình), lưu checkpoint cuối cùng `dpo_lora.pt`
    - [ ]  **Chốt Checkpoint 3B:** Xuất 2 biểu đồ: DPO Loss hội tụ và **Implicit Reward Margin**

## Giai đoạn 5: Guardrails, đánh giá + viết báo cáo

- **[Cải tiến 3] Xây dựng hệ thống Input/Output Guardrails**
    - [ ]  **Input Guardrail:** Viết bộ lọc tiền xử lý chặn các mẫu tấn công Prompt Injection phổ biến (ví dụ: *"Ignore previous instructions"*, *"System override"*), kiểm tra từ khóa độc hại và giới hạn độ dài token đầu vào
    - [ ]  **Output Guardrail:** Viết bộ hậu kiểm rà soát văn bản mô hình vừa sinh ra (kiểm tra danh sách từ cấm, phát hiện lặp từ vô tận, hoặc dùng chính mô hình tự đánh giá an toàn trước khi hiển thị). Nếu vi phạm, thay thế bằng thông điệp từ chối tiêu chuẩn
    - [ ]  Template phản hồi từ chối chuẩn mực (Refusal Message)
- **Đánh giá Định lượng & Phân tích Lỗi (Chiếm 10% điểm)**
    - [ ]  Chạy sinh câu trả lời của 2 phiên bản (**SFT Model** vs. **DPO Model**) trên bộ kiểm thử `MT-Bench` (subset) và `ToxiGen`
    - [ ]  Viết script gọi API **LLM-as-a-Judge** (GPT-4o mini hoặc Claude 3.5 Haiku) chấm điểm so sánh cặp (Pairwise Comparison) và tính tỷ lệ thắng (**Win-rate %**) của DPO so với SFT
    - [ ]  Đo tỷ lệ chặn thành công các câu hỏi độc hại (Safety Refusal Rate) trên tập `ToxiGen` khi có vs. không có Guardrails
    - [ ]  **Error Analysis:** Chọn lọc và phân tích sâu **ít nhất 5 trường hợp thất bại thực tế** (ví dụ: mô hình từ chối nhầm câu hỏi vô hại - Over-refusal, lỗi chia từ BPE làm sai nghĩa, hoặc lọt lưới Guardrail)
- **Đóng gói Demo, GitHub & Viết Báo cáo Khoa học (Chiếm 30% điểm cuối)**
    - [ ]  Xây dựng giao diện web tương tác bằng **Gradio** có nút gạt so sánh trực tiếp 3 chế độ: `Base Model` | `SFT Model` | `DPO + Guardrails`
    - [ ]  Dọn dẹp mã nguồn trên GitHub, viết file `README.md` kèm file cấu hình/script để người chấm có thể chạy tái lập kết quả chỉ bằng 1 câu lệnh
    - [ ]  Viết hoàn chỉnh Báo cáo Khoa học **6–8 trang chuẩn format IEEE hoặc ICLR** đầy đủ các mục: *Abstract, Introduction, Methodology (nhấn mạnh Weight Tying, KV Cache, LoRA, DPO), Experimental Setup, Results, Error Analysis, Ethics & Limitations*

# PHÂN LUỒNG TRIỂN KHAI

```
THỜI GIAN    │ LUỒNG 1: TUẦN TỰ (Đường găng bắt buộc)                  │ LUỒNG 2: SONG SONG (Chuẩn bị trước & Độc lập)
─────────────┼─────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────
 NGÀY 1 – 2  │ [GĐ 1] Code Tokenizer BPE ──► Code Transformer Lõi      │ • Tải & lọc sẵn dữ liệu cho cả 3 pha (TinyStories,
 (2 ngày)    │        ──► Unit Test Tensor [B, T] -> [B, T, |V|]       │   Alpaca-Cleaned, Anthropic/hh-rlhf)
             │        ──► Chốt Checkpoint 1 (Đề cương 2 trang)         │ • Dựng sẵn khung file Báo cáo LaTeX (chuẩn IEEE/ICLR)
─────────────┼─────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────
 NGÀY 3 – 6  │ [GĐ 2] Code KV Cache ──► Pre-train Base Model (Kaggle)  │ • Code sẵn class LoRALinear (có công tắc bật/tắt)
 (4 ngày)    │        ──► Đo Profiling KV Cache (L=128..2048)          │ • Code sẵn hàm DPO Loss & test thử trên mạng nháp 2
             │        ──► Chốt Checkpoint 2 (Biểu đồ KV Cache + Demo)  │   layers ở CPU để đảm bảo không lỗi gradient/NaN
─────────────┼─────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────
 NGÀY 7 – 9  │ [GĐ 3] Gắn LoRA vào Base ──► Train SFT (Alpaca)         │ • Lập trình bộ lọc Input/Output Guardrails (Regex
 (3 ngày)    │        ──► Chốt Checkpoint 3A (sft_lora.pt + Loss)      │   chặn Prompt Injection, từ cấm, lặp từ vô tận)
─────────────┼─────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────
 NGÀY 10 – 11│ [GĐ 4] Nạp sft_lora.pt ──► Train DPO (hh-rlhf)          │ • Viết sẵn script gọi API LLM-as-a-Judge (GPT-4o mini
 (2 ngày)    │        ──► Chốt Checkpoint 3B (dpo_lora.pt + Margin)    │   / Claude 3.5 Haiku) chấm cặp tự động
─────────────┼─────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────
 NGÀY 12 – 14│ [GĐ 5] Chạy Benchmark (MT-Bench, ToxiGen)               │ • Dựng giao diện web Gradio (so sánh 3 chế độ)
 (3 ngày)    │        ──► Error Analysis (5 ca) ──► Nộp Checkpoint 4  │ • Viết README.md (one-click run) & làm Slide bảo vệ
```
