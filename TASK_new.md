# PHÂN TÍCH BÀI TOÁN

> **CẬP NHẬT KẾ HOẠCH:** Lần Pretraining trước đã chạy với Vocab 261 (5 Special Tokens + 256 Raw Bytes, 0 BPE merges), không đúng cấu hình Byte-Fallback BPE trong `Byte_Fallback_BPE_Tokenizer.md`. Checkpoint cũ vẫn được bảo toàn làm mốc đối chứng, nhưng **không dùng để tiếp tục hoặc làm Base Model chính thức** cho kiến trúc Vocab 8.192 (8K). Kế hoạch bên dưới bổ sung việc train BPE thực sự trên TinyStories và Pretraining lại từ Step 0.

1. **Xây dựng nền móng (Pre-training/Base Model):** Tạo ra cốt lõi LLM có khả năng Next-Token Prediction dựa trên kiến trúc Decoder-Only Transformer (Dùng Masked Self-Attention) và Weight Tying để tiết kiệm tham số cho mô hình nhỏ
2. **Dạy định dạng giao tiếp (SFT với LoRA tự viết):** Biến mô hình từ chỉ biết nối tiếp văn bản thành trợ lý biết nghe lệnh và trả lời theo cấu trúc hỏi - đáp. Tự code LoRA đóng băng trọng số gốc → giảm tham số cần cập nhật Gradient
3. **Uốn nắn hành vi theo 3H (DPO tự viết):** Tinh chỉnh xác suất sinh từ mà không cần huấn luyện 1 Reward Model riêng biệt → Ép mô hình ưu tiên câu trả lời đáp ứng 3H
4. **Thiết lập rào chắn bảo vệ (Input/Output Guardrails):** Bọc mô hình bằng lớp kiểm duyệt 2 đầu để chặn đứng các Prompt Injection, Jailbreak hoặc nội dung độc hại trước khi người dùng nhìn thấy

# NGUYÊN TẮC XUYÊN SUỐT CÁC GIAI ĐOẠN

- **Một tokenizer duy nhất:** Tokenizer Byte-Fallback BPE Vocab 8.192 (8K) được train trước Pretraining, sau đó giữ nguyên file và token-to-ID mapping cho Pretraining, SFT, DPO, Guardrails token counting và inference.
- **Nguồn dữ liệu tokenizer:** Dùng corpus TinyStories Train đã split theo story để train BPE; Validation không tham gia học merges. Alpaca-Cleaned và HH-RLHF chỉ dùng để kiểm tra khả năng phân đoạn trong kế hoạch hiện tại.
- **Checkpoint Vocab 261 là legacy:** Giữ để đối chứng và phục vụ tái lập thí nghiệm cũ; không tiếp tục train hoặc load vào model Vocab 8K.
- **Không so sánh trực tiếp loss/perplexity giữa Vocab 261 và Vocab 8K** như thể cùng một tokenizer. Đánh giá mỗi tokenizer/model bằng validation set đã encode tương ứng và thêm kiểm thử sinh văn bản.
- **Bảo toàn artifact:** Mọi checkpoint phải đi kèm model config, tokenizer fingerprint, dataset/split metadata và training config; không xóa hoặc ghi đè checkpoint gốc.

# CÁC GIAI ĐOẠN THỰC HIỆN

## Giai đoạn 1: Tokenizer & Kiến trúc Transformer lõi

- **Thiết lập môi trường & Đường ống dữ liệu**
    - [x]  Tạo cấu trúc thư mục dự án (`src/model`, `src/tokenizer`, `src/alignment`, `src/guardrails`, `scripts/`, `notebooks/`)
    - [x]  Thiết lập bộ theo dõi huấn luyện (Weight & Biases/TensorBoard)
- **Tự xây dựng Tokenizer (Thuật toán Byte-Fallback BPE - Chiếm 10% điểm)**
    - [x]  Cài đặt thuật toán Byte-Pair Encoding từ số 0; không sử dụng tokenizer có sẵn của mô hình khác.
    - [x]  Khai báo và đăng ký 5 Special Tokens với ID cố định `0..4`.
    - [x]  Đăng ký 256 Raw Byte tokens với ID cố định `5..260` để bảo đảm byte fallback.
    - [x]  Viết `encode(text) → List[int]` và `decode(ids) → str`.
    - [x]  Viết `save_vocab(path)` và `load(path, ...)` (đối chiếu đúng tên API thực tế) để lưu/nạp `vocab.json` và `merges.txt`.
    - [x]  Kiểm thử round-trip, special tokens, dấu câu, số, Unicode/UTF-8 và byte fallback.
    - [x]  Chọn 500.000 mẫu TinyStories và split theo story trước khi train tokenizer: 475.000 Train / 25.000 Validation; cố định split seed và lưu danh sách sample IDs.
    - [x]  **Train BPE thực tế chỉ trên 475.000 story Train**; tuyệt đối không dùng 25.000 story Validation để học merge rules.
    - [x]  Chốt `vocab_size=8192` theo `Byte_Fallback_BPE_Tokenizer.md`: Special Tokens `0..4`, Raw Bytes `5..260`, Merge Tokens `261..8191` (mục tiêu 7.931 merge operations).
    - [x]  Kiểm tra số merge thực tế, ID liên tục/không trùng, thứ tự merge, encode/decode và save-load round-trip sau khi train. Nếu thuật toán dừng sớm hoặc không tạo đủ merges, dừng và báo cáo thay vì tự nhận đã tạo Vocab 8K.
    - [x]  Lưu tokenizer artifact bất biến (`vocab.json`, `merges.txt`, special-token map, config và fingerprint/hash); không train lại hoặc sửa tokenizer sau khi Pretraining bắt đầu.
    - [ ]  Đo tokenization statistics trên TinyStories và kiểm tra thêm fertility/độ dài chuỗi trên mẫu Alpaca-Cleaned, HH-RLHF để đánh giá khả năng dùng chung cho SFT/DPO. Các tập SFT/DPO chỉ dùng để kiểm tra, không đưa vào corpus train BPE trong kế hoạch hiện tại.
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
- **Chuẩn bị dữ liệu và Tokenize cho Pretraining mới**
    - [ ]  Nạp đúng 500.000 story đã split và khóa ở Giai đoạn 1 (475.000 Train / 25.000 Validation).
    - [ ]  Load tokenizer artifact Vocab 8.192 (8K) đã train; kiểm tra fingerprint/hash và special-token IDs trước khi encode.
    - [ ]  Dùng đúng cùng tokenizer artifact để encode cả Train và Validation; không train lại tokenizer trong notebook Pretraining.
    - [ ]  Chèn `<|eos|>` đúng một lần tại ranh giới mỗi story; kiểm tra không mất hoặc lặp EOS. Ghi rõ chiến lược packing và xử lý ranh giới document; không xem EOS đơn thuần là cơ chế ngăn attention nhìn sang document trước.
    - [ ]  Lưu `train.bin`, `val.bin` và metadata: dataset version, số stories, số tokens sau encode, tokenizer hash, split seed và special-token IDs.
    - [ ]  Không tái sử dụng `train.bin`/`val.bin` cũ đã tokenize bằng Vocab 261.
- **Tiền huấn luyện Base Model mới trên Kaggle (Scratch – Vocab 8K)**
    - [ ]  Khởi tạo GPTModel mới với `vocab_size=8192`; giữ kiến trúc Backbone đã chốt (d_model=512, n_layers=8, n_heads=8, RMSNorm, RoPE, SwiGLU, Weight Tying). Đối chiếu `block_size` trong config/model với `TRAIN_BLOCK_SIZE=512`; không để hai khái niệm bị nhầm lẫn.
    - [ ]  Không load trọng số checkpoint Vocab 261 vào model Vocab 8K. Giữ checkpoint cũ ở vùng lưu trữ legacy để đối chứng; không ghi đè.
    - [ ]  Benchmark 20–50 optimizer steps trước khi train chính thức; thử micro-batch nhỏ (bắt đầu 4, cân nhắc 8 nếu VRAM cho phép), điều chỉnh gradient accumulation theo peak VRAM và effective batch thực tế.
    - [ ]  Ghi nhận throughput, peak VRAM, loss, gradient norm và tình trạng AMP/NaN/Inf; chỉ chốt batch config sau benchmark trên GPU thực tế.
    - [ ]  Chạy Pretraining mới từ Step 0; mục tiêu 4.600 optimizer steps (1.5 Epochs), đánh giá tại các mốc 500/1.000/1.500/2.000/2.500/4.600 steps.
    - [ ]  Thiết lập Learning Rate schedule mới từ Step 0 (warmup + cosine); không dùng LR/scheduler state của checkpoint Vocab 261. Chọn peak LR sau pilot, không mặc định giữ nguyên lịch cũ khi chưa kiểm tra ổn định.
    - [ ]  Tính số token thực tế sau BPE và số token positions đã xử lý (`steps × effective_batch_sequences × sequence_length`); không suy số token bằng giả định tokens/story cố định và không gọi random memmap sampling là một epoch đầy đủ.
    - [ ]  Lưu checkpoint định kỳ và tách riêng `initial`, `last`, `best`; lưu đầy đủ model, optimizer, AMP scaler, step, RNG, config, tokenizer hash, dataset metadata và trạng thái sampling cần thiết để resume.
    - [ ]  Lưu output vào thư mục versioned riêng, ví dụ `checkpoints/pretrain_bpe8k_500k/`; không ghi đè `base_model.pt`/`last_checkpoint.pt` của lần chạy Vocab 261.
    - [ ]  Chốt Base Model mới theo best validation checkpoint và kiểm tra generation trước khi chuyển sang SFT.
- **Đo đạc thực nghiệm KV Cache (Profiling)**
    - [x]  Viết script benchmark đo đạc hiệu năng ở 2 trạng thái **Bật KV Cache** vs. **Tắt KV Cache** tại 4 mốc độ dài ngữ cảnh: $L \in \{128, 512, 1024, 2048\}$
    - [x]  Ghi nhận đủ 4 chỉ số: Tốc độ sinh từ (`Tokens/giây`), `First Token Latency` (TTFT), `Next Token Latency` (TPOT), và `Peak VRAM` (`torch.cuda.max_memory_allocated()`)
    - [ ]  **Chốt Checkpoint 2 cuối:** Chạy lại profiling và demo bằng Base Model BPE 8K; giữ kết quả Vocab 261 làm baseline legacy, không dùng thay cho kết quả model cuối.

## Giai đoạn 3: Tự code LoRA & Huấn luyện SFT

- **Lập trình module LoRA từ số 0**
    - [x]  Tạo lớp `LoRALinear(nn.Module)` bọc lấy lớp `nn.Linear` gốc, chứa 2 ma trận hạng thấp $A \in \mathbb{R}^{r \times d_{\text{in}}}$ (khởi tạo Kaiming/Gaussian) và $B \in \mathbb{R}^{d_{\text{out}} \times r}$ (khởi tạo toàn số `0`)
    - [x]  Cài đặt công thức tính đầu ra: $h = W_0 x + \frac{\alpha}{r} B A x$
    - [x]  Viết hàm tiện ích `inject_lora(model, r, alpha)` tự động duyệt qua tất cả các tầng Attention và thay thế ma trận chiếu $W_q, W_v$ bằng `LoRALinear`
    - [x]  Viết hàm `set_lora_enabled(model, enabled: bool)` để bật/tắt nhánh LoRA linh hoạt (phục vụ giai đoạn DPO sau này)
    - [x]  **[CẢI TIẾN 1 - 10% điểm]** Hàm `merge_and_unload()` gập trọng số cho zero-latency inference
    - [x]  Đóng băng toàn bộ trọng số gốc (`requires_grad = False`), chỉ mở khóa tham số của `LoRALinear` và kiểm tra tỷ lệ tham số trainable (thường < 3%)
- **Huấn luyện SFT (Supervised Fine-Tuning)**
    - [x]  Tải tập `Alpaca-Cleaned`, định dạng lại theo Chat Template thống nhất: `<|user|>{prompt}<|assistant|>{response}<|eos|>`
    - [ ]  Load đúng tokenizer artifact Vocab 8K đã dùng trong Pretraining; kiểm tra tokenizer hash, special-token IDs và `model.config.vocab_size` khớp nhau. Xóa mọi assert/hardcode cố định `vocab_size=261`.
    - [ ]  Nạp Base Model mới từ `pretrain_bpe8k_500k/best_checkpoint`; không nạp checkpoint legacy Vocab 261.
    - [ ]  Kiểm tra encode/decode trên prompt/response tiếng Anh và tiếng Việt, EOS, padding và giới hạn sequence length trước khi SFT.
    - [x]  **Tạo Loss Mask (Quan trọng):** Gán nhãn `-100` cho toàn bộ phần `<|user|>{prompt}` để hàm Cross-Entropy Loss chỉ tính lỗi trên câu trả lời của Assistant (không ép mô hình học thuộc lòng câu hỏi)
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

# TIÊU CHÍ NGHIỆM THU CHO LẦN PRETRAINING BPE 8K MỚI

- [ ]  `len(tokenizer.vocab) == 8192`; Special Tokens có ID `0..4`, Raw Bytes có ID `5..260`, merge tokens chiếm các ID còn lại đến `8191` theo đúng đặc tả.
- [ ]  Số BPE merges thực tế đạt mục tiêu thiết kế; nếu thuật toán dừng sớm hoặc không tạo đủ merges, phải báo rõ nguyên nhân, không gán nhãn Vocab 8K khi chưa đạt.
- [ ]  `save()` rồi `load()` tokenizer cho kết quả encode giống hệt; round-trip thành công với văn bản ASCII, Unicode, tiếng Việt, dấu câu và chuỗi có special tokens.
- [ ]  Train/Validation được split theo story trước tokenize; không có story trùng hoặc bị chia đôi giữa hai tập; lưu seed và danh sách định danh mẫu để tái lập.
- [ ]  `train.bin` và `val.bin` được tạo bằng cùng tokenizer hash; mọi ID nằm trong `[0, 8191]`; dtype `uint16` không overflow.
- [ ]  `model.config.vocab_size`, input embedding và tied LM Head cùng có chiều vocabulary 8.192; forward trả logits `[B, T, 8192]`.
- [ ]  Pilot 20–50 steps không OOM, không NaN/Inf; ghi nhận peak VRAM, tokens/second và optimizer steps/second trên GPU thực tế.
- [ ]  LR bắt đầu đúng warmup/schedule mới từ Step 0, không load LR/scheduler state của checkpoint Vocab 261.
- [ ]  Resume thử nghiệm khôi phục model, optimizer, scaler, step, RNG và metadata; checkpoint legacy không bị ghi đè.
- [ ]  Validation được tính trên tập cố định; chọn best checkpoint; không dùng ngưỡng loss `<= 0.50` của Vocab 261 làm điều kiện bắt buộc cho Vocab 8K vì tokenizer và phân phối token đã thay đổi.
- [ ]  Base Model sinh được văn bản có nghĩa, không lỗi UTF-8, không lặp vô hạn và dừng EOS hợp lý trên bộ prompt đánh giá.
- [ ]  SFT notebook nạp đúng tokenizer/model Vocab 8K bằng kiểm tra config/fingerprint, không còn hardcode 261.

# PHÂN LUỒNG TRIỂN KHAI

```
THỜI GIAN    │ LUỒNG 1: TUẦN TỰ (Đường găng bắt buộc)                  │ LUỒNG 2: SONG SONG (Chuẩn bị trước & Độc lập)
─────────────┼─────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────
 NGÀY 1 – 2  │ [GĐ 1] Code/kiểm thử BPE ──► Split TinyStories      │ • Chuẩn bị Alpaca-Cleaned, HH-RLHF và bộ đánh giá
 (2 ngày)    │        ──► Train BPE 32K trên Train split               │   độc lập; không trộn validation vào corpus BPE
             │        ──► Save/Load + Round-trip + Unit Test model     │ • Dựng sẵn khung báo cáo và lưu artifacts/checkpoints
─────────────┼─────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────
 NGÀY 3 – 6  │ [GĐ 2] Tokenize 500K ──► Pretrain BPE 32K từ Step 0  │ • Code sẵn class LoRALinear (có công tắc bật/tắt)
 (4 ngày)    │        ──► Pilot VRAM/throughput rồi train 4.600 steps │ • Test DPO Loss trên mô hình nhỏ ở CPU
             │        ──► KV Cache profiling + demo Base Model mới     │ • Chuẩn bị bộ prompt đánh giá Base Model
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


## GHI CHÚ VỀ THỜI GIAN VÀ TÀI NGUYÊN

- Không chốt thời gian Pretraining bằng throughput của model Vocab 261. Vocab 8K làm logits/LM Head lớn hơn, vì vậy phải benchmark model mới trước khi ước tính thời gian cuối cùng.
- Sau khi tokenize 500.000 stories, ghi nhận số tokens thực tế (total/train/validation), token length mean/median/p95 và kích thước `train.bin`/`val.bin`. Không sử dụng giả định cố định tokens/story.
- Trên Kaggle, chia quá trình thành các phiên có checkpoint và kiểm tra việc lưu artifacts ra vùng persistent storage. Không phụ thuộc một phiên chạy liên tục.
- Dành ngân sách GPU còn lại cho SFT LoRA, DPO, evaluation và các lần sửa lỗi; không tăng Pretraining steps chỉ vì còn dữ liệu.
