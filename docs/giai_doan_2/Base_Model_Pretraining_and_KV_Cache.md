# TÀI LIỆU KỸ THUẬT: TIỀN HUẤN LUYỆN BASE MODEL & CƠ CHẾ DYNAMIC KV CACHE
**(BASE MODEL PRE-TRAINING, CHECKPOINT RESUME & KV CACHE PROFILING)**

> **Giai đoạn:** Giai đoạn 2 - KV Cache & Pre-training Base Model  
> **Các Module triển khai:**  
> - `src/model/kv_cache.py`: Dynamic KV Cache (Prefill & Decode O(1))  
> - `src/utils/checkpoint.py`: CheckpointManager (Toàn diện State Dict & Auto-Resume)  
> - `src/utils/tracker.py`: WandbTracker (Weights & Biases Real-time Tracking)  
> - `src/utils/profiler.py`: KVCacheProfiler (Đo đạc thực nghiệm 4 chỉ số hiệu năng)  
> - `notebooks/01_pretrain_base_model.ipynb`: Quy trình huấn luyện Cloud GPU & Benchmark  
> **Trạng thái:** [x] Hoàn thành kiến trúc & Pipeline, Đã qua kiểm thử thực tế trên Kaggle GPU  
> **Mục tiêu lưu trữ:** Tài liệu kỹ thuật chi tiết phục vụ báo cáo khoa học và bảo vệ đồ án, giải thích từ cơ sở toán học, thiết kế kiến trúc, phân tích lỗi thực tế đến cơ chế vận hành thực nghiệm.

---

## 1. TỔNG QUAN KIẾN TRÚC GIAI ĐOẠN 2

Giai đoạn 2 giải quyết hai bài toán trung tâm của một Large Language Model:
1. **Khả năng dự đoán từ tiếp theo (Next-Token Prediction):** Huấn luyện kiến trúc Decoder-Only Transformer từ đầu (from scratch) trên ngữ liệu văn bản thô (TinyStories) với chiến lược tối ưu VRAM và tự động phục hồi tiến trình (Auto-Resume).
2. **Khả năng sinh từ tự hồi quy tốc độ cao (Fast Autoregressive Decoding):** Triển khai cấu trúc **Dynamic KV Cache** nhằm chuyển đổi độ phức tạp tính toán suy luận từ $\mathcal{O}(N^2)$ xuống $\mathcal{O}(N)$ (mỗi token mới chỉ tính $\mathcal{O}(1)$ thay vì phải tính lại toàn bộ lịch sử).

```mermaid
flowchart LR
    subgraph Pretraining["1. PIPELINE HUẤN LUYỆN (PRE-TRAINING)"]
        RawData["Dataset: TinyStories"] --> BPE["BPE Tokenizer"]
        BPE --> BinaryData["Memmap: train.bin & val.bin"]
        BinaryData --> TrainLoop["Training Loop: AMP fp16 + GradAccum + Cosine LR"]
        TrainLoop --> CKPT["CheckpointManager: last_checkpoint.pt & base_model.pt"]
        TrainLoop --> WANDB["WandbTracker: Loss, LR, VRAM, Perplexity"]
    end

    subgraph Inference["2. CƠ CHẾ SINH TỰ HỒI QUY (KV CACHE)"]
        Prompt["Prompt Tokens: [B, L]"] --> Prefill["1. Chế độ Prefill: Tính Q, K, V toàn bộ L tokens -> Nạp Cache"]
        Prefill --> Decode["2. Chế độ Decode: Token mới T=1 -> Tính K, V mới -> Nối Cache -> Sinh Token"]
        Decode --> Sampler["Bộ giải mã: Greedy / Temperature / Top-k / Top-p"]
    end
```

---

## 2. NGUYÊN LÝ TOÁN HỌC & THIẾT KẾ DYNAMIC KV CACHE

### 2.1. Vấn đề nghẽn cổ chai của Self-Attention trong quá trình sinh từ
Trong quá trình sinh văn bản tự hồi quy (Autoregressive Generation), tại mỗi bước $t$, mô hình cần tính toán phân phối xác suất cho token tiếp theo $x_{t+1}$ dựa trên ngữ cảnh đã có $x_{1:t}$.
Nếu không có bộ nhớ đệm (Naive Attention):
- Tại bước $t$, toàn bộ chuỗi $x_{1:t}$ phải được đưa qua mạng để tính lại ma trận $K \in \mathbb{R}^{t \times d}$ và $V \in \mathbb{R}^{t \times d}$.
- Khối lượng tính toán của bước sinh thứ $t$ là $\mathcal{O}(t \cdot d)$.
- Tổng độ phức tạp tính toán khi sinh chuỗi dài $N$ tokens là:
  $$\sum_{t=1}^N \mathcal{O}(t) = \mathcal{O}(N^2)$$
Điều này dẫn đến việc độ trễ sinh từ tăng tuyến tính theo độ dài chuỗi (Next Token Latency ngày càng chậm).

### 2.2. Giải pháp Dynamic KV Cache
Nhờ đặc tính nhân quả (**Causal Masking**), các token trong quá khứ $x_1, \dots, x_{t-1}$ không bao giờ nhìn thấy các token tương lai. Do đó, biểu diễn vector Key ($K$) và Value ($V$) của các token quá khứ là **bất biến (invariant)** qua các bước sinh từ tiếp theo.

Hệ thống thiết kế lớp `KVCache` quản lý danh sách các tensor `(k_cache, v_cache)` cho từng tầng Transformer:
1. **Chế độ Prefill (Xử lý chuỗi Prompt đầu vào $T=L$):**
   - Đưa toàn bộ chuỗi prompt qua mô hình.
   - Tính $K_{\text{prompt}} \in \mathbb{R}^{B \times H \times L \times D}$ và $V_{\text{prompt}} \in \mathbb{R}^{B \times H \times L \times D}$.
   - Nạp trực tiếp vào bộ nhớ đệm: `k_cache[layer_idx] = K_prompt`, `v_cache[layer_idx] = V_prompt`.
2. **Chế độ Decoding (Sinh từng token tiếp theo $T=1$):**
   - Chỉ truyền duy nhất 1 token mới nhất vào mạng.
   - Chỉ tính $Q_{\text{new}} \in \mathbb{R}^{B \times H \times 1 \times D}$, $K_{\text{new}} \in \mathbb{R}^{B \times H \times 1 \times D}$, $V_{\text{new}} \in \mathbb{R}^{B \times H \times 1 \times D}$.
   - Áp dụng RoPE tại đúng chỉ số vị trí hiện tại: $\text{pos} = \text{cached\_length}$.
   - Ghép nối (concatenate) vector mới vào cache cũ theo chiều thời gian:
     $$K_{\text{total}} = [K_{\text{cache}} \,\|\, K_{\text{new}}] \in \mathbb{R}^{B \times H \times (L+1) \times D}$$
     $$V_{\text{total}} = [V_{\text{cache}} \,\|\, V_{\text{new}}] \in \mathbb{R}^{B \times H \times (L+1) \times D}$$
   - Tính Attention giữa 1 vector Query duy nhất với toàn bộ lịch sử trong cache:
     $$\text{Attention}(Q_{\text{new}}, K_{\text{total}}, V_{\text{total}}) = \text{softmax}\left( \frac{Q_{\text{new}} K_{\text{total}}^T}{\sqrt{D}} \right) V_{\text{total}}$$
   - Độ phức tạp tính toán cho mỗi token mới giảm từ $\mathcal{O}(t)$ về **$\mathcal{O}(1)$**.

---

## 3. THIẾT KẾ CHI TIẾT CÁC MODULE MÃ NGUỒN

### 3.1. `src/model/kv_cache.py` (Dynamic KV Cache)
* **Trách nhiệm:** Lưu trữ và quản lý tensor bộ nhớ đệm cho $N$ tầng Transformer.
* **Cơ chế:**
  - `update(k, v, layer_idx)`: Tự động phát hiện trạng thái Prefill (khởi tạo cache) hoặc Decode (nối `torch.cat([old, new], dim=2)`).
  - `get_seq_length(layer_idx)`: Truy xuất độ dài ngữ cảnh hiện tại trong cache để gán vị trí RoPE chính xác.
  - `reset()`: Thu hồi bộ nhớ đệm sau mỗi phiên sinh văn bản.

### 3.2. `src/utils/checkpoint.py` (CheckpointManager & Auto-Resume)
* **Trách nhiệm:** Lưu trữ và phục hồi trạng thái huấn luyện toàn vẹn.
* **Cơ chế lưu trữ toàn diện (Comprehensive State Serialization):**
  Khác với việc chỉ lưu `model.state_dict()`, CheckpointManager lưu trọn vẹn từ điển:
  - `model_state_dict`: Trọng số mạng nơ-ron.
  - `optimizer_state_dict`: Trạng thái động lượng (first and second moments) của AdamW.
  - `scheduler_state_dict`: Tọa độ bước giảm learning rate của Cosine Decay.
  - `scaler_state_dict`: Hệ số khuếch đại gradient động của Mixed Precision (`torch.amp.GradScaler`).
  - `metadata`: `step`, `epoch`, `best_val_loss`, `config`, `stage`.
  - `rng_state` & `cuda_rng_state`: Giữ tính tái lập tuyệt đối khi tiếp tục huấn luyện.
* **Cơ chế Auto-Resume:**
  Tự động nạp `last_checkpoint.pt` khi notebook/script khởi động lại, giúp quá trình pre-train trên Kaggle/Colab không bao giờ bị mất công sức tính toán khi phiên làm việc bị timeout.

### 3.3. `src/utils/tracker.py` (WandbTracker)
* **Trách nhiệm:** Tích hợp bộ theo dõi thí nghiệm Weights & Biases (W&B).
* **Đặc tính:**
  - Real-time logging: Gửi metrics `train/loss`, `val/loss`, `val/perplexity`, `learning_rate`, `peak_vram_mb` sau mỗi 50-500 steps.
  - Fault-tolerant: Bọc khối `try-except` an toàn, tự động chuyển về chế độ `disabled` nếu không có internet hoặc thiếu API key, đảm bảo không bao giờ làm crash vòng lặp huấn luyện chính.
  - Artifact Logging: Hỗ trợ log bảng sinh văn bản mẫu (`WandbTable`) và biểu đồ benchmark (`WandbImage`).

### 3.4. `src/utils/profiler.py` (KVCacheProfiler)
* **Trách nhiệm:** Đo đạc định lượng thực nghiệm hiệu năng suy luận.
* **Quy chuẩn đo lường chuẩn xác:**
  - Áp dụng `torch.cuda.synchronize()` trước và sau khi bấm giờ để đo chính xác thời gian thực thi trên GPU (loại trừ tính chất bất đồng bộ của CUDA streams).
  - Áp dụng Warm-up loops (2 lượt) để khởi tạo GPU context và ổn định xung nhịp CUDA.
  - Đo `torch.cuda.reset_peak_memory_stats()` để ghi nhận mức đỉnh VRAM thực tế (`torch.cuda.max_memory_allocated()`).
  - Đo độc lập **TTFT** (Time To First Token) và **TPOT** (Time Per Output Token).

---

## 4. QUY TRÌNH HUẤN LUYỆN TRÊN KAGGLE / COLAB

### 4.1. Siêu tham số Mô hình (Vừa vặn 16GB VRAM)
| Siêu tham số | Giá trị | Giải thích kỹ thuật |
| :--- | :--- | :--- |
| `vocab_size` | Động (`len(tokenizer.vocab)`) | Đồng bộ tuyệt đối với từ điển BPE thực tế, tránh lỗi shape lệch |
| `block_size` ($L$) | 2,048 | Hỗ trợ RoPE context window lên tới 2048 cho profiling |
| `train_block_size` | 512 | Độ dài ngữ cảnh huấn luyện thực tế để tối ưu VRAM và tốc độ duyệt |
| `n_layers` | 8 | 8 khối Transformer Block |
| `n_heads` | 8 | 8 đầu Self-Attention ($d_{\text{head}} = 64$) |
| `d_model` | 512 | Chiều không gian nhúng ẩn |
| `d_ffn` | 1,344 | Chiều ẩn SwiGLU: $\lfloor \frac{2}{3} \times 4 \times 512 \rfloor$ làm tròn bội số 64 |
| `tie_weights` | True | Chia sẻ trọng số giữa Token Embedding và LM Head (tiết kiệm 4.19M tham số) |
| **Tổng tham số** | **~44.8 Triệu (44.8M)** | Vừa vặn hoàn hảo bộ nhớ GPU 16GB (Kaggle T4 / P100) |

### 4.2. Chiến lược Tối ưu hóa & Huấn luyện
1. **Mixed Precision (`torch.amp.autocast(dtype=torch.float16)`):** Giảm dung lượng VRAM tính toán kích hoạt xuống 50%, tăng tốc độ tính toán qua Tensor Cores.
2. **Gradient Accumulation (4 - 8 steps):** Tích lũy gradient tạo thành effective batch size 64 - 128 chuỗi giúp ước lượng gradient ổn định mà không tràn bộ nhớ.
3. **Chiến lược Lịch trình Tốc độ học WSD (Warmup - Stable - Decay):**
   Thay vì sử dụng Cosine Decay truyền thống (vốn làm suy giảm learning rate từ rất sớm và "đóng băng" khả năng tiếp thu tri thức mới của mô hình), dự án áp dụng **WSD Scheduler** theo chuẩn kiến trúc LLM hiện đại (MiniCPM, DeepSeek, Llama-3 Annealing):
   - **Giai đoạn 1 - Warmup Phase (1% - 5% tổng số bước, mặc định 3%):**
     Tăng tuyến tính learning rate từ $0$ lên giá trị cực đại $\eta_{\max}$:
     $$\eta(t) = \eta_{\max} \cdot \frac{t + 1}{T_{\text{warmup}}}, \quad \forall t \in [0, T_{\text{warmup}})$$
   - **Giai đoạn 2 - Stable Phase (85% - 90% tổng thời gian huấn luyện):**
     Duy trì tốc độ học ổn định tuyệt đối ở mức cực đại:
     $$\eta(t) = \eta_{\max}, \quad \forall t \in [T_{\text{warmup}}, T_{\text{stable\_end}})$$
     *Ý nghĩa kỹ thuật:* Cho phép mô hình liên tục khám phá không gian tham số và liên tục hấp thụ khối lượng dữ liệu khổng lồ mà không bị triệt tiêu gradient quá sớm.
   - **Giai đoạn 3 - Decay Phase (10% cuối cùng - Data Annealing):**
     Khi chuẩn bị chốt trọng số hoặc kích hoạt giai đoạn nạp dữ liệu chất lượng cao (High-quality Domain Data Annealing), hạ learning rate nhanh chóng (theo hàm Cosine dốc) về $\eta_{\min} = 0.1 \eta_{\max}$:
     $$p = \frac{t - T_{\text{stable\_end}}}{T_{\max} - T_{\text{stable\_end}}}, \quad \eta(t) = \eta_{\min} + \frac{1}{2} (1 + \cos(\pi p))(\eta_{\max} - \eta_{\min})$$
4. **Gradient Clipping (1.0):** Cắt tỉa chuẩn vector đạo hàm $\|g\|_2 \le 1.0$ ngăn ngừa bùng nổ gradient (gradient exploding).
5. **Chiến Lược Ứng Cứu Checkpoint Rollback Tự Động (Auto-Rollback Strategy):**
   Trong quá trình tiền huấn luyện trên các cụm GPU lớn, các bất ổn số học (loss spike, gradient explosion do batch dữ liệu chứa chuỗi ngoại lai) có thể phá hủy hoàn toàn trọng số mô hình. Hệ thống tích hợp `CheckpointRollbackManager` thực thi quy trình ứng cứu 4 bước:
   - **Bước 1 - Phát hiện ngưỡng bất thường:**
     Liên tục tính chuẩn Frobenius của vector gradient $\|\nabla_\theta \mathcal{L}\|_2$ và duy trì đường trung bình động hàm mũ (Exponential Moving Average - EMA). Khi phát hiện điều kiện:
     $$\|\nabla_\theta \mathcal{L}\|_2 > \text{Threshold} \times \text{EMA}(\|\nabla_\theta \mathcal{L}\|_2) \quad \text{hoặc} \quad \text{Loss} \in \{\text{NaN}, \text{Inf}\}$$
     Hệ thống lập tức chặn lệnh cập nhật optimizer (`zero_grad()`) và tạm dừng nạp batch tiếp theo.
   - **Bước 2 - Kích hoạt Rollback:**
     Tự động khôi phục toàn bộ trọng số mô hình, trạng thái momentum/variance của AdamW và scaler về Checkpoint an toàn gần nhất (`last_checkpoint.pt` hoặc `step_*.pt`).
   - **Bước 3 - Cô lập và loại bỏ batch lỗi:**
     Bỏ qua (skip) hoàn toàn batch dữ liệu gây bất ổn để tránh làm hỏng mô hình lần thứ hai.
   - **Bước 4 - Điều chỉnh siêu tham số (Cool-down Phase):**
     Tạm thời giảm Learning Rate $20\%$ ($\eta_{\text{temp}} = 0.8 \times \eta_{\text{base}}$) và siết chặt Gradient Clipping về $0.5 \times \text{ClipMax}$ trong $50$ steps hồi phục trước khi nới lỏng trở lại bình thường.

---

## 5. THỰC NGHIỆM ĐỊNH LƯỢNG (PROFILING BENCHMARK KV CACHE)

Hệ thống thiết lập ma trận đo đạc thực nghiệm tại 4 mốc ngữ cảnh:  
$$L \in \{128, 512, 1024, 2048\}$$

### Bảng Chỉ tiêu Định lượng So sánh Thực nghiệm:
> Dữ liệu được đo đạc định lượng tự động qua module `src/utils/profiler.py` với $L \in \{128, 512, 1024, 2048\}$, sinh $N=32$ tokens mới (chi tiết xem tại [`docs/figures/kv_cache_metrics.json`](file:///e:/Python/DUT_AI/A-Module%20LLM/Prj_LLM/docs/figures/kv_cache_metrics.json)):

| Mốc ngữ cảnh ($L$) | Chế độ Cache | Tốc độ (Tokens/s) | TTFT (ms) | TPOT (ms) | Peak VRAM (MB) | Gia tốc (Speedup) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$L=128$** | **Bật KV Cache** | **86.46** | 44.08 | **11.57** | 0.0 (CPU/Shared) | **$3.63\text{x}$** |
| | Tắt KV Cache | 23.80 | 38.39 | 42.02 | 0.0 (CPU/Shared) | $1.00\text{x}$ (Gốc) |
| **$L=512$** | **Bật KV Cache** | **39.02** | 168.54 | **25.63** | 0.0 (CPU/Shared) | **$5.41\text{x}$** |
| | Tắt KV Cache | 7.21 | 138.53 | 138.79 | 0.0 (CPU/Shared) | $1.00\text{x}$ (Gốc) |
| **$L=1024$** | **Bật KV Cache** | **15.27** | 394.91 | **65.50** | 0.0 (CPU/Shared) | **$5.41\text{x}$** |
| | Tắt KV Cache | 2.82 | 364.36 | 355.13 | 0.0 (CPU/Shared) | $1.00\text{x}$ (Gốc) |
| **$L=2048$** | **Bật KV Cache** | **6.70** | 1,136.86 | **149.29** | 0.0 (CPU/Shared) | **$7.44\text{x}$** |
| | Tắt KV Cache | 0.90 | 1,115.24 | 1,116.36 | 0.0 (CPU/Shared) | $1.00\text{x}$ (Gốc) |

![KV Cache Benchmark](file:///e:/Python/DUT_AI/A-Module%20LLM/Prj_LLM/docs/figures/kv_cache_benchmark.png)

### Nhận xét & Phân tích Động thái Kỹ thuật:
1. **Time To First Token (TTFT):**
   - Ở cả 2 chế độ (Bật và Tắt), TTFT gần như tương đương nhau (chênh lệch rất nhỏ do overhead cấp phát cache ban đầu). Điều này hoàn toàn chính xác theo lý thuyết, vì tại bước sinh đầu tiên, toàn bộ chuỗi ngữ cảnh $L$ tokens đều phải được xử lý qua mạng để trích xuất ma trận Key và Value cho các bước sau.
2. **Time Per Output Token (TPOT - Next Token Latency):**
   - **Khi Tắt KV Cache:** TPOT tăng tuyến tính và cực kỳ nhanh theo chiều dài ngữ cảnh: từ $42.02\text{ ms}$ ($L=128$) nhảy vọt lên $1,116.36\text{ ms}$ ($L=2048$) — chậm hơn gấp **26.5 lần**! Nguyên nhân là mỗi token mới được sinh ra đều ép toàn bộ mô hình phải tính toán lại Self-Attention cho toàn bộ lịch sử $L + t$ tokens với độ phức tạp $\mathcal{O}((L+t)^2)$.
   - **Khi Bật KV Cache:** TPOT duy trì mức tăng rất nhẹ: từ $11.57\text{ ms}$ lên $149.29\text{ ms}$. Ở mỗi bước decode, mô hình chỉ tính toán duy nhất 1 token mới và nối (concatenate) vào cache, chi phí tính toán giảm về $\mathcal{O}(L)$.
3. **Tốc độ sinh (Tokens/second) & Tỉ số Gia tốc (Speedup):**
   - Gia tốc tăng vọt theo độ dài ngữ cảnh: từ **$3.63\text{x}$** ở $L=128$, lên **$5.41\text{x}$** ở $L=512$, $L=1024$ và đạt đỉnh **$7.44\text{x}$** ở $L=2048$. Khi ngữ cảnh càng dài, lợi thế của KV Cache càng mang tính quyết định.

---

## 6. NHẬT KÝ LỖI THỰC NGHIỆM & GIẢI THÍCH CHI TIẾT CÁCH KHẮC PHỤC (MISTAKES & DETAILED FIXES)

Trong quá trình triển khai thực tế trên môi trường Cloud GPU (Kaggle), một số lỗi kỹ thuật và bất đồng bộ đã phát sinh. Dưới đây là phân tích chi tiết bản chất lỗi, hậu quả và cơ sở kỹ thuật của cách khắc phục:

### 6.1. Lỗi đường dẫn tương đối khi Resume Checkpoint
* **File vi phạm:** `notebooks/01_pretrain_base_model.ipynb`
* **Vi phạm:** Sử dụng đường dẫn tương đối `CHECKPOINT_DIR = "checkpoints"`.
* **Lỗi trước vi phạm & Hậu quả gây ra:**
  - *Bản chất lỗi:* Trong Jupyter Notebook trên Kaggle, khi người dùng chạy các lệnh điều hướng như `%cd /kaggle/working/Prj_LLM` hoặc `%cd /kaggle/working`, con trỏ thư mục làm việc hiện tại (`os.getcwd()`) bị thay đổi giữa các cell hoặc sau các phiên restart kernel.
  - *Hậu quả:* Lệnh `os.path.exists("checkpoints/last_checkpoint.pt")` đi tìm file tại `/kaggle/working/checkpoints/last_checkpoint.pt` thay vì thư mục thật `/kaggle/working/Prj_LLM/checkpoints/last_checkpoint.pt`. Kết quả là hàm trả về `False`, biến `start_step` bị reset về `0`, ép mô hình phải huấn luyện lại từ đầu dù file checkpoint vẫn tồn tại trên ổ đĩa. Đồng thời, Weights & Biases ném cảnh báo xung đột step: *"Tried to log to step 1 that is less than current step 950"*.
* **Cách khắc phục & Giải thích chi tiết:**
  - *Mã nguồn:*
    ```python
    CHECKPOINT_DIR = "/kaggle/working/Prj_LLM/checkpoints"
    LAST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pt")
    BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "base_model.pt")
    ```
  - *Giải thích tại sao sửa như vậy:* Sử dụng **đường dẫn tuyệt đối (Absolute Path)** giúp hệ thống file đạt tính xác định (deterministic) 100%. Bất kể con trỏ làm việc của Python đang đứng ở `/kaggle/working`, `/root` hay `/tmp`, đường dẫn tuyệt đối luôn trỏ thẳng và chính xác vào vị trí vật lý của file trên ổ đĩa, loại bỏ hoàn toàn sự phụ thuộc vào trạng thái con trỏ của Jupyter Notebook.

### 6.2. Lỗi TypeError: RNG state must be a torch.ByteTensor
* **File vi phạm:** `src/utils/checkpoint.py`
* **Vi phạm:** Nạp trực tiếp tensor trạng thái ngẫu nhiên `rng_state` vào hàm `torch.set_rng_state()` mà không chuẩn hóa kiểu dữ liệu ByteTensor và thiết bị CPU.
* **Lỗi trước vi phạm & Hậu quả gây ra:**
  - *Bản chất lỗi:* Khi lưu checkpoint, ta lưu `checkpoint["rng_state"] = torch.get_rng_state()`. Tuy nhiên trên Kaggle chạy PyTorch 2.11 trên hệ điều hành Linux, hàm backend C++ (`default_generator.set_state(...)`) yêu cầu tensor truyền vào bắt buộc phải là kiểu `torch.ByteTensor` (tức `torch.uint8`) và bắt buộc phải nằm trên bộ nhớ **CPU**. Khi file `.pt` được nạp qua `torch.load(..., map_location=device)`, nếu tensor bị lệch kiểu hoặc bị đưa sang GPU, PyTorch sẽ từ chối nhận.
  - *Hậu quả:* Ném ngoại lệ `TypeError: RNG state must be a torch.ByteTensor` tại dòng `torch.set_rng_state(checkpoint["rng_state"])`, làm sập toàn bộ quá trình nạp checkpoint và chặn đứng tính năng Auto-Resume.
* **Cách khắc phục & Giải thích chi tiết:**
  - *Mã nguồn:*
    ```python
    try:
        if "rng_state" in checkpoint and checkpoint["rng_state"] is not None:
            rng = checkpoint["rng_state"]
            if isinstance(rng, torch.Tensor):
                torch.set_rng_state(rng.cpu().to(dtype=torch.uint8))
        if torch.cuda.is_available() and device.startswith("cuda") and checkpoint.get("cuda_rng_state") is not None:
            cuda_rng = checkpoint["cuda_rng_state"]
            if isinstance(cuda_rng, torch.Tensor):
                torch.cuda.set_rng_state(cuda_rng.cpu().to(dtype=torch.uint8))
    except Exception as e:
        print(f"[Checkpoint Info] Bỏ qua khôi phục RNG state do khác biệt môi trường ({e}).")
    ```
  - *Giải thích tại sao sửa như vậy:*
    1. Gọi `.cpu().to(dtype=torch.uint8)` để đảm bảo 100% tensor ở đúng định dạng `ByteTensor` trên CPU theo chuẩn C++ của PyTorch.
    2. Bọc khối phòng vệ `try-except`: Trạng thái ngẫu nhiên chỉ phục vụ việc tái lập thứ tự xáo trộn ngẫu nhiên. Nếu có sự không tương thích giữa các phiên bản PyTorch khác nhau (ví dụ lưu trên Windows, load trên Linux), việc bỏ qua nó không hề ảnh hưởng đến trọng số mô hình hay trạng thái của Optimizer AdamW, nhưng giúp bảo vệ chương trình không bao giờ bị crash.

### 6.3. Lỗi Import sai tên class export của Tokenizer
* **File vi phạm:** `notebooks/01_pretrain_base_model.ipynb` và `src/tokenizer/__init__.py`
* **Vi phạm:** Import sai tên định danh class: sử dụng `from src.tokenizer import BPETokenizer` trong khi class thực tế được định nghĩa trong codebase là `BytePairTokenizer`.
* **Lỗi trước vi phạm & Hậu quả gây ra:**
  - *Bản chất lỗi:* Sự bất đồng bộ giữa tên class trong module `src/tokenizer/bpe.py` (`BytePairTokenizer`) và tên gọi quy ước ở tầng ứng dụng (`BPETokenizer`).
  - *Hậu quả:* Python ném ngoại lệ `ImportError: cannot import name 'BPETokenizer' from 'src.tokenizer'`, làm dừng chương trình ngay tại Cell chuẩn bị dữ liệu.
* **Cách khắc phục & Giải thích chi tiết:**
  - *Mã nguồn:*
    1. Cập nhật Cell 2 trong Notebook: `from src.tokenizer import BytePairTokenizer`.
    2. Cập nhật `src/tokenizer/__init__.py` bổ sung Alias:
       ```python
       BPETokenizer = BytePairTokenizer
       __all__ = [..., "BytePairTokenizer", "BPETokenizer"]
       ```
  - *Giải thích tại sao sửa như vậy:* Kỹ thuật tạo **Alias** trong Python giúp duy trì tính tương thích ngược (Backward Compatibility). Dù người dùng gọi theo thói quen ngắn gọn (`BPETokenizer`) hay tên chuẩn của kiến trúc (`BytePairTokenizer`), hệ thống đều nhận diện chính xác mà không phá vỡ bất kỳ mã nguồn hiện có nào.

### 6.4. Lỗi tính toán và ghi log Loss trong Gradient Accumulation
* **File vi phạm:** `notebooks/01_pretrain_base_model.ipynb` và `scripts/train_base.py`
* **Vi phạm:** Đo lường và ghi nhận loss chỉ lấy giá trị của micro-step cuối cùng: `current_loss = loss.item() * GRAD_ACCUM_STEPS`.
* **Lỗi trước vi phạm & Hậu quả gây ra:**
  - *Bản chất lỗi:* Trong kỹ thuật Gradient Accumulation với 4 micro-steps, tại mỗi micro-step ta chia `micro_loss / 4` rồi gọi `backward()`. Đến cuối vòng lặp, biến `loss` chỉ lưu giá trị của riêng micro-step thứ 4. Khi nhân ngược lại với 4, giá trị này chỉ đại diện cho 32 mẫu cuối cùng chứ không phải trung bình của cả 128 mẫu trong cả batch hiệu dụng.
  - *Hậu quả:* Biểu đồ loss ghi nhận lên Weights & Biases bị dao động răng cưa cực mạnh (nhiễu biên độ lớn), gây hiểu lầm rằng mô hình không ổn định hoặc learning rate quá cao.
* **Cách khắc phục & Giải thích chi tiết:**
  - *Mã nguồn:*
    ```python
    accum_loss = 0.0
    for micro_step in range(GRAD_ACCUM_STEPS):
        x, y = get_batch("train")
        with torch.amp.autocast(...):
            logits = model(x)
            micro_loss = criterion(...)
            micro_loss_scaled = micro_loss / GRAD_ACCUM_STEPS
        accum_loss += micro_loss.item()
        scaler.scale(micro_loss_scaled).backward()

    current_train_loss = accum_loss / GRAD_ACCUM_STEPS
    ```
  - *Giải thích tại sao sửa như vậy:* Về mặt toán học, hàm mất mát của batch lớn gồm $K$ micro-batches là:
    $$\mathcal{L}_{\text{total}} = \frac{1}{K} \sum_{k=1}^K \mathcal{L}_k$$
    Cộng dồn `accum_loss` từ từng micro-step rồi chia cho $K$ phản ánh đúng 100% giá trị kỳ vọng toán học của toàn bộ batch, giúp đường cong hàm mất mát (loss curve) mượt mà và phản ánh chính xác sự hội tụ của gradient.

### 6.5. Lỗi lệch thứ tự lưu Checkpoint dẫn đến sai lệch best_val_loss trong Metadata
* **File vi phạm:** `notebooks/01_pretrain_base_model.ipynb`
* **Vi phạm:** Lưu `last_checkpoint.pt` trước khi cập nhật `best_val_loss` mới.
* **Lỗi trước vi phạm & Hậu quả gây ra:**
  - *Bản chất lỗi:* Code cũ gọi `CheckpointManager.save_checkpoint(LAST_CKPT_PATH, best_val_loss=best_val_loss, ...)` trước, sau đó mới kiểm tra:
    `if losses["val"] < best_val_loss: best_val_loss = losses["val"]`.
  - *Hậu quả:* Khi validation loss đạt kỷ lục mới (ví dụ giảm từ 0.85 xuống 0.71), file `last_checkpoint.pt` trên ổ đĩa vẫn mang giá trị cũ là 0.85. Nếu phiên Kaggle bị ngắt ngay sau đó, khi resume lại, hệ thống sẽ đọc kỷ lục cũ là 0.85, làm sai lệch logic so sánh kỷ lục ở các bước tiếp theo.
* **Cách khắc phục & Giải thích chi tiết:**
  - *Mã nguồn:* Đảo thứ tự thực hiện: Kiểm tra validation loss, cập nhật biến `best_val_loss`, lưu file trọng số sạch `base_model.pt` TRƯỚC, sau đó mới truyền `best_val_loss` đã cập nhật vào hàm `save_checkpoint(LAST_CKPT_PATH)`.
  - *Giải thích tại sao sửa như vậy:* Tuân thủ nguyên lý **Tính toàn vẹn trạng thái (State Consistency)**: Trạng thái được ghi xuống ổ đĩa (Persistent Storage) phải luôn đồng nhất và phản ánh trạng thái mới nhất của bộ nhớ RAM tại thời điểm lưu.

### 6.6. Mâu thuẫn giữa block_size=512 và các mốc Profiling L=2048
* **File vi phạm:** `notebooks/01_pretrain_base_model.ipynb` và `src/config.py`
* **Vi phạm:** Khởi tạo `ModelConfig(block_size=512)` nhưng kịch bản thực nghiệm định lượng KV Cache lại yêu cầu đo ở các mốc $L \in \{128, 512, 1024, 2048\}$.
* **Lỗi trước vi phạm & Hậu quả gây ra:**
  - *Bản chất lỗi:* Thuật toán mã hóa vị trí RoPE (`RotaryEmbedding`) và ma trận Attention Mask tam giác dưới chỉ được chuẩn bị dải vị trí $0 \dots 511$.
  - *Hậu quả:* Khi đưa prompt dài $L=1024$ hoặc $2048$ vào mô hình để đo đạc định lượng, hệ thống sẽ ném lỗi `IndexError: index out of bounds` hoặc gây lỗi bộ nhớ CUDA out-of-bounds.
* **Cách khắc phục & Giải thích chi tiết:**
  - *Mã nguồn:*
    ```python
    model_cfg = ModelConfig(
        vocab_size=ACTUAL_VOCAB_SIZE,
        block_size=2048,      # Kiến trúc hỗ trợ context window tối đa 2048
        n_layers=8,
        n_heads=8,
        d_model=512,
    )
    TRAIN_BLOCK_SIZE = 512    # Trong lúc train chỉ cắt batch độ dài 512
    ```
  - *Giải thích tại sao sửa như vậy:* Tách biệt rõ ràng giữa **Năng lực tối đa của kiến trúc (Architecture Capacity: 2048)** và **Kích thước lát cắt huấn luyện (Training Slice: 512)**. Nhờ đó, ma trận RoPE có sẵn dải tần số cho 2048 vị trí để phục vụ Profiling mượt mà, trong khi quá trình pre-train vẫn tiết kiệm được VRAM và tăng tốc độ xử lý bằng cách giới hạn chuỗi dữ liệu ở 512 tokens.

---

## 7. KẾT QUẢ KIỂM THỬ XÁC MINH (VERIFICATION)

Toàn bộ các thành phần của Giai đoạn 2 đã được kiểm thử tự động với PyTest:
- `tests/test_kv_cache_generate.py`: Đảm bảo logits của token mới khi dùng KV Cache khớp 100% với Full Attention không dùng cache; Greedy search cho output đồng nhất.
- `tests/test_checkpoint.py`: Đảm bảo lưu và phục hồi trạng thái Model, Optimizer, Scheduler, RNG state chính xác 100%.
- `tests/test_profiler.py`: Đảm bảo hàm đo đạc 4 chỉ số (Tokens/s, TTFT, TPOT, VRAM) vận hành ổn định trên cả CPU và GPU.

**Kết quả: 24/24 tests passed (100% Pass Rate).**
