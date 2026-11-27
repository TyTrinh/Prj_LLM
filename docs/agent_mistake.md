# AGENT MISTAKE LOG (NHẬT KÝ LỖI, NGUYÊN NHÂN & BIỆN PHÁP KHẮC PHỤC)

> **Mục đích:** Ghi nhận toàn bộ các lỗi phát sinh trong quá trình thiết kế, triển khai, chạy thực nghiệm trên môi trường Cloud (Kaggle/Colab) và kiểm thử nhằm ngăn ngừa việc lặp lại các sai lầm trong tương lai (theo Điều 20, 21, 22 trong `docs/Constitution.md`).
>
> **Cấu trúc bắt buộc:**
> 1. File vi phạm
> 2. Vi phạm gì?
> 3. Lỗi trước vi phạm & Hậu quả gây ra
> 4. Cách khắc phục & Giải thích chi tiết tại sao khắc phục như vậy

---

## MISTAKE 01: LỖI ĐƯỜNG DẪN TƯƠNG ĐỐI TRONG CHECKPOINT PATH TRÊN KAGGLE

* **1. File vi phạm:** `notebooks/01_pretrain_base_model.ipynb`
* **2. Vi phạm gì:** Sử dụng đường dẫn tương đối không an toàn (`CHECKPOINT_DIR = "checkpoints"`) trong môi trường thực thi Jupyter Notebook trên Cloud.
* **3. Lỗi trước vi phạm & Hậu quả gây ra:**
  - **Bản chất lỗi:** Trong Jupyter Notebook trên Kaggle, khi người dùng chạy các lệnh điều hướng như `%cd /kaggle/working/Prj_LLM` hoặc `%cd /kaggle/working`, con trỏ thư mục làm việc hiện tại (`os.getcwd()`) bị thay đổi giữa các cell hoặc sau các phiên restart kernel.
  - **Hậu quả:** Lệnh `os.path.exists("checkpoints/last_checkpoint.pt")` đi tìm file tại `/kaggle/working/checkpoints/last_checkpoint.pt` thay vì thư mục thật `/kaggle/working/Prj_LLM/checkpoints/last_checkpoint.pt`. Kết quả là hàm trả về `False`, biến `start_step` bị reset về `0`, ép mô hình phải huấn luyện lại từ đầu dù file checkpoint vẫn tồn tại trên ổ đĩa. Đồng thời, Weights & Biases ném cảnh báo xung đột step: *"Tried to log to step 1 that is less than current step 950"*.
* **4. Cách khắc phục & Giải thích chi tiết:**
  - **Mã nguồn khắc phục:**
    ```python
    CHECKPOINT_DIR = "/kaggle/working/Prj_LLM/checkpoints"
    LAST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pt")
    BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "base_model.pt")
    ```
  - **Giải thích lý do:** Sử dụng **đường dẫn tuyệt đối (Absolute Path)** giúp hệ thống file đạt tính xác định (deterministic) 100%. Bất kể con trỏ làm việc của Python đang đứng ở `/kaggle/working`, `/root` hay `/tmp`, đường dẫn tuyệt đối luôn trỏ thẳng và chính xác vào vị trí vật lý của file trên ổ đĩa, loại bỏ hoàn toàn sự phụ thuộc vào trạng thái con trỏ của Jupyter Notebook.

---

## MISTAKE 02: LỖI KHÔNG TƯƠNG THÍCH KIỂU DỮ LIỆU RNG STATE TRONG PYTORCH (TYPEERROR)

* **1. File vi phạm:** `src/utils/checkpoint.py`
* **2. Vi phạm gì:** Nạp trực tiếp tensor trạng thái ngẫu nhiên `rng_state` vào hàm `torch.set_rng_state()` mà không chuẩn hóa kiểu dữ liệu ByteTensor và thiết bị CPU.
* **3. Lỗi trước vi phạm & Hậu quả gây ra:**
  - **Bản chất lỗi:** Khi lưu checkpoint, ta lưu `checkpoint["rng_state"] = torch.get_rng_state()`. Tuy nhiên trên Kaggle chạy PyTorch 2.11 trên hệ điều hành Linux, hàm backend C++ (`default_generator.set_state(...)`) yêu cầu tensor truyền vào bắt buộc phải là kiểu `torch.ByteTensor` (tức `torch.uint8`) và bắt buộc phải nằm trên bộ nhớ **CPU**. Khi file `.pt` được nạp qua `torch.load(..., map_location=device)`, nếu tensor bị lệch kiểu hoặc bị đưa sang GPU, PyTorch sẽ từ chối nhận.
  - **Hậu quả:** Ném ngoại lệ `TypeError: RNG state must be a torch.ByteTensor` tại dòng `torch.set_rng_state(checkpoint["rng_state"])`, làm sập toàn bộ quá trình nạp checkpoint và chặn đứng tính năng Auto-Resume.
* **4. Cách khắc phục & Giải thích chi tiết:**
  - **Mã nguồn khắc phục:**
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
  - **Giải thích lý do:**
    1. Gọi `.cpu().to(dtype=torch.uint8)` để đảm bảo 100% tensor ở đúng định dạng `ByteTensor` trên CPU theo chuẩn C++ của PyTorch.
    2. Bọc khối phòng vệ `try-except`: Trạng thái ngẫu nhiên chỉ phục vụ việc tái lập thứ tự xáo trộn ngẫu nhiên. Nếu có sự không tương thích giữa các phiên bản PyTorch khác nhau (ví dụ lưu trên Windows, load trên Linux), việc bỏ qua nó không hề ảnh hưởng đến trọng số mô hình hay trạng thái của Optimizer AdamW, nhưng giúp bảo vệ chương trình không bao giờ bị crash.

---

## MISTAKE 03: LỖI IMPORT SAI TÊN CLASS EXPORT CỦA TOKENIZER

* **1. File vi phạm:** `notebooks/01_pretrain_base_model.ipynb` và `src/tokenizer/__init__.py`
* **2. Vi phạm gì:** Import sai tên định danh class: sử dụng `from src.tokenizer import BPETokenizer` trong khi class thực tế được định nghĩa trong codebase là `BytePairTokenizer`.
* **3. Lỗi trước vi phạm & Hậu quả gây ra:**
  - **Bản chất lỗi:** Sự bất đồng bộ giữa tên class trong module `src/tokenizer/bpe.py` (`BytePairTokenizer`) và tên gọi quy ước ở tầng ứng dụng (`BPETokenizer`).
  - **Hậu quả:** Python ném ngoại lệ `ImportError: cannot import name 'BPETokenizer' from 'src.tokenizer'`, làm dừng chương trình ngay tại Cell chuẩn bị dữ liệu.
* **4. Cách khắc phục & Giải thích chi tiết:**
  - **Mã nguồn khắc phục:**
    1. Cập nhật Cell 2 trong Notebook: `from src.tokenizer import BytePairTokenizer`.
    2. Cập nhật `src/tokenizer/__init__.py` bổ sung Alias:
       ```python
       BPETokenizer = BytePairTokenizer
       __all__ = [..., "BytePairTokenizer", "BPETokenizer"]
       ```
  - **Giải thích lý do:** Kỹ thuật tạo **Alias** trong Python giúp duy trì tính tương thích ngược (Backward Compatibility). Dù người dùng gọi theo thói quen ngắn gọn (`BPETokenizer`) hay tên chuẩn của kiến trúc (`BytePairTokenizer`), hệ thống đều nhận diện chính xác mà không phá vỡ bất kỳ mã nguồn hiện có nào.

---

## MISTAKE 04: LỆCH NHỊP GIỮA TÍNH LOSS GRADIENT ACCUMULATION VÀ LOGGING

* **1. File vi phạm:** `notebooks/01_pretrain_base_model.ipynb` và `scripts/train_base.py`
* **2. Vi phạm gì:** Đo lường và ghi nhận loss chỉ lấy giá trị của micro-step cuối cùng: `current_loss = loss.item() * GRAD_ACCUM_STEPS`.
* **3. Lỗi trước vi phạm & Hậu quả gây ra:**
  - **Bản chất lỗi:** Trong kỹ thuật Gradient Accumulation với 4 micro-steps, tại mỗi micro-step ta chia `micro_loss / 4` rồi gọi `backward()`. Đến cuối vòng lặp, biến `loss` chỉ lưu giá trị của riêng micro-step thứ 4. Khi nhân ngược lại với 4, giá trị này chỉ đại diện cho 32 mẫu cuối cùng chứ không phải trung bình của cả 128 mẫu trong cả batch hiệu dụng.
  - **Hậu quả:** Biểu đồ loss ghi nhận lên Weights & Biases bị dao động răng cưa cực mạnh (nhiễu biên độ lớn), gây hiểu lầm rằng mô hình không ổn định hoặc learning rate quá cao.
* **4. Cách khắc phục & Giải thích chi tiết:**
  - **Mã nguồn khắc phục:**
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
  - **Giải thích lý do:** Về mặt toán học, hàm mất mát của batch lớn gồm $K$ micro-batches là:
    $$\mathcal{L}_{\text{total}} = \frac{1}{K} \sum_{k=1}^K \mathcal{L}_k$$
    Cộng dồn `accum_loss` từ từng micro-step rồi chia cho $K$ phản ánh đúng 100% giá trị kỳ vọng toán học của toàn bộ batch, giúp đường cong hàm mất mát (loss curve) mượt mà và phản ánh chính xác sự hội tụ của gradient.

---

## MISTAKE 05: LỆCH THỨ TỰ LƯU CHECKPOINT DẪN ĐẾN SAI LỆCH BEST_VAL_LOSS TRONG METADATA

* **1. File vi phạm:** `notebooks/01_pretrain_base_model.ipynb`
* **2. Vi phạm gì:** Lưu `last_checkpoint.pt` trước khi cập nhật `best_val_loss` mới.
* **3. Lỗi trước vi phạm & Hậu quả gây ra:**
  - **Bản chất lỗi:** Code cũ gọi `CheckpointManager.save_checkpoint(LAST_CKPT_PATH, best_val_loss=best_val_loss, ...)` trước, sau đó mới kiểm tra:
    `if losses["val"] < best_val_loss: best_val_loss = losses["val"]`.
  - **Hậu quả:** Khi validation loss đạt kỷ lục mới (ví dụ giảm từ 0.85 xuống 0.71), file `last_checkpoint.pt` trên ổ đĩa vẫn mang giá trị cũ là 0.85. Nếu phiên Kaggle bị ngắt ngay sau đó, khi resume lại, hệ thống sẽ đọc kỷ lục cũ là 0.85, làm sai lệch logic so sánh kỷ lục ở các bước tiếp theo.
* **4. Cách khắc phục & Giải thích chi tiết:**
  - **Mã nguồn khắc phục:** Đảo thứ tự thực hiện: Kiểm tra validation loss, cập nhật biến `best_val_loss`, lưu file trọng số sạch `base_model.pt` TRƯỚC, sau đó mới truyền `best_val_loss` đã cập nhật vào hàm `save_checkpoint(LAST_CKPT_PATH)`.
  - **Giải thích lý do:** Tuân thủ nguyên lý **Tính toàn vẹn trạng thái (State Consistency)**: Trạng thái được ghi xuống ổ đĩa (Persistent Storage) phải luôn đồng nhất và phản ánh trạng thái mới nhất của bộ nhớ RAM tại thời điểm lưu.

---

## MISTAKE 06: MÂU THUẪN GIỮA BLOCK_SIZE=512 VÀ CÁC MỐC PROFILING L=2048

* **1. File vi phạm:** `notebooks/01_pretrain_base_model.ipynb` và `src/config.py`
* **2. Vi phạm gì:** Khởi tạo `ModelConfig(block_size=512)` nhưng kịch bản thực nghiệm định lượng KV Cache lại yêu cầu đo ở các mốc $L \in \{128, 512, 1024, 2048\}$.
* **3. Lỗi trước vi phạm & Hậu quả gây ra:**
  - **Bản chất lỗi:** Thuật toán mã hóa vị trí RoPE (`RotaryEmbedding`) và ma trận Attention Mask tam giác dưới chỉ được chuẩn bị dải vị trí $0 \dots 511$.
  - **Hậu quả:** Khi đưa prompt dài $L=1024$ hoặc $2048$ vào mô hình để đo đạc định lượng, hệ thống sẽ ném lỗi `IndexError: index out of bounds` hoặc gây lỗi bộ nhớ CUDA out-of-bounds.
* **4. Cách khắc phục & Giải thích chi tiết:**
  - **Mã nguồn khắc phục:**
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
  - **Giải thích lý do:** Tách biệt rõ ràng giữa **Năng lực tối đa của kiến trúc (Architecture Capacity: 2048)** và **Kích thước lát cắt huấn luyện (Training Slice: 512)**. Nhờ đó, ma trận RoPE có sẵn dải tần số cho 2048 vị trí để phục vụ Profiling mượt mà, trong khi quá trình pre-train vẫn tiết kiệm được VRAM và tăng tốc độ xử lý bằng cách giới hạn chuỗi dữ liệu ở 512 tokens.

---

### 7. Lỗi Kaggle Secrets không tự nạp khiến Notebook bị treo hỏi API Key W&B
* **File vi phạm:** `notebooks/LLM_01_pretrain_base_model.ipynb`
* **Vi phạm:** Chỉ đọc cứng duy nhất tên nhãn `"WANDB_API_KEY"` và gọi fallback `wandb.login()` trần khi không thấy key.
* **Lỗi trước vi phạm & Hậu quả gây ra:**
  - *Bản chất lỗi:* Trên Kaggle, người dùng có thể đặt nhãn bí mật khác tên (như `wandb`, `wandb_key`) hoặc đã tạo Secret nhưng **quên chưa tích ô vuông (checkbox)** trong menu `Add-ons -> Secrets` để gắn secret vào notebook đang mở. Khi đó, hàm `user_secrets.get_secret("WANDB_API_KEY")` ném ngoại lệ hoặc trả về rỗng. Code cũ gọi tiếp lệnh `wandb.login()`, khiến Jupyter Notebook kích hoạt interactive prompt yêu cầu nhập API key bằng bàn phím.
  - *Hậu quả:* Cell bị đứng chờ nhập liệu, người dùng ngỡ rằng Kaggle không nhận key dù đã thêm vào Secrets.
* **Cách khắc phục & Giải thích chi tiết:**
  - *Mã nguồn:*
    ```python
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        for secret_name in ["WANDB_API_KEY", "wandb_api_key", "wandb_key", "wandb", "WANDB"]:
            try:
                val = user_secrets.get_secret(secret_name)
                if val and val.strip():
                    WANDB_API_KEY = val.strip()
                    break
            except Exception:
                continue
    except Exception:
        pass
    ```
    Nếu không tìm thấy, tự động fallback an toàn sang `WANDB_MODE="offline"` kèm hướng dẫn rõ ràng, không gọi `wandb.login()` rỗng để tránh treo kernel.
  - *Giải thích lý do:* Tăng tính thích ứng (fault-tolerant) cho notebook, tự động hỗ trợ các cách đặt tên nhãn khác nhau của người dùng và ngăn chặn hoàn toàn việc block kernel khi chạy không có giám sát.
