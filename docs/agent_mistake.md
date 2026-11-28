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

---

## MISTAKE 08: LỆCH ĐƯỜNG DẪN CHECKPOINT KHI CON TRỎ LÀM VIỆC BỊ THAY ĐỔI TRÊN KAGGLE (CƠ CHẾ RESUME THẤT BẠI)

* **1. File vi phạm:** `notebooks/LLM_01_pretrain_base_model.ipynb` và `scripts/train_base.py`
* **2. Vi phạm gì:** Sử dụng `CHECKPOINT_DIR = os.path.abspath('checkpoints')` phụ thuộc vào con trỏ thư mục hiện tại (`os.getcwd()`), dẫn đến việc tìm checkpoint ở thư mục rỗng và ép mô hình chạy lại từ Step 0 dù checkpoint cũ vẫn tồn tại.
* **3. Lỗi trước vi phạm & Hậu quả gây ra:**
  - **Bản chất lỗi:** Trên môi trường Kaggle, hệ thống tệp tin phân tầng thành `/kaggle/working` (thư mục root mặc định) và `/kaggle/working/Prj_LLM` (thư mục repo git). Sau khi người dùng mở lại notebook hoặc sau khi restart kernel, con trỏ làm việc có thể đứng ở `/kaggle/working`.
  - Lệnh `os.path.abspath('checkpoints')` tự động dịch thành `/kaggle/working/checkpoints/` (thư mục rỗng), trong khi file thực tế `step_500.pt`, `step_1000.pt`, `last_checkpoint.pt` được lưu trong `/kaggle/working/Prj_LLM/checkpoints/`.
  - **Hậu quả:** Hàm `os.path.exists()` kiểm tra trả về `False`, biến `candidate_ckpt` bị gán bằng `None`. Hệ thống kết luận sai lầm rằng *"chưa từng có checkpoint nào"* và in ra dòng: `🌱 Bắt đầu huấn luyện mới hoàn toàn từ Step 0`, xóa sạch công sức huấn luyện trước đó.
* **4. Cách khắc phục & Giải thích chi tiết:**
  - **Mã nguồn khắc phục:** Tự động đối chiếu đa tầng danh sách các đường dẫn ứng viên có thể xảy ra:
    ```python
    possible_ckpt_dirs = [
        '/kaggle/working/Prj_LLM/checkpoints',
        '/kaggle/working/checkpoints',
        os.path.abspath('checkpoints'),
    ]
    CHECKPOINT_DIR = next((d for d in possible_ckpt_dirs if os.path.exists(d)), '/kaggle/working/Prj_LLM/checkpoints')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Quét toàn diện tất cả các step_*.pt để bắt mốc step cao nhất
    all_step_ckpts = []
    for d in possible_ckpt_dirs:
        if os.path.exists(d):
            all_step_ckpts.extend(glob.glob(os.path.join(d, 'step_*.pt')))
            
    if all_step_ckpts:
        candidate_ckpt = max(all_step_ckpts, key=os.path.getmtime)
    elif os.path.exists(os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pt')):
        candidate_ckpt = os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pt')
    ```
  - **Giải thích lý do:** Đảm bảo tính thích ứng đa môi trường (Environment Invariance). Hệ thống chủ động kiểm tra sự tồn tại thực tế của file trên đĩa cứng thay vì tin tưởng một đường dẫn tương đối duy nhất, bảo vệ 100% tiến trình Auto-Resume.

---

## MISTAKE 09: LỖI THỨ TỰ KHỞI TẠO BIẾN TRONG CELL JUPYTER (NAMEERROR: CANDIDATE_CKPT & LR)

* **1. File vi phạm:** `notebooks/LLM_01_pretrain_base_model.ipynb` (Cell 4.5 và Cell 5)
* **2. Vi phạm gì:** Gọi sử dụng biến trước khi khai báo định nghĩa biến (`candidate_ckpt` trước khi tìm checkpoint; `lr` thay vì `effective_lr` trong vòng lặp log).
* **3. Lỗi trước vi phạm & Hậu quả gây ra:**
  - **Bản chất lỗi:** 
    1. Lệnh `rollback_mgr.register_safe_checkpoint(candidate_ckpt)` được đặt ở dòng 57, trong khi khối code gán `candidate_ckpt = None` và tìm file lại nằm ở dòng 59 trở đi.
    2. Trong vòng lặp huấn luyện, biến được cập nhật cho optimizer là `effective_lr = base_lr * rollback_mgr.current_lr_multiplier`, nhưng dòng `print()` và `tracker.log()` lại gọi tên biến cũ chưa khai báo là `lr`.
  - **Hậu quả:** Ngay khi restart session và chạy lại cell, Python ném ngoại lệ `NameError: name 'candidate_ckpt' is not defined` và `NameError: name 'lr' is not defined`, làm gián đoạn toàn bộ phiên làm việc của người dùng.
* **4. Cách khắc phục & Giải thích chi tiết:**
  - **Mã nguồn khắc phục:**
    1. Đưa toàn bộ logic tìm kiếm `candidate_ckpt` lên TRƯỚC khi khởi tạo `rollback_mgr`.
    2. Đổi đồng bộ tất cả các chỗ in và log sang `effective_lr`:
       ```python
       print(f'... | LR: {effective_lr:.6f} | ...')
       tracker.log({'train/learning_rate': effective_lr, ...})
       ```
  - **Giải thích lý do:** Tuân thủ nguyên tắc **Định nghĩa trước, Sử dụng sau (Declare Before Use)**. Trong môi trường tương tác notebook, các cell phải độc lập, tự chứa (self-contained) và có thứ tự thực thi từ trên xuống dưới nghiêm ngặt để không bao giờ bị lỗi phụ thuộc trạng thái bộ nhớ đệm (stale kernel state).
---

## MISTAKE 10: LỖI RUNTIMEERROR GRADSCALER "UNSCALE_() HAS ALREADY BEEN CALLED" KHI KÍCH HOẠT CHECKPOINT ROLLBACK

* **1. File vi phạm:** `src/utils/checkpoint.py` và `notebooks/LLM_01_pretrain_base_model.ipynb` (Cell 11)
* **2. Vi phạm gì:** Khi phát hiện bất thường và gọi lệnh `continue` bỏ qua batch lỗi, trạng thái nội bộ của `torch.amp.GradScaler` vẫn giữ nguyên cờ `OptState.UNSCALED` từ lần gọi `scaler.unscale_(optimizer)` trước đó mà không được `scaler.update()` reset lại. Đồng thời ngưỡng phát hiện dị thường quá nhạy cảm (`3.0 x EMA` khi norm chỉ 1.91) dẫn đến việc kích hoạt rollback nhầm lẫn và chọn lùi về `step_1500.pt` thay vì `step_2000.pt`.
* **3. Lỗi trước vi phạm & Hậu quả gây ra:**
  - **Bản chất lỗi:**
    1. Cơ chế của PyTorch `GradScaler`: trong 1 vòng lặp bước tối ưu, sau khi gọi `scaler.unscale_(optimizer)`, đối tượng scaler chuyển trạng thái thành `UNSCALED`. Nếu sau đó gặp điều kiện rollback và dùng lệnh `continue`, vòng lặp bước sang step tiếp theo mà không đi qua `scaler.step(optimizer)` và `scaler.update()`. Đến đầu step mới, khi gọi `scaler.unscale_(optimizer)` lần nữa, PyTorch phát hiện cờ trạng thái chưa được dọn dẹp và lập tức ném: `RuntimeError: unscale_() has already been called on this optimizer since the last update()`.
    2. Ngưỡng phát hiện dị thường quá thấp: `grad_norm = 1.91` vốn là giá trị hoàn toàn bình thường đối với mô hình 44.8M tham số, nhưng do `grad_norm_ema` trước đó thấp (~0.6), phép so sánh `1.91 > 3.0 * 0.6` kích hoạt còi báo động sai.
    3. Bộ nhớ `last_safe_checkpoint_path` chưa được đăng ký kịp thời tại thời điểm lưu `step_2000.pt`, khiến hàm tìm kiếm lùi sâu về `step_1500.pt`, làm lãng phí 500 steps huấn luyện.
  - **Hậu quả:** Tiến trình huấn luyện bị crash hoàn toàn tại Step 2210 do ngoại lệ RuntimeError không được xử lý, dừng đứng kernel trên Kaggle.
* **4. Cách khắc phục & Giải thích chi tiết:**
  - **Mã nguồn khắc phục:**
    1. Thêm phương thức `find_latest_safe_checkpoint()` quét tự động các file `step_*.pt` mới nhất và `last_checkpoint.pt` trên đĩa cứng.
    2. Trong `trigger_rollback()`, gọi bắt buộc `scaler.update()` và `optimizer.zero_grad(set_to_none=True)` để dọn sạch hoàn toàn trạng thái nội bộ của `GradScaler`:
       ```python
       if scaler is not None:
           scaler.update()
       ```
    3. Hiệu chỉnh lại điều kiện báo động: bổ sung ngưỡng an toàn sàn `min_grad_norm_threshold = 5.0` và nâng `threshold_factor = 5.0`:
       ```python
       threshold = max(self.threshold_factor * self.grad_norm_ema, self.min_grad_norm_threshold)
       ```
    4. Trong notebook Cell 11: Đăng ký `rollback_mgr.register_safe_checkpoint(...)` ngay khi vừa lưu `step_*.pt` và `last_checkpoint.pt`.
  - **Giải thích lý do:** Tuân thủ đúng vòng đời quản lý trạng thái của PyTorch Mixed Precision `torch.amp.GradScaler`. Bằng cách reset trạng thái thông qua `scaler.update()` ngay trong thủ tục rollback, chu kỳ huấn luyện tiếp theo có thể tiếp tục trơn tru mà không bị xung đột cờ trạng thái nội bộ. Đồng thời, việc cân chỉnh ngưỡng giúp tránh báo động giả (False Positives) đối với các biến động gradient thông thường.

