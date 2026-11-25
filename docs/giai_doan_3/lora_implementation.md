# TÀI LIỆU KỸ THUẬT: TRIỂN KHAI MODULE LORA (LOW-RANK ADAPTATION) TỪ SỐ 0

> **Module:** `src.peft`  
> **Source Files:** [`src/peft/lora_layer.py`](../src/peft/lora_layer.py), [`src/peft/lora_manager.py`](../src/peft/lora_manager.py)  
> **Test File:** [`tests/test_lora_freeze.py`](../tests/test_lora_freeze.py)  
> **Tham chiếu khoa học:** Hu et al. (2021) *"LoRA: Low-Rank Adaptation of Large Language Models"* (Microsoft).

---

## 1. Cơ sở Lý thuyết & Công thức Toán học

Khi tinh chỉnh (fine-tuning) toàn bộ các tham số của mô hình ngôn ngữ lớn $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, ma trận cập nhật trọng số $\Delta W$ thường có "chiều nội tại" (intrinsic dimension) rất thấp. 

Cơ chế LoRA tham số hóa $\Delta W$ thông qua tích của hai ma trận hạng thấp:
$$\Delta W = \frac{\alpha}{r} B A$$
Trong đó:
- $r \ll \min(d_{\text{in}}, d_{\text{out}})$ là hạng (rank), thường chọn $r \in \{4, 8, 16\}$.
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$.
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$.
- $\alpha$ là hệ số tỷ lệ không đổi (scaling factor). Giá trị scaling thực tế là $\frac{\alpha}{r}$.

Công thức lan truyền xuôi (Forward Pass):
$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x$$

### Chiến lược Khởi tạo Trọng số Chuẩn
Để đảm bảo trạng thái mô hình ngay lúc bắt đầu huấn luyện tương đương 100% với mô hình gốc:
1. **Ma trận $B$:** Khởi tạo bằng toàn số `0` ($B = 0$). Khi đó $\Delta W = 0$, nghĩa là ban đầu nhánh LoRA không gây bất kỳ độ lệch nào lên output của base model.
2. **Ma trận $A$:** Khởi tạo theo phân phối Gaussian chuẩn $\mathcal{N}\left(0, \frac{1}{r}\right)$ theo bài báo gốc của Microsoft, giúp variance của activations và gradients ổn định khi rank $r$ thay đổi.

---

## 2. Kiến trúc Module `src.peft`

Hệ thống được thiết kế theo nguyên lý Single Responsibility Principle (SRP) gồm 2 thành phần chính:

### 2.1. Lớp `LoRALinear` (`src/peft/lora_layer.py`)
Bọc quanh một lớp `nn.Linear` có sẵn:
- **Nhánh song song:** Đầu vào $x$ vừa đi qua `base_layer(x)`, vừa đi qua `(dropout(x) @ A.T @ B.T) * (alpha / r)`.
- **Chế độ Toggle `enabled`:** Khi `enabled=False`, lớp bỏ qua nhánh LoRA và chỉ tính `base_layer(x)`. Đây là tính năng cốt lõi cho giai đoạn DPO (Direct Preference Optimization) để tính xác suất của Reference Policy $\pi_{\text{ref}}$ mà không cần nhân đôi bộ nhớ VRAM để lưu 2 mô hình.
- **In-place Merge/Unmerge:** Hỗ trợ `merge()` và `unmerge()` trực tiếp lên `base_layer.weight.data`.
- **Cải tiến Zero-Latency Inference (`merge_and_unload()`):** Gập vĩnh viễn $\Delta W$ vào $W_0$ và hoàn trả lại đối tượng `nn.Linear` nguyên bản, loại bỏ hoàn toàn chi phí overhead của phép tính ma trận phụ trợ trong quá trình phục vụ suy luận.

### 2.2. Bộ quản lý `LoRAManager` (`src/peft/lora_manager.py`)
Cung cấp các API cấp cao để quản lý vòng đời adapter:
- `inject_lora(model, r, lora_alpha, target_modules)`: Tự động quét cây module và thay thế các ma trận chiếu đích (mặc định là `q_proj` và `v_proj`) thành `LoRALinear`.
- `freeze_base_model(model)`: Đóng băng toàn bộ trọng số gốc (`requires_grad = False`), chỉ mở khóa các tensor có tiền tố `"lora_"`.
- `set_lora_enabled(model, enabled: bool)`: Bật/tắt nhánh LoRA trên toàn bộ mô hình trong một lệnh gọi.
- `merge_and_unload(model)`: Thay thế toàn bộ các module con `LoRALinear` bằng `nn.Linear` đã gập trọng số trên toàn cây mô hình.
- `save_lora_weights(model, path, metadata)`: Chỉ trích xuất các tensor `lora_A` và `lora_B` cùng metadata cấu hình ($r$, $\alpha$) để lưu ra file `.pt`. Dung lượng file chỉ vài chục KB đến vài MB (thay vì hàng trăm MB nếu lưu cả base model).
- `load_lora_weights(model, path, strict)`: Nạp các tensor adapter đã lưu vào mô hình đã inject LoRA tương ứng.
- `get_trainable_parameters(model)`: Thống kê số lượng tham số trainable, tổng tham số và tỷ lệ phần trăm (thường `< 3%`).

---

## 3. Kết quả Kiểm thử & Nghiệm thu (Verification)

Tất cả các chức năng đã được xác minh toàn diện trong [`tests/test_lora_freeze.py`](../tests/test_lora_freeze.py):

| Test Case | Mục tiêu kiểm chứng | Kết quả |
| :--- | :--- | :---: |
| `test_lora_math_and_forward` | Xác minh công thức toán học $W_0 x + \frac{\alpha}{r} B A x$ và $B=0 \Rightarrow \Delta W = 0$ | **PASSED** |
| `test_lora_enable_toggle` | Xác minh bật/tắt nhánh LoRA phục vụ DPO reference policy | **PASSED** |
| `test_lora_injection_and_freezing` | Xác minh tiêm vào GPTModel, đóng băng base model, trainable params < 3% | **PASSED** |
| `test_merge_and_unload_zero_latency` | Xác minh gập trọng số, dỡ bỏ wrapper LoRALinear, output sai số `< 1e-5` | **PASSED** |
| `test_lora_save_and_load_weights` | Xác minh lưu/nạp riêng rẽ Adapter weights file nhẹ (< 50 KB), logits khớp 100% | **PASSED** |

```bash
============================= test session starts =============================
collected 5 items
tests/test_lora_freeze.py::test_lora_math_and_forward PASSED             [ 20%]
tests/test_lora_freeze.py::test_lora_enable_toggle PASSED                [ 40%]
tests/test_lora_freeze.py::test_lora_injection_and_freezing PASSED       [ 60%]
tests/test_lora_freeze.py::test_merge_and_unload_zero_latency PASSED     [ 80%]
tests/test_lora_freeze.py::test_lora_save_and_load_weights PASSED        [100%]
============================== 5 passed in 5.45s ==============================
```
