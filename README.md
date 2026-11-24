# LLM From Scratch: Alignment & Safety (Track B)

Đồ án cuối khóa môn Large Language Models: **Xây dựng, Tối ưu và Căn chỉnh Mô hình Ngôn ngữ Lớn từ Số 0**  
Chuyên sâu: **Hướng B - Tinh chỉnh Hiệu quả, Căn chỉnh Hành vi & An toàn (Alignment & Safety)**

---

## 🎯 Điểm Nổi Bật Của Hệ Thống
1. **Khung Cốt Lõi Bắt Buộc (Core):**
   - Tự xây dựng kiến trúc **Decoder-Only Transformer** bằng PyTorch thuần (RMSNorm, RoPE, SwiGLU, Weight Tying).
   - Tự lập trình cấu trúc dữ liệu **Dynamic KV Cache** phục vụ Autoregressive Decoding.
   - Tự xây dựng thuật toán mã hóa từ vựng **Byte-Pair Encoding (BPE)** từ số 0.
2. **Chuyên Sâu Hướng B (Alignment & Safety):**
   - Module **PEFT (LoRA)** tự viết từ số 0: Gắn adapter vào $W_q, W_v$ và đóng băng trọng số gốc (< 3% trainable params).
   - Hàm mất mát **DPO (Direct Preference Optimization)** tự viết không cần Reward Model.
   - Hệ thống màng lọc an toàn **Hybrid Guardrails (2 tầng)** chống Prompt Injection, Jailbreak và lỗi lặp từ.
3. **Cải Tiến Kỹ Thuật (10% điểm Rubric):**
   - **LoRA Zero-Overhead Weight Merging & Adapter Hot-swapping:** Cho phép gập trọng số trực tiếp vào base model lúc inference và hoán đổi adapter linh hoạt.
   - **Length-Normalized DPO Variant:** Khử hiện tượng Length Bias (mô hình thiên vị câu trả lời dài lan man).
   - **Two-Tier Hybrid Guardrail Architecture:** Kết hợp Fast-path (Regex heuristic) và Deep-path (Semantic output filter).

---

## 📂 Tài Liệu Kiến Trúc & Thiết Kế Hệ Thống
- 📐 [Tổng quan Kiến trúc Clean Architecture](docs/overview_architecture.md)
- 🔄 [Chi tiết các Luồng Hoạt động & Luồng Dữ liệu](docs/detailed_code_flow.md)
- 📋 [Bảng Đơn Trách Nhiệm Từng File (Single Responsibility Catalog)](docs/file_responsibilities.md)
- ☁️ [Hướng dẫn Huấn luyện trên Kaggle/Colab & Đồng bộ Local](notebooks/cloud_workflow_guide.md)
- 📌 [Bảng Theo dõi Tiến độ Dự án](TASKS.md)

---

## 🚀 Cài Đặt & Chạy Demo Một Lệnh (One-Click Run)

### 1. Khởi tạo môi trường
Dự án quản lý môi trường bằng công cụ hiện đại `uv` (Python 3.12):
```bash
# Tạo môi trường ảo và cài đặt gói
uv venv --python 3.12 .venv
.\.venv\Scripts\activate
uv pip install -e .
```

### 2. Kiểm thử tự động (Unit Tests)
Toàn bộ kiến trúc và các hàm toán học đều có test kiểm chứng:
```bash
pytest -v
```

### 3. Khởi chạy Giao diện Web Gradio Dashboard
Khởi chạy giao diện tương tác đa tab trực quan:
```bash
python demo.py
```
Truy cập tại: `http://localhost:7860` để trải nghiệm:
- Tab 1: **Interactive Chat** (Chat với mô hình có KV Cache & Guardrails).
- Tab 2: **Tri-mode Comparison** (So sánh song song Base vs SFT vs DPO).
- Tab 3: **Safety Guardrails Monitor** (Thanh tra kiểm thử Prompt Injection / Jailbreak).
- Tab 4: **KV Cache Profiling Dashboard** (Biểu đồ Tokens/s, Latency, Peak VRAM).
