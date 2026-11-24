# HƯỚNG DẪN ĐỒNG BỘ HUẤN LUYỆN: LOCAL vs CLOUD (KAGGLE / COLAB)

Quy trình chuẩn kỹ thuật để phát triển mô hình lớn: **Code & Test tại Local -> Đồng bộ lên Cloud GPU để Train -> Kéo Checkpoint về Local để Demo & Đánh giá**.

---

## 1. VÌ SAO KHÔNG NÊN VIẾT TOÀN BỘ CODE TRONG NOTEBOOK?
- **Nhược điểm của việc code trên Notebook Cloud:** Rất dễ mất code khi timeout/disconnect, khó quản lý phiên bản Git, code bị trùng lặp (*spaghetti code*), khó tái lập kết quả một lệnh.
- **Giải pháp Clean Architecture:**
  - Toàn bộ kiến trúc Model, Tokenizer, LoRA, DPO, Guardrail nằm ở thư mục `src/`.
  - Kaggle/Colab chỉ đóng vai trò là **môi trường phần cứng (Compute Runner)** thực thi các file trong `scripts/` hoặc import các module từ `src/`.

---

## 2. QUY TRÌNH 4 BƯỚC TRIỂN KHAI

```
  [ LOCAL MACHINE ]                                   [ CLOUD GPU (Kaggle/Colab) ]
  1. Viết code sạch trong `src/`                      
  2. Viết Unit Test kiểm tra shape [B, T] -> [B, T, V]
  3. Git push lên GitHub ─────────────────────────►   4. Clone repo: git clone <url>
                                                      5. Cài đặt: pip install -e .
                                                      6. Chạy: python scripts/train_base.py
  8. Tải trọng số về `checkpoints/` ◄──────────────   7. Lưu file checkpoint .pt
  9. Chạy Demo Gradio & Đánh giá offline
```

### Bước 1: Phát triển & Kiểm thử tại Local
- Viết code các module trong `src/`.
- Chạy unit test kiểm tra tensor shape, forward/backward pass trên CPU với batch nhỏ (`B=2, T=8`).
- Commit và push code lên GitHub repo cá nhân:
  ```bash
  git add .
  git commit -m "feat: implement transformer core & lora module"
  git push origin main
  ```

### Bước 2: Thiết lập môi trường trên Kaggle / Colab
Trong Notebook Kaggle hoặc Colab (chọn GPU T4/P100), chỉ cần chạy 1 cell duy nhất để chuẩn bị:
```python
# Cell 1: Clone repo và cài đặt dạng editable package
!git clone https://github.com/<your-username>/Prj_LLM.git
%cd Prj_LLM
!pip install -e .
```
> Nhờ có `pyproject.toml`, lệnh `pip install -e .` giúp bạn có thể `import src.model` từ bất kỳ đâu trên Cloud mà không bao giờ bị lỗi `ModuleNotFoundError`.

### Bước 3: Chạy script huấn luyện trên Cloud
Không cần copy-paste hàng nghìn dòng code vào notebook, bạn chỉ cần gọi script:
```python
# Pre-training Base Model
!python scripts/train_base.py --epochs 3 --batch_size 32

# Hoặc Fine-tuning LoRA SFT
!python scripts/train_sft.py --base_model checkpoints/base_model.pt

# Hoặc DPO Alignment
!python scripts/train_dpo.py --sft_model checkpoints/sft_lora.pt
```

### Bước 4: Kéo Checkpoint về Local
Sau khi train xong, bạn chỉ cần tải các file sau về đặt vào thư mục `checkpoints/` trên máy tính:
1. `checkpoints/vocab.json` & `checkpoints/merges.txt` (~vài trăm KB)
2. `checkpoints/base_model.pt` (~150MB–200MB đối với model 40M–60M)
3. `checkpoints/sft_lora.pt` (~3MB–5MB cực kỳ nhẹ!)
4. `checkpoints/dpo_lora.pt` (~3MB–5MB)

Tại máy Local, bạn chỉ cần chạy:
```bash
python demo.py
```
Giao diện Gradio sẽ tự động nạp Base Model cùng các adapter LoRA để bạn và người chấm trực tiếp tương tác, test jailbreak và so sánh câu trả lời.
