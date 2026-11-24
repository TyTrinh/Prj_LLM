# DANH MỤC VÀ ĐƠN TRÁCH NHIỆM TỪNG FILE (SINGLE RESPONSIBILITY CATALOG)

Tài liệu này định nghĩa rõ ràng nguyên lý **Đơn Trách Nhiệm (Single Responsibility Principle - SRP)** cho từng file trong toàn bộ dự án. Mỗi file chỉ chịu trách nhiệm cho một lý do thay đổi duy nhất.

---

## 1. TẦNG GIAO DIỆN NGƯỜI DÙNG (FRONTEND LAYER)

| Đường dẫn File | Đơn Trách Nhiệm (SRP) | Đầu Vào (Inputs) | Đầu Ra (Outputs) |
| :--- | :--- | :--- | :--- |
| `demo.py` | Điểm khởi chạy (Entry Point) của web app. Nhận tham số host/port và kích hoạt Gradio server. | CLI arguments (`--host`, `--port`) | Khởi chạy server tại `http://localhost:7860` |
| `frontend/__init__.py` | Đóng gói package frontend và expose hàm `create_ui`. | Không | Module export |
| `frontend/app.py` | Lắp ráp các tab thành một Dashboard hoàn chỉnh; quản lý theme và bố cục chính. | Các component con | Đối tượng `gr.Blocks` hoàn chỉnh |
| `frontend/components/chat_tab.py` | Quản lý màn hình hội thoại đơn lẻ và các nút điều chỉnh siêu tham số giải mã. | Prompt, Temperature, Top-p, Top-k, Max Tokens | Luồng tin nhắn Chatbot cập nhật |
| `frontend/components/comparison_tab.py` | Quản lý màn hình so sánh đối đầu song song 3 mô hình (Base vs SFT vs DPO + Guardrails). | 1 Prompt thử nghiệm chung | 3 khung phản hồi cạnh nhau + độ trễ đo được |
| `frontend/components/guardrails_tab.py` | Quản lý màn hình kiểm thử và thanh tra các mẫu tấn công Prompt Injection/Jailbreak. | Adversarial Prompt | Trạng thái kích hoạt màng lọc Tầng 1 / Tầng 2 |
| `frontend/components/profiling_tab.py` | Quản lý màn hình trực quan hóa đồ thị thực nghiệm KV Cache và bảng số liệu định lượng. | File số liệu JSON từ script profile | Biểu đồ Matplotlib và Dataframe so sánh |

---

## 2. TẦNG ĐIỀU PHỐI SUY LUẬN (APPLICATION / INFERENCE LAYER)

| Đường dẫn File | Đơn Trách Nhiệm (SRP) | Đầu Vào (Inputs) | Đầu Ra (Outputs) |
| :--- | :--- | :--- | :--- |
| `src/inference/__init__.py` | Expose `InferenceEngine` và `Sampler` cho các tầng bên ngoài. | Không | Module export |
| `src/inference/engine.py` | Điều phối luồng sinh văn bản từ đầu đến cuối: Guardrail $\rightarrow$ Tokenize $\rightarrow$ Model $\rightarrow$ Sampler $\rightarrow$ Decode. | Prompt (str) + cấu hình sinh từ | Kết quả văn bản (str) + metrics độ trễ |
| `src/inference/sampler.py` | Biến đổi phân phối Logits thành Token ID tiếp theo qua Greedy, Temperature, Top-k, Top-p. | `logits` [B, vocab_size] + siêu tham số | `next_token_id` [B, 1] |

---

## 3. TẦNG THUẬT TOÁN CỐT LÕI (DOMAIN CORE LAYER)

### A. Khối Kiến trúc Mô hình (`src/model/`)
| Đường dẫn File | Đơn Trách Nhiệm (SRP) | Đầu Vào (Inputs) | Đầu Ra (Outputs) |
| :--- | :--- | :--- | :--- |
| `src/model/layers.py` | Cài đặt các khối toán học nguyên thủy: RMSNorm, RoPE, SwiGLU Feed-Forward. | Hidden states `[B, T, D]` | Tensor sau chuẩn hóa / biến đổi `[B, T, D]` |
| `src/model/attention.py` | Tính toán Causal Multi-Head Self-Attention kết hợp mặt nạ tam giác dưới và tích hợp cache. | Hidden states `[B, T, D]`, KVCache | Tensor sau chú ý `[B, T, D]` |
| `src/model/kv_cache.py` | Quản lý cấu trúc bộ đệm tensor Key và Value động cho từng tầng phục vụ autoregressive. | Token mới $K, V$ `[B, H, 1, D_h]` | Chuỗi tích lũy $K, V$ `[B, H, L, D_h]` |
| `src/model/transformer.py` | Lắp ráp TransformerBlock và toàn bộ GPTModel có Weight Tying và chuẩn khởi tạo trọng số. | Chuỗi token ID `[B, T]` | Logits xác suất `[B, T, vocab_size]` |

### B. Khối Xử lý Từ vựng (`src/tokenizer/`)
| Đường dẫn File | Đơn Trách Nhiệm (SRP) | Đầu Vào (Inputs) | Đầu Ra (Outputs) |
| :--- | :--- | :--- | :--- |
| `src/tokenizer/special_tokens.py` | Định nghĩa danh sách và chuỗi hằng số cho các Special Tokens (`<|pad|>`, `<|user|>`, ...). | Không | Các hằng số chuỗi |
| `src/tokenizer/bpe.py` | Triển khai thuật toán Byte-Pair Encoding từ số 0 (train, encode, decode, lưu/nạp vocab). | Văn bản (str) hoặc Token IDs (List[int]) | Token IDs hoặc Văn bản giải mã |

### C. Khối Tinh chỉnh Hiệu quả Tham số (`src/peft/`)
| Đường dẫn File | Đơn Trách Nhiệm (SRP) | Đầu Vào (Inputs) | Đầu Ra (Outputs) |
| :--- | :--- | :--- | :--- |
| `src/peft/lora_layer.py` | Module `LoRALinear`: Phân rã hạng thấp $h = W_0x + \frac{\alpha}{r}BAx$, hỗ trợ bật/tắt và merge. | Tensor đầu vào `[..., d_in]` | Tensor đầu ra `[..., d_out]` |
| `src/peft/lora_manager.py` | Tiêm LoRA vào mô hình, đóng băng trọng số gốc, gập trọng số `merge_and_unload()`. | `GPTModel` | Mô hình đã gắn adapter LoRA |

### D. Khối Căn chỉnh Hành vi (`src/alignment/`)
| Đường dẫn File | Đơn Trách Nhiệm (SRP) | Đầu Vào (Inputs) | Đầu Ra (Outputs) |
| :--- | :--- | :--- | :--- |
| `src/alignment/dpo_loss.py` | Tính hàm mất mát DPO Loss, Implicit Reward Margin và Length Normalization. | Logps của Policy và Reference | Giá trị Loss (vô hướng) + Reward margins |
| `src/alignment/preference_dataset.py` | Xử lý tập dữ liệu cặp ưu tiên (`hh-rlhf`) và tạo `response_mask` tách biệt prompt. | Cặp mẫu JSON/Dict | PyTorch Tensors sẵn sàng cho DataLoader |

### E. Khối Màng bọc An toàn (`src/guardrails/`)
| Đường dẫn File | Đơn Trách Nhiệm (SRP) | Đầu Vào (Inputs) | Đầu Ra (Outputs) |
| :--- | :--- | :--- | :--- |
| `src/guardrails/input_guard.py` | Rà quét prompt đầu vào để phát hiện và chặn Prompt Injection, Jailbreak, ký tự tràn. | Prompt (str) | `(is_safe: bool, reason: str)` |
| `src/guardrails/output_guard.py` | Rà quét câu trả lời của mô hình để phát hiện lỗi lặp từ vô tận và nội dung độc hại. | Response (str) | `(is_safe: bool, reason: str, text: str)` |
| `src/guardrails/pipeline.py` | Đóng gói màng bọc 2 tầng thống nhất bọc lấy vòng đời sinh từ của mô hình. | Prompt hoặc Response | Quyết định an toàn + Text chuẩn mực |

---

## 4. TẦNG HẠ TẦNG VÀ TIỆN ÍCH (INFRASTRUCTURE & UTILITIES LAYER)

| Đường dẫn File | Đơn Trách Nhiệm (SRP) | Đầu Vào (Inputs) | Đầu Ra (Outputs) |
| :--- | :--- | :--- | :--- |
| `src/config.py` | Tập trung hóa toàn bộ tham số cấu hình hệ thống bằng các Typed Dataclasses. | Không | `ModelConfig`, `LoRAConfig`, `DPOConfig`, ... |
| `src/utils/profiler.py` | Đo đạc định lượng tốc độ Tokens/s, TTFT, TPOT và VRAM với `torch.cuda.synchronize()`. | Mô hình, độ dài L, use_kv_cache | Đối tượng `ProfilingResult` |
| `src/utils/checkpoint.py` | Quản lý lưu và nạp riêng biệt trọng số Base Model (`.pt`) và Adapter LoRA (`.pt`). | Mô hình và đường dẫn file | File nhị phân lưu trữ an toàn |
| `src/utils/seed.py` | Thiết lập seed toàn cục đảm bảo tái lập 100% kết quả thử nghiệm. | Seed (int) | Cố định trạng thái ngẫu nhiên |
| `src/utils/logger.py` | Cung cấp Logger có định dạng chuẩn xuất ra Console và File. | Tên module (str) | Đối tượng `logging.Logger` |

---

## 5. TẦNG SCRIPT DÒNG LỆNH (CLI RUNNERS)

| Đường dẫn File | Đơn Trách Nhiệm (SRP) | Đầu Vào (Inputs) | Đầu Ra (Outputs) |
| :--- | :--- | :--- | :--- |
| `scripts/train_base.py` | Kích hoạt huấn luyện Base Model từ dữ liệu nhị phân `train.bin`. | CLI Flags | Checkpoint `base_model.pt` |
| `scripts/train_sft.py` | Kích hoạt huấn luyện SFT LoRA trên tập Alpaca-Cleaned. | CLI Flags | Checkpoint `sft_lora.pt` |
| `scripts/train_dpo.py` | Kích hoạt huấn luyện DPO Alignment trên tập hh-rlhf. | CLI Flags | Checkpoint `dpo_lora.pt` |
| `scripts/profile_kv.py` | Chạy bộ kiểm thử định lượng KV Cache và tự động xuất biểu đồ PNG + số liệu JSON. | CLI Flags | `kv_cache_benchmark.png`, JSON |
| `scripts/eval_judge.py` | Gọi LLM-as-a-Judge (GPT-4o-mini/Claude) chấm điểm đối đầu và tính Win Rate %. | CLI Flags + API Key | Báo cáo Markdown `eval_results.md` |
