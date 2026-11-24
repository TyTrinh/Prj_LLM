"""
==============================================================================
FILE: frontend/components/chat_tab.py
MODULE: frontend.components.chat_tab
RESPONSIBILITY:
    - Đơn trách nhiệm: Quản lý giao diện trò chuyện đơn lẻ (Interactive Chatbot).
    - Cung cấp các nút điều chỉnh siêu tham số giải mã: Temperature, Top-K, Top-P,
      bật/tắt KV Cache và bật/tắt lớp Guardrails.
INPUT:
    - Tin nhắn người dùng (text), cấu hình giải mã (temperature, top_p, top_k, max_tokens).
OUTPUT:
    - Cập nhật luồng tin nhắn (Chatbot history) thời gian thực.
DEPENDENCY:
    - Nhận hàm callback sinh từ từ `src.inference.engine.InferenceEngine`.
==============================================================================
"""

import gradio as gr


def render_chat_tab() -> None:
    """Render giao diện màn hình chat đơn lẻ."""
    with gr.Row():
        with gr.Column(scale=8):
            chatbot = gr.Chatbot(label="Hội thoại với Mô hình", height=500)
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Nhập câu lệnh hoặc câu hỏi của bạn tại đây...",
                    label="Tin nhắn",
                    scale=9,
                )
                send_btn = gr.Button("Gửi ↵", variant="primary", scale=1)
            with gr.Row():
                clear_btn = gr.Button("Xóa hội thoại")

        with gr.Column(scale=4):
            gr.Markdown("### ⚙️ Cấu hình Giải mã (Decoding Parameters)")
            model_mode = gr.Radio(
                choices=["DPO Model (Khuyến nghị)", "SFT Model", "Base Model"],
                value="DPO Model (Khuyến nghị)",
                label="Phiên bản Mô hình",
            )
            use_guardrails = gr.Checkbox(value=True, label="Kích hoạt Hybrid Guardrails (Bảo vệ 2 tầng)")
            use_kv_cache = gr.Checkbox(value=True, label="Kích hoạt Dynamic KV Cache (Tăng tốc độ)")
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p (Nucleus Sampling)")
            top_k = gr.Slider(1, 100, value=50, step=1, label="Top-k Sampling")
            max_tokens = gr.Slider(16, 512, value=128, step=16, label="Số lượng Token tối đa sinh ra")

    # Placeholder handlers for later binding
    def _mock_respond(history, message):
        history = history or []
        history.append((message, "[Hệ thống: Sẵn sàng kết nối InferenceEngine khi nạp weights]"))
        return history, ""

    send_btn.click(_mock_respond, inputs=[chatbot, user_input], outputs=[chatbot, user_input])
    user_input.submit(_mock_respond, inputs=[chatbot, user_input], outputs=[chatbot, user_input])
    clear_btn.click(lambda: None, None, chatbot, queue=False)
