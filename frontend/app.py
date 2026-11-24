"""
==============================================================================
FILE: frontend/app.py
MODULE: frontend.app
RESPONSIBILITY:
    - Đơn trách nhiệm: Entry point chính của tầng Presentation (Gradio Web UI).
    - Lắp ráp các component/tab độc lập thành một Dashboard tương tác hoàn chỉnh.
    - Quản lý theme, CSS và vòng đời khởi tạo ứng dụng web.
INPUT:
    - Các callback và service từ tầng Application (InferenceEngine).
OUTPUT:
    - Đối tượng `gr.Blocks` sẵn sàng phục vụ trình duyệt web qua `demo.launch()`.
DEPENDENCY:
    - Phụ thuộc vào `gradio` và các component con trong `frontend.components.*`.
    - Không trực tiếp chứa mã thuật toán toán học của mô hình.
==============================================================================
"""

import gradio as gr
from frontend.components.chat_tab import render_chat_tab
from frontend.components.comparison_tab import render_comparison_tab
from frontend.components.guardrails_tab import render_guardrails_tab
from frontend.components.profiling_tab import render_profiling_tab


def create_ui() -> gr.Blocks:
    """Khởi tạo giao diện Dashboard web hoàn chỉnh với 4 tab nghiệp vụ."""
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    )

    with gr.Blocks(theme=theme, title="LLM Capstone Dashboard - Track B") as demo:
        gr.Markdown(
            """
            # 🚀 LLM Capstone Project: Alignment & Safety (Track B)
            **Mô hình Ngôn ngữ Lớn Decoder-Only xây dựng từ số 0 bằng PyTorch thuần**  
            *Hệ thống tích hợp: Custom BPE, Dynamic KV Cache, LoRA PEFT, DPO Alignment & Two-Tier Guardrails*
            """
        )

        with gr.Tabs():
            with gr.TabItem("💬 Trò chuyện Tương tác (Interactive Chat)"):
                render_chat_tab()

            with gr.TabItem("⚖️ So sánh Đối đầu (Base vs SFT vs DPO)"):
                render_comparison_tab()

            with gr.TabItem("🛡️ Kiểm tra An toàn & Rào chắn (Guardrails Monitor)"):
                render_guardrails_tab()

            with gr.TabItem("📊 Đo đạc Thực nghiệm KV Cache (Profiling Benchmark)"):
                render_profiling_tab()

    return demo


if __name__ == "__main__":
    app = create_ui()
    app.launch(share=False)
