"""
==============================================================================
FILE: frontend/components/profiling_tab.py
MODULE: frontend.components.profiling_tab
RESPONSIBILITY:
    - Đơn trách nhiệm: Trực quan hóa kết quả đo đạc thực nghiệm KV Cache (Profiling Dashboard).
    - Hiển thị bảng biểu và đồ thị định lượng theo yêu cầu bắt buộc của Capstone:
      1. Tốc độ sinh từ (Tokens/giây) Bật vs Tắt KV Cache với L ∈ {128, 512, 1024, 2048}.
      2. First Token Latency (TTFT) vs Next Token Latency (TPOT).
      3. Mức chiếm dụng bộ nhớ VRAM đỉnh (Peak VRAM).
INPUT:
    - File dữ liệu log benchmark JSON từ `scripts/profile_kv.py`.
OUTPUT:
    - Biểu đồ Matplotlib/Plotly và bảng số liệu so sánh định lượng.
DEPENDENCY:
    - Độc lập với runtime inference, chỉ đọc và hiển thị dữ liệu thực nghiệm đã ghi lại.
==============================================================================
"""

import gradio as gr


def render_profiling_tab() -> None:
    """Render giao diện hiển thị biểu đồ và số liệu thực nghiệm KV Cache."""
    gr.Markdown(
        """
        ### 📊 Bảng Đo đạc Thực nghiệm KV Cache (Quantitative Profiling Dashboard)
        *Nghiệm thu Yêu cầu Cốt lõi: Đo lường định lượng hiệu năng tại các mốc ngữ cảnh $L \in \{128, 512, 1024, 2048\}$*
        """
    )

    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("#### 📈 Chọn Tiêu chí So sánh")
            metric_type = gr.Radio(
                choices=[
                    "Tốc độ sinh từ (Tokens/giây) - Bật vs Tắt Cache",
                    "Độ trễ First Token (TTFT) vs Next Token (TPOT)",
                    "Mức chiếm dụng VRAM Đỉnh (MB) - Peak Memory",
                ],
                value="Tốc độ sinh từ (Tokens/giây) - Bật vs Tắt Cache",
                label="Chỉ số Đo lường",
            )
            run_live_profile = gr.Button("Chạy Benchmark Nhanh (L=128..512)", variant="secondary")
            gr.Markdown(
                """
                > **Ghi chú kỹ thuật:**
                > - **Không có KV Cache:** Độ phức tạp tính toán là $O(T^2)$, tốc độ sinh từ giảm dốc theo hàm parabol khi độ dài context tăng.
                > - **Có KV Cache:** Độ phức tạp giảm xuống $O(T)$, tốc độ ổn định với đánh đổi dung lượng bộ nhớ VRAM tuyến tính $O(T)$.
                """
            )

        with gr.Column(scale=8):
            profiling_plot = gr.Plot(label="Biểu đồ Đo đạc Thực nghiệm")
            profiling_table = gr.Dataframe(
                headers=["Độ dài L", "Tokens/s (Tắt Cache)", "Tokens/s (Bật Cache)", "Tăng tốc (Speedup)", "VRAM Bật Cache (MB)"],
                value=[
                    ["128", "45.2", "88.6", "1.96x", "124 MB"],
                    ["512", "18.4", "84.1", "4.57x", "186 MB"],
                    ["1024", "7.1", "81.5", "11.47x", "270 MB"],
                    ["2048", "2.3", "77.8", "33.82x", "438 MB"],
                ],
                label="Bảng Số liệu Thực nghiệm Mẫu",
            )
