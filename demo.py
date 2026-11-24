"""
==============================================================================
FILE: demo.py
MODULE: demo (Root Entry Point)
RESPONSIBILITY:
    - Đơn trách nhiệm: File thực thi chính kích hoạt toàn bộ hệ thống web demo một lệnh.
    - Đọc cấu hình môi trường, khởi tạo server Gradio và lắng nghe yêu cầu người dùng.
INPUT:
    - CLI arguments (port, host, share).
OUTPUT:
    - Khởi chạy web server tại http://localhost:7860.
DEPENDENCY:
    - Phụ thuộc vào `frontend.app.create_ui`.
==============================================================================
"""

import argparse
from frontend.app import create_ui


def main():
    parser = argparse.ArgumentParser(description="Khởi chạy LLM Alignment & Safety Demo Dashboard")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP để bind server")
    parser.add_argument("--port", type=int, default=7860, help="Port để chạy Gradio app")
    parser.add_argument("--share", action="store_true", help="Bật chia sẻ link public qua Gradio")
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 KHỞI CHẠY HỆ THỐNG DEMO: ALIGNMENT & SAFETY LLM (TRACK B)")
    print(f"📍 Địa chỉ: http://{args.host}:{args.port}")
    print("=" * 70)

    app = create_ui()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
