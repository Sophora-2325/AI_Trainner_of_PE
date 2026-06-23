"""Web 控制台入口 — 浏览器 UI + 最新 MediaPipe Tasks API.

无需降级 mediapipe，无需 OpenCV 弹窗，避免 GUI 子进程退出码 120。

运行:
  python web_coach.py
  python web_coach.py --movement squat --no-llm
"""

import argparse
import os
import signal
import socket
import sys
import threading
import webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

from src.utils.paths import setup_runtime, resource_path

setup_runtime()


def find_free_port(preferred: int = 8080, attempts: int = 20) -> int:
    """从 preferred 起寻找可用端口."""
    for port in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError(f"无法在 {preferred}-{preferred + attempts - 1} 范围内找到可用端口")


def start_http_server(port: int) -> tuple[ThreadingHTTPServer, int]:
    web_dir = resource_path("web")
    if not os.path.isdir(web_dir):
        os.makedirs(web_dir, exist_ok=True)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=web_dir, **kwargs)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[WebCoach] 控制台: http://127.0.0.1:{port}/coach_dashboard.html")
    return server, port


def main():
    parser = argparse.ArgumentParser(description="AI 健身教练 — Web 控制台")
    parser.add_argument("--movement", "-m", default="squat",
                        choices=["squat", "deadlift", "pushup", "pullup", "plank", "shooting"])
    parser.add_argument("--port", type=int, default=8080, help="Web 控制台端口 (被占用时自动递增)")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-opensim", action="store_true", default=True)
    parser.add_argument("--video", "-v", default=None, help="分析视频（不打开摄像头）")
    parser.add_argument("--camera", type=int, default=None, help="摄像头设备 ID (默认 0)")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    port = find_free_port(args.port)
    if port != args.port:
        print(f"[WebCoach] 端口 {args.port} 已被占用，改用 {port}")
    http_server, port = start_http_server(port)
    url = f"http://127.0.0.1:{port}/coach_dashboard.html"

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    from app import FitnessCoach, load_config

    config = load_config(resource_path("config", "settings.yaml"))
    config["use_opensim"] = not args.no_opensim
    config["use_llm"] = not args.no_llm
    config["display_mode"] = "headless" if args.video else "web"
    config["ws_enabled"] = True
    if args.camera is not None:
        config.setdefault("camera", {})["device_id"] = args.camera

    coach = FitnessCoach(config)
    signal.signal(signal.SIGINT, lambda s, f: coach.stop())

    print(f"\n{'='*52}")
    print("  AI 健身教练 — Web 控制台模式")
    print(f"  浏览器: {url}")
    print("  终止: 在终端按 Ctrl+C")
    print(f"{'='*52}\n")

    try:
        coach.start(movement=args.movement, video_path=args.video)
    except Exception as e:
        print(f"[WebCoach] 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        coach.stop()
        http_server.shutdown()


if __name__ == "__main__":
    main()
