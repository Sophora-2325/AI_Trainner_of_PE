"""AI 健身教练 — 一键启动器（GUI 入口）.

功能:
  - Web 控制台（推荐）
  - 摄像头实时教练（OpenCV 窗口）
  - 视频分析模式
  - 课程流程自动验收
  - 查看评分历史折线图

运行:
  python launcher.py
"""

import os
import sys
import threading
import webbrowser
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from src.utils.paths import setup_runtime, resource_path, data_path

setup_runtime()

MOVEMENTS = [
    ("squat", "深蹲"),
    ("deadlift", "硬拉"),
    ("pushup", "俯卧撑"),
    ("pullup", "引体向上"),
    ("plank", "平板支撑"),
]

# Windows: 从 tkinter 子进程启动 OpenCV 窗口时，未分配控制台常导致异常退出码 120
CREATE_NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)


def check_dependencies() -> list[str]:
    """启动前检查关键依赖，返回错误列表."""
    errors = []

    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision  # noqa: F401
        ver = getattr(mp, "__version__", "未知")
        if not hasattr(mp, "tasks"):
            errors.append(f"MediaPipe {ver} 缺少 Tasks API，请升级: pip install -U mediapipe")
    except ImportError:
        errors.append("未安装 mediapipe，请运行: pip install -r requirements.txt")

    try:
        import cv2  # noqa: F401
    except ImportError:
        errors.append("未安装 opencv-python")

    return errors


def explain_exit_code(code: int, stderr: str = "") -> str:
    """将子进程退出码翻译为用户可读的说明."""
    if code == 0:
        return ""

    detail = stderr.strip()
    if len(detail) > 800:
        detail = detail[:800] + "\n...(已截断)"

    if code == 120:
        base = (
            "程序异常退出（退出码 120）。\n\n"
            "这是 Windows 上从 GUI 启动子进程时的常见现象，通常表示：\n"
            "  1. OpenCV 摄像头窗口无法从 GUI 子进程创建\n"
            "  2. 缺少 Visual C++ 运行库\n\n"
            "建议先在终端测试:\n"
            "  python web_coach.py --no-llm"
        )
    elif code == 1:
        base = "程序运行失败（退出码 1）。"
    else:
        base = f"程序退出码: {code}"

    if detail:
        return f"{base}\n\n详细错误:\n{detail}"
    return base


def subprocess_flags(new_console: bool = False) -> dict:
    """构造 subprocess.run 的 Windows 兼容参数."""
    kwargs = {"env": os.environ.copy()}
    if sys.platform == "win32" and new_console:
        kwargs["creationflags"] = CREATE_NEW_CONSOLE
    return kwargs


class CoachLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI 健身教练")
        self.geometry("480x420")
        self.resizable(False, False)
        self._coach_thread = None
        self._running = False

        self._build_ui()
        self._show_startup_warnings()

    def _show_startup_warnings(self):
        errors = check_dependencies()
        if errors:
            messagebox.showwarning(
                "环境检查",
                "检测到以下问题，部分功能可能无法使用:\n\n" + "\n\n".join(errors),
            )

    def _build_ui(self):
        pad = {"padx": 16, "pady": 6}

        ttk.Label(self, text="AI 健身教练", font=("Microsoft YaHei UI", 16, "bold")).pack(pady=(16, 4))
        ttk.Label(self, text="一键启动完整训练流程", font=("Microsoft YaHei UI", 9)).pack()

        frame = ttk.LabelFrame(self, text="训练设置", padding=12)
        frame.pack(fill="x", **pad)

        ttk.Label(frame, text="动作:").grid(row=0, column=0, sticky="w")
        self.movement_var = tk.StringVar(value="squat")
        movement_combo = ttk.Combobox(
            frame, textvariable=self.movement_var, state="readonly", width=28,
            values=[f"{code} — {name}" for code, name in MOVEMENTS],
        )
        movement_combo.current(0)
        movement_combo.grid(row=0, column=1, sticky="ew")
        frame.columnconfigure(1, weight=1)

        self.open_3d_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="自动打开 3D 孪生页面", variable=self.open_3d_var).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self.no_llm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="关闭大模型（无需 Ollama，仅规则建议）", variable=self.no_llm_var).grid(
            row=2, column=0, columnspan=2, sticky="w")

        self.no_opensim_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="关闭 OpenSim（推荐，无需 WSL）", variable=self.no_opensim_var).grid(
            row=3, column=0, columnspan=2, sticky="w")

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", **pad)

        ttk.Button(btn_frame, text="Web 控制台 (推荐)", command=self._start_web).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="开始实时教练 (OpenCV窗口)", command=self._start_live).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="分析视频文件", command=self._start_video).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="自动验收（第1-6周）", command=self._run_auto_demo).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="查看评分历史", command=self._show_history).pack(fill="x", pady=2)

        self.status = tk.StringVar(value="就绪")
        ttk.Label(self, textvariable=self.status, foreground="#666").pack(pady=(4, 12))

    def _selected_movement(self) -> str:
        text = self.movement_var.get()
        return text.split(" — ")[0] if " — " in text else "squat"

    def _open_3d_page(self):
        html = resource_path("sports_twin.html")
        if os.path.exists(html):
            webbrowser.open(f"file:///{html.replace(os.sep, '/')}")
        else:
            messagebox.showwarning("提示", f"未找到 3D 页面:\n{html}")

    def _build_coach_args(self, video: str = None) -> list:
        cmd = [sys.executable, os.path.join(data_path(), "app.py"),
               "--movement", self._selected_movement()]
        if self.no_llm_var.get():
            cmd.append("--no-llm")
        if self.no_opensim_var.get():
            cmd.append("--no-opensim")
        if video:
            cmd += ["--video", video]
        return cmd

    def _start_web(self):
        if self._running:
            messagebox.showinfo("提示", "教练程序已在运行中")
            return
        errors = check_dependencies()
        if errors:
            messagebox.showerror("无法启动", "\n\n".join(errors))
            return
        self._running = True
        self.status.set("Web 控制台启动中…")
        threading.Thread(target=self._run_web_coach, daemon=True).start()

    def _run_web_coach(self):
        try:
            cmd = [sys.executable, os.path.join(data_path(), "web_coach.py"),
                   "-m", self._selected_movement()]
            if self.no_llm_var.get():
                cmd.append("--no-llm")
            proc = subprocess.run(cmd, cwd=data_path(), **subprocess_flags(new_console=True))
            if proc.returncode not in (0, None) and proc.returncode != 0:
                hint = explain_exit_code(proc.returncode)
                self.after(0, lambda h=hint: messagebox.showerror("Web 控制台异常", h))
        except Exception as e:
            self.after(0, lambda err=str(e): messagebox.showerror("错误", err))
        finally:
            self._running = False
            self.after(0, lambda: self.status.set("就绪"))

    def _start_live(self):
        if self._running:
            messagebox.showinfo("提示", "教练程序已在运行中")
            return

        errors = check_dependencies()
        if errors:
            messagebox.showerror("无法启动", "\n\n".join(errors))
            return

        if self.open_3d_var.get():
            self._open_3d_page()

        self._running = True
        self.status.set("正在启动实时教练…（会弹出摄像头窗口，按 q 退出）")
        self._coach_thread = threading.Thread(target=self._run_coach_subprocess, args=(None,), daemon=True)
        self._coach_thread.start()

    def _start_video(self):
        path = filedialog.askopenfilename(
            title="选择训练视频",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")],
        )
        if not path:
            return

        errors = check_dependencies()
        if errors:
            messagebox.showerror("无法启动", "\n\n".join(errors))
            return

        self._running = True
        self.status.set(f"正在分析: {os.path.basename(path)}")
        threading.Thread(target=self._run_coach_subprocess, args=(path,), daemon=True).start()

    def _run_coach_subprocess(self, video: str):
        try:
            cmd = self._build_coach_args(video)
            use_console = video is None
            proc = subprocess.run(
                cmd,
                cwd=data_path(),
                **subprocess_flags(new_console=use_console),
            )
            if proc.returncode != 0:
                hint = explain_exit_code(proc.returncode)
                self.after(0, lambda h=hint: messagebox.showerror("教练程序异常", h))
        except Exception as e:
            self.after(0, lambda err=str(e): messagebox.showerror("错误", err))
        finally:
            self._running = False
            self.after(0, lambda: self.status.set("就绪"))

    def _run_auto_demo(self):
        self.status.set("正在运行自动验收…")
        threading.Thread(target=self._auto_demo_worker, daemon=True).start()

    def _auto_demo_worker(self):
        try:
            script = os.path.join(data_path(), "scripts", "auto_demo.py")
            cmd = [sys.executable, script]
            if self.no_llm_var.get():
                cmd.append("--skip-llm")
            proc = subprocess.run(
                cmd,
                cwd=data_path(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                **subprocess_flags(new_console=False),
            )
            if proc.stdout:
                print(proc.stdout)
            if proc.stderr:
                print(proc.stderr, file=sys.stderr)
            code = proc.returncode
            if code != 0 and proc.stderr:
                err = proc.stderr.strip()
                self.after(0, lambda e=err: messagebox.showerror(
                    "自动验收失败", explain_exit_code(code, e)))

            if code == 0:
                msg = "全部通过"
                self.after(0, lambda: messagebox.showinfo("自动验收", msg))
            elif code != 0:
                self.after(0, lambda: messagebox.showinfo(
                    "自动验收", "部分未通过，请查看终端输出"))
        except Exception as e:
            self.after(0, lambda err=str(e): messagebox.showerror("错误", err))
        finally:
            self.after(0, lambda: self.status.set("就绪"))

    def _show_history(self):
        try:
            from scripts.score_history import plot_history
            plot_history()
        except Exception as e:
            messagebox.showerror("错误", f"无法显示历史图表:\n{e}")


def main():
    app = CoachLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
