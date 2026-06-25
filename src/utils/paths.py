"""资源路径工具 — 项目根目录与资源路径."""

import os
import sys


def get_app_root() -> str:
    """项目根目录."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resource_path(*parts: str) -> str:
    """读取项目内资源（config、templates、sports_twin.html 等）."""
    return os.path.join(get_app_root(), *parts)


def data_path(*parts: str) -> str:
    """读写用户数据（score_history.json、输出视频等）."""
    return os.path.join(get_app_root(), *parts)


def setup_runtime():
    """启动前调用：切换工作目录并确保项目根在 sys.path 中."""
    root = get_app_root()
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)
