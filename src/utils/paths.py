"""资源路径工具 — 兼容开发环境与 PyInstaller 打包后的 exe."""

import os
import sys


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def get_app_root() -> str:
    """可写数据目录（exe 所在目录 / 项目根目录）."""
    if is_frozen():
        return os.path.dirname(sys.executable)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_bundle_root() -> str:
    """只读资源目录（打包后为 _MEIPASS）."""
    if is_frozen():
        return sys._MEIPASS
    return get_app_root()


def resource_path(*parts: str) -> str:
    """读取 bundled 资源（config、templates、sports_twin.html 等）."""
    return os.path.join(get_bundle_root(), *parts)


def data_path(*parts: str) -> str:
    """读写用户数据（score_history.json、输出视频等）."""
    return os.path.join(get_app_root(), *parts)


def setup_runtime():
    """启动前调用：切换工作目录并确保项目根在 sys.path 中."""
    root = get_app_root()
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)
