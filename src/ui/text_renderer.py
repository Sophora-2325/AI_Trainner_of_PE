"""在 OpenCV 图像上绘制中文/Unicode 文字 (Pillow + 系统字体)."""

import os
from functools import lru_cache

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except ImportError:
    _PIL_OK = False


def _ascii_fallback(text: str) -> str:
    """Pillow 不可用时，尽量保留可读 ASCII."""
    mapping = {
        "深蹲": "Squat", "硬拉": "Deadlift", "俯卧撑": "Pushup",
        "引体向上": "Pullup", "平板支撑": "Plank", "投篮": "Shooting",
        "准备": "Setup", "下降": "Down", "底部": "Bottom",
        "起身": "Up", "锁定": "Lock", "休息": "Rest",
        "综合评分": "Score", "关节角度": "Joint", "对称性": "Sym",
        "稳定性": "Stab", "节奏": "Tempo", "次数": "Reps",
        "未检测到人体": "No person",
    }
    for cn, en in mapping.items():
        text = text.replace(cn, en)
    return text.encode("ascii", errors="replace").decode("ascii")


@lru_cache(maxsize=8)
def _load_font(size: int):
    if not _PIL_OK:
        return None
    candidates = [
        os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", name)
        for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc")
    ]
    candidates = [p for p in candidates if os.path.isfile(p)]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_text(
    frame: np.ndarray,
    text: str,
    xy: tuple[int, int],
    color_bgr: tuple[int, int, int],
    font_size: int = 24,
    thickness: int = 1,
) -> None:
    """在 BGR 图像上绘制文字，xy 为左上角坐标."""
    if not text:
        return

    if not _PIL_OK:
        cv2.putText(frame, _ascii_fallback(text), xy,
                    cv2.FONT_HERSHEY_SIMPLEX, font_size / 32, color_bgr, max(1, thickness))
        return

    font = _load_font(font_size)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    fill = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(xy, text, font=font, fill=fill)
    frame[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def text_size(text: str, font_size: int = 24) -> tuple[int, int]:
    """返回文字宽高 (w, h)."""
    if not _PIL_OK or not text:
        return (len(text) * font_size // 2, font_size)
    font = _load_font(font_size)
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]
