"""UI 叠加层 — 在摄像头画面上叠加评分、阶段、错误信息."""

import cv2
import numpy as np
from typing import Optional
from src.pose.tracker import Phase
from src.comparison.scorer import FrameScore
from src.ui.text_renderer import draw_text


# 颜色定义
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BG = (40, 40, 40)       # 半透明背景色


class OverlayRenderer:
    """在摄像头画面上渲染 UI 叠加层."""

    def __init__(
        self,
        show_skeleton: bool = True,
        show_score: bool = True,
        show_phase: bool = True,
    ):
        self.show_skeleton = show_skeleton
        self.show_score = show_score
        self.show_phase = show_phase

    def render(
        self,
        frame: np.ndarray,
        landmarks_2d: Optional[np.ndarray] = None,
        score: Optional[FrameScore] = None,
        phase: Optional[Phase] = None,
        movement_name: str = "",
        rep_count: int = 0,
        advice: str = "",
    ) -> np.ndarray:
        """在帧上叠加所有 UI 元素."""
        output = frame.copy()
        h, w = output.shape[:2]

        if self.show_skeleton and landmarks_2d is not None:
            self._draw_skeleton(output, landmarks_2d)

        self._draw_top_bar(output, movement_name, rep_count)

        if self.show_score and score is not None:
            self._draw_score_panel(output, score)

        if advice:
            self._draw_advice_bar(output, advice)

        if self.show_phase and phase is not None:
            self._draw_phase_indicator(output, phase)

        return output

    def _draw_skeleton(self, frame: np.ndarray, landmarks: np.ndarray):
        connections = [
            (11, 12), (11, 23), (12, 24), (23, 24),
            (12, 14), (14, 16),
            (11, 13), (13, 15),
            (24, 26), (26, 28), (28, 30), (28, 32),
            (23, 25), (25, 27), (27, 29), (27, 31),
            (12, 0), (11, 0),
        ]

        for i, j in connections:
            pt1 = landmarks[i, :2].astype(int)
            pt2 = landmarks[j, :2].astype(int)
            if np.all(pt1 > 0) and np.all(pt2 > 0):
                cv2.line(frame, tuple(pt1), tuple(pt2), COLOR_GREEN, 2)

        for pt in landmarks[:, :2].astype(int):
            if pt[0] > 0 and pt[1] > 0:
                cv2.circle(frame, tuple(pt), 4, COLOR_WHITE, -1)

    def _draw_top_bar(self, frame: np.ndarray, movement: str, reps: int):
        h, w = frame.shape[:2]
        bar_h = 50

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        if movement:
            draw_text(frame, movement, (15, 8), COLOR_WHITE, font_size=28)

        draw_text(frame, f"次数: {reps}", (w - 130, 10), COLOR_YELLOW, font_size=22)

    def _draw_score_panel(self, frame: np.ndarray, score: FrameScore):
        h, w = frame.shape[:2]
        panel_w = 200
        panel_h = 160
        x = w - panel_w - 15
        y = 65

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        total = score.total
        score_color = _score_color(total)
        draw_text(frame, f"{total:.0f}", (x + 55, y + 18), score_color, font_size=52)
        draw_text(frame, "综合评分", (x + 15, y + 8), COLOR_WHITE, font_size=18)

        dims = [
            ("关节角度", score.joint_deviation),
            ("对称性", score.symmetry),
            ("稳定性", score.stability),
            ("节奏", score.tempo),
        ]

        y0 = y + 78
        for i, (label, val) in enumerate(dims):
            ly = y0 + i * 22
            bar_w = 100
            cv2.rectangle(frame, (x + 70, ly), (x + 70 + bar_w, ly + 12), (60, 60, 60), -1)
            fill_w = int(bar_w * val / 100)
            cv2.rectangle(frame, (x + 70, ly), (x + 70 + fill_w, ly + 12),
                          _score_color(val), -1)
            draw_text(frame, label, (x + 8, ly - 2), COLOR_WHITE, font_size=14)

    def _draw_advice_bar(self, frame: np.ndarray, advice: str):
        h, w = frame.shape[:2]
        bar_h = 45

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        draw_text(frame, advice[:40], (15, h - bar_h + 10), COLOR_YELLOW, font_size=22)

    def _draw_phase_indicator(self, frame: np.ndarray, phase: Phase):
        h, w = frame.shape[:2]
        phase_names = {
            Phase.SETUP: "准备", Phase.DESCENT: "下降",
            Phase.BOTTOM: "底部", Phase.ASCENT: "起身",
            Phase.LOCKOUT: "锁定", Phase.REST: "休息",
        }
        name = phase_names.get(phase, str(phase))
        color = COLOR_GREEN if phase != Phase.REST else COLOR_WHITE
        draw_text(frame, name, (15, h // 2 - 20), color, font_size=32)


def _score_color(score: float) -> tuple:
    if score >= 85:
        return COLOR_GREEN
    elif score >= 60:
        return COLOR_YELLOW
    return COLOR_RED
