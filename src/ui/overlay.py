"""UI 叠加层 — 在摄像头画面上叠加评分、阶段、错误信息."""

import cv2
import numpy as np
from typing import Optional
from src.pose.tracker import Phase
from src.comparison.scorer import FrameScore


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
        """在帧上叠加所有 UI 元素.

        Args:
            frame: BGR图像 (H, W, 3)
            landmarks_2d: (33, 3) 2D关键点
            score: 当前帧评分
            phase: 当前动作阶段
            movement_name: 动作名称
            rep_count: 已完成次数
            advice: 当前纠正建议

        Returns:
            叠加后的BGR图像
        """
        output = frame.copy()
        h, w = output.shape[:2]

        # 1. 骨骼叠加
        if self.show_skeleton and landmarks_2d is not None:
            self._draw_skeleton(output, landmarks_2d)

        # 2. 顶部信息栏
        self._draw_top_bar(output, movement_name, rep_count)

        # 3. 右侧评分面板
        if self.show_score and score is not None:
            self._draw_score_panel(output, score)

        # 4. 底部建议条
        if advice:
            self._draw_advice_bar(output, advice)

        # 5. 阶段指示器
        if self.show_phase and phase is not None:
            self._draw_phase_indicator(output, phase)

        return output

    # ─── 骨骼绘制 ────────────────────────────────────────────

    def _draw_skeleton(self, frame: np.ndarray, landmarks: np.ndarray):
        """绘制简化的骨骼连接线."""
        # 关键连接对
        connections = [
            # 躯干
            (11, 12), (11, 23), (12, 24), (23, 24),
            # 右臂
            (12, 14), (14, 16),
            # 左臂
            (11, 13), (13, 15),
            # 右腿
            (24, 26), (26, 28), (28, 30), (28, 32),
            # 左腿
            (23, 25), (25, 27), (27, 29), (27, 31),
            # 头
            (12, 0), (11, 0),
        ]

        for i, j in connections:
            pt1 = landmarks[i, :2].astype(int)
            pt2 = landmarks[j, :2].astype(int)
            if np.all(pt1 > 0) and np.all(pt2 > 0):
                cv2.line(frame, tuple(pt1), tuple(pt2), COLOR_GREEN, 2)

        # 关节点
        for pt in landmarks[:, :2].astype(int):
            if pt[0] > 0 and pt[1] > 0:
                cv2.circle(frame, tuple(pt), 4, COLOR_WHITE, -1)

    # ─── 顶部信息栏 ──────────────────────────────────────────

    def _draw_top_bar(self, frame: np.ndarray, movement: str, reps: int):
        """绘制顶部信息栏."""
        h, w = frame.shape[:2]
        bar_h = 50

        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 动作名称
        if movement:
            cv2.putText(frame, movement, (15, 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 2)

        # 次数
        cv2.putText(frame, f"次数: {reps}", (w - 150, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_YELLOW, 2)

    # ─── 评分面板 ────────────────────────────────────────────

    def _draw_score_panel(self, frame: np.ndarray, score: FrameScore):
        """在右上角绘制评分面板."""
        h, w = frame.shape[:2]
        panel_w = 200
        panel_h = 160
        x = w - panel_w - 15
        y = 65

        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 总分
        total = score.total
        score_color = _score_color(total)
        cv2.putText(frame, f"{total:.0f}", (x + 50, y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, score_color, 4)

        cv2.putText(frame, "综合评分", (x + 15, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)

        # 各维度
        dims = [
            ("关节角度", score.joint_deviation),
            ("对称性", score.symmetry),
            ("稳定性", score.stability),
            ("节奏", score.tempo),
        ]

        y0 = y + 80
        for i, (label, val) in enumerate(dims):
            ly = y0 + i * 22
            # 进度条
            bar_w = 100
            cv2.rectangle(frame, (x + 70, ly), (x + 70 + bar_w, ly + 12), (60, 60, 60), -1)
            fill_w = int(bar_w * val / 100)
            cv2.rectangle(frame, (x + 70, ly), (x + 70 + fill_w, ly + 12),
                          _score_color(val), -1)
            cv2.putText(frame, label, (x + 10, ly + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

    # ─── 底部建议条 ──────────────────────────────────────────

    def _draw_advice_bar(self, frame: np.ndarray, advice: str):
        """在底部绘制建议文字."""
        h, w = frame.shape[:2]
        bar_h = 45

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, advice[:50], (15, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW, 2)

    # ─── 阶段指示器 ──────────────────────────────────────────

    def _draw_phase_indicator(self, frame: np.ndarray, phase: Phase):
        """在左中位置绘制当前阶段指示器."""
        h, w = frame.shape[:2]
        phase_names = {
            Phase.SETUP: "准备", Phase.DESCENT: "下降",
            Phase.BOTTOM: "底部", Phase.ASCENT: "起身",
            Phase.LOCKOUT: "锁定", Phase.REST: "休息",
        }
        name = phase_names.get(phase, str(phase))

        x, y = 15, h // 2
        cv2.putText(frame, name, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    COLOR_GREEN if phase != Phase.REST else COLOR_WHITE, 3)


def _score_color(score: float) -> tuple:
    """评分 → 颜色."""
    if score >= 85:
        return COLOR_GREEN
    elif score >= 60:
        return COLOR_YELLOW
    else:
        return COLOR_RED
