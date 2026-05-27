"""数据仪表板 — 实时显示动作数据和趋势图."""

import numpy as np
from collections import deque
from typing import Optional
from src.pose.tracker import Phase


class DashboardData:
    """实时仪表板数据缓冲."""

    def __init__(self, history_size: int = 300):
        self.history_size = history_size
        self.scores: deque = deque(maxlen=history_size)
        self.knee_angles_r: deque = deque(maxlen=history_size)
        self.knee_angles_l: deque = deque(maxlen=history_size)
        self.hip_angles_r: deque = deque(maxlen=history_size)
        self.phases: deque = deque(maxlen=history_size)
        self.frame_indices: deque = deque(maxlen=history_size)
        self._frame_count = 0

    def update(
        self,
        score: float,
        joint_angles: dict,
        phase: Optional[Phase] = None,
    ):
        """更新一帧数据."""
        self._frame_count += 1
        self.scores.append(score)
        self.knee_angles_r.append(joint_angles.get("knee_angle_r", np.nan))
        self.knee_angles_l.append(joint_angles.get("knee_angle_l", np.nan))
        self.hip_angles_r.append(joint_angles.get("hip_flexion_r", np.nan))
        self.phases.append(phase)
        self.frame_indices.append(self._frame_count)

    def get_trend_data(self) -> dict:
        """获取当前趋势数据用于绘图."""
        return {
            "scores": list(self.scores),
            "knee_r": list(self.knee_angles_r),
            "knee_l": list(self.knee_angles_l),
            "hip_r": list(self.hip_angles_r),
            "phases": list(self.phases),
            "frames": list(self.frame_indices),
        }

    def reset(self):
        self.scores.clear()
        self.knee_angles_r.clear()
        self.knee_angles_l.clear()
        self.hip_angles_r.clear()
        self.phases.clear()
        self.frame_indices.clear()


class ConsoleDashboard:
    """控制台文本仪表板 — 轻量级实时数据显示."""

    def __init__(self, refresh_interval: int = 30):
        self.refresh_interval = refresh_interval
        self._frame_count = 0

    def update(
        self,
        score: float,
        phase: Optional[Phase],
        errors: list,
        rep_count: int,
    ):
        """刷新控制台显示."""
        self._frame_count += 1
        if self._frame_count % self.refresh_interval != 0:
            return

        phase_str = phase.value if phase else "---"
        bar_len = 30
        filled = int(score / 100 * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        error_str = ", ".join(e.name for e in errors[:2]) if errors else "无"

        print(f"\r[{phase_str:8s}] [{bar}] {score:5.1f}分 | 错误: {error_str:20s} | 次数: {rep_count}", end="")
