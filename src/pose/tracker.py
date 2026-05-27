"""动作阶段检测器 — 通过关节角度变化率识别当前动作阶段."""

import numpy as np
from enum import Enum
from typing import Optional, Tuple


class Phase(Enum):
    """通用动作阶段."""
    SETUP = "setup"          # 准备姿势
    DESCENT = "descent"      # 离心/下降
    BOTTOM = "bottom"        # 底部/转换点
    ASCENT = "ascent"        # 向心/上升
    LOCKOUT = "lockout"      # 锁定/完成
    REST = "rest"            # 休息


# 各动作的阶段序列
MOVEMENT_PHASES = {
    "squat":    [Phase.SETUP, Phase.DESCENT, Phase.BOTTOM, Phase.ASCENT, Phase.LOCKOUT],
    "deadlift": [Phase.SETUP, Phase.DESCENT, Phase.BOTTOM, Phase.ASCENT, Phase.LOCKOUT],
    "pushup":   [Phase.SETUP, Phase.DESCENT, Phase.BOTTOM, Phase.ASCENT, Phase.LOCKOUT],
    "pullup":   [Phase.SETUP, Phase.DESCENT, Phase.BOTTOM, Phase.ASCENT, Phase.LOCKOUT],
    "plank":    [Phase.SETUP],
}


class MovementPhaseTracker:
    """基于关节角度的动作阶段实时检测器.

    通过监测关键关节角度的变化率来判断当前处于哪个阶段。
    以深蹲为例，主要关注膝关节角度：
    - 膝关节角度减小 → DESCENT
    - 膝关节角度最小 → BOTTOM
    - 膝关节角度增大 → ASCENT
    - 膝关节角度稳定在大值 → LOCKOUT/SETUP
    """

    def __init__(self, movement: str = "squat", hysteresis: float = 5.0):
        """
        Args:
            movement: 动作名称
            hysteresis: 迟滞角度(度)，防止阶段抖动
        """
        self.movement = movement
        self.hysteresis = hysteresis
        self.current_phase = Phase.REST
        self.previous_phase = Phase.REST
        self.phase_duration = 0     # 当前阶段持续帧数
        self.min_phase_frames = 3   # 最小阶段帧数，避免瞬时切换

        # 存储历史角度用于计算变化率
        self._angle_history = []
        self._max_history = 10

        # 深蹲/硬拉用膝角，俯卧撑用肘角
        self._primary_joint = "knee" if movement in ("squat", "deadlift") else "elbow"

    def update(
        self,
        joint_angles: dict,
        landmarks: Optional[np.ndarray] = None,
    ) -> Phase:
        """更新当前动作阶段.

        Args:
            joint_angles: 当前关节角度 dict
            landmarks: (33,4) 关键点（备选方案）

        Returns:
            当前 Phase
        """
        # 尝试从关节角度中提取主要关节角度
        primary_angle = self._extract_primary_angle(joint_angles, landmarks)

        if primary_angle is None:
            return self.current_phase

        self._angle_history.append(primary_angle)
        if len(self._angle_history) > self._max_history:
            self._angle_history.pop(0)

        # 计算角度变化趋势
        new_phase = self._classify_phase(primary_angle)

        # 迟滞 + 最小帧数过滤
        if new_phase != self.current_phase:
            self.phase_duration += 1
            if self.phase_duration >= self.min_phase_frames:
                self.previous_phase = self.current_phase
                self.current_phase = new_phase
                self.phase_duration = 0
        else:
            self.phase_duration = 0

        return self.current_phase

    def _extract_primary_angle(
        self,
        joint_angles: dict,
        landmarks: Optional[np.ndarray],
    ) -> Optional[float]:
        """提取主要监测关节角度."""
        if self._primary_joint == "knee":
            # 优先从关节角度字典取
            r = joint_angles.get("knee_angle_r")
            l = joint_angles.get("knee_angle_l")
            if r is not None and l is not None:
                return (r + l) / 2.0

            # 从关键点计算膝角（备选）
            if landmarks is not None:
                return self._calc_knee_angle(landmarks)

        elif self._primary_joint == "elbow":
            r = joint_angles.get("elbow_angle_r")
            l = joint_angles.get("elbow_angle_l")
            if r is not None and l is not None:
                return (r + l) / 2.0

            if landmarks is not None:
                return self._calc_elbow_angle(landmarks)

        return None

    def _calc_knee_angle(self, landmarks: np.ndarray) -> float:
        """从关键点计算膝关节角度（右侧）."""
        hip = landmarks[24, :3]     # right_hip
        knee = landmarks[26, :3]    # right_knee
        ankle = landmarks[28, :3]   # right_ankle
        return _angle_between(hip - knee, ankle - knee)

    def _calc_elbow_angle(self, landmarks: np.ndarray) -> float:
        """从关键点计算肘关节角度（右侧）."""
        shoulder = landmarks[12, :3]  # right_shoulder
        elbow = landmarks[14, :3]     # right_elbow
        wrist = landmarks[16, :3]     # right_wrist
        return _angle_between(shoulder - elbow, wrist - elbow)

    def _classify_phase(self, angle: float) -> Phase:
        """根据关节角度和历史趋势分类阶段."""
        if len(self._angle_history) < 3:
            return self.current_phase

        trend = angle - self._angle_history[-3]

        if self.movement == "squat":
            return self._classify_squat(angle, trend)
        elif self.movement == "deadlift":
            return self._classify_squat(angle, trend)  # 相似逻辑
        elif self.movement == "pushup":
            return self._classify_pushup(angle, trend)
        else:
            return self._classify_squat(angle, trend)

    def _classify_squat(self, knee_angle: float, trend: float) -> Phase:
        """深蹲/硬拉的阶段分类."""
        # 使用迟滞防止在边界附近抖动
        h = self.hysteresis

        if knee_angle > 150:
            if self.current_phase == Phase.ASCENT and knee_angle > 160:
                return Phase.LOCKOUT
            return Phase.SETUP

        if trend < -h and knee_angle > 90:
            if self.current_phase == Phase.BOTTOM and knee_angle < 95:
                return Phase.BOTTOM
            return Phase.DESCENT

        if knee_angle < 100:
            return Phase.BOTTOM

        if trend > h:
            if knee_angle > 140:
                return Phase.LOCKOUT
            return Phase.ASCENT

        return self.current_phase

    def _classify_pushup(self, elbow_angle: float, trend: float) -> Phase:
        """俯卧撑的阶段分类."""
        h = self.hysteresis

        if elbow_angle > 160:
            return Phase.SETUP if self.current_phase != Phase.DESCENT else Phase.LOCKOUT

        if trend < -h:
            return Phase.DESCENT

        if elbow_angle < 100:
            return Phase.BOTTOM

        if trend > h:
            return Phase.ASCENT

        return self.current_phase

    def reset(self):
        self.current_phase = Phase.REST
        self.previous_phase = Phase.REST
        self.phase_duration = 0
        self._angle_history.clear()


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算两个向量之间的角度（度）."""
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-9:
        return 180.0
    cos = np.clip(dot / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))
