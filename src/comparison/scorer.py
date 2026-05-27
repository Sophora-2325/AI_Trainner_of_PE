"""多维度动作质量评分引擎."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from collections import deque
from src.pose.tracker import Phase


@dataclass
class FrameScore:
    """单帧评分."""
    total: float = 100.0
    joint_deviation: float = 100.0   # 关节角度偏差分
    symmetry: float = 100.0          # 对称性分
    stability: float = 100.0         # 稳定性分
    tempo: float = 100.0             # 节奏分
    phase: Optional[Phase] = None
    errors_detected: list = field(default_factory=list)


@dataclass
class RepScore:
    """单次完整动作评分."""
    movement: str = ""
    total: float = 0.0               # 综合评分 0-100
    joint_deviation: float = 0.0
    symmetry: float = 0.0
    stability: float = 0.0
    tempo: float = 0.0
    frame_scores: list = field(default_factory=list)
    error_summary: dict = field(default_factory=dict)  # {error_id: count}
    duration_seconds: float = 0.0
    rep_count: int = 0


class MovementScorer:
    """多维度动作质量实时评分器.

    评分维度:
    - 关节角度偏差 (50%): 用户关节角度与标准模板的逐帧偏差
    - 对称性 (20%): 左右侧关节角度差异
    - 稳定性 (15%): 关键点抖动程度
    - 节奏 (15%): 动作速度与标准模板的匹配度
    """

    def __init__(
        self,
        joint_deviation_weight: float = 0.50,
        symmetry_weight: float = 0.20,
        stability_weight: float = 0.15,
        tempo_weight: float = 0.15,
    ):
        self.weights = {
            "joint_deviation": joint_deviation_weight,
            "symmetry": symmetry_weight,
            "stability": stability_weight,
            "tempo": tempo_weight,
        }

        # 稳定性计算所需的历史数据
        self._landmark_history = deque(maxlen=30)
        self._angle_history = deque(maxlen=30)

        # 评分历史（用于整次动作汇总）
        self._frame_scores: list[FrameScore] = []
        self._error_counts: dict = {}

    def score_frame(
        self,
        user_angles: dict,
        template_angles: np.ndarray,
        joint_names: list,
        landmarks: Optional[np.ndarray] = None,
        phase: Optional[Phase] = None,
        errors: Optional[list] = None,
    ) -> FrameScore:
        """对单帧进行评分.

        Args:
            user_angles: 用户当前帧关节角度 dict
            template_angles: 模板当前帧关节角度 (num_joints,)
            joint_names: 关节名称列表
            landmarks: (33, 4) 原始关键点（用于稳定性计算）
            phase: 当前动作阶段
            errors: 检测到的错误列表

        Returns:
            FrameScore
        """
        score = FrameScore(phase=phase)

        # 1. 关节角度偏差评分
        score.joint_deviation = self._score_joint_deviation(
            user_angles, template_angles, joint_names
        )

        # 2. 对称性评分
        score.symmetry = self._score_symmetry(user_angles)

        # 3. 稳定性评分
        if landmarks is not None:
            self._landmark_history.append(landmarks)
            score.stability = self._score_stability()
        else:
            score.stability = 100.0

        # 4. 节奏评分
        if len(self._angle_history) > 1:
            score.tempo = self._score_tempo(user_angles)
        self._angle_history.append(user_angles)

        # 5. 综合评分
        score.total = (
            self.weights["joint_deviation"] * score.joint_deviation +
            self.weights["symmetry"] * score.symmetry +
            self.weights["stability"] * score.stability +
            self.weights["tempo"] * score.tempo
        )

        # 6. 错误扣分
        if errors:
            score.errors_detected = errors
            for e in errors:
                severity_penalty = {"high": 15, "medium": 8, "low": 3}
                penalty = severity_penalty.get(e.severity, 5)
                score.total = max(0, score.total - penalty)
                self._error_counts[e.id] = self._error_counts.get(e.id, 0) + 1

        self._frame_scores.append(score)
        return score

    def score_rep(self, movement: str = "") -> RepScore:
        """汇总一次完整动作的评分.

        Returns:
            RepScore 包含各维度平均分和错误汇总
        """
        if not self._frame_scores:
            return RepScore(movement=movement)

        n = len(self._frame_scores)
        rep = RepScore(
            movement=movement,
            total=float(np.mean([s.total for s in self._frame_scores])),
            joint_deviation=float(np.mean([s.joint_deviation for s in self._frame_scores])),
            symmetry=float(np.mean([s.symmetry for s in self._frame_scores])),
            stability=float(np.mean([s.stability for s in self._frame_scores])),
            tempo=float(np.mean([s.tempo for s in self._frame_scores])),
            frame_scores=self._frame_scores.copy(),
            error_summary=dict(self._error_counts),
            duration_seconds=n / 30.0,
        )
        return rep

    def reset(self):
        """开始新一次动作时重置."""
        self._frame_scores.clear()
        self._error_counts.clear()
        self._landmark_history.clear()
        self._angle_history.clear()

    # ─── 各维度评分实现 ──────────────────────────────────────

    def _score_joint_deviation(
        self,
        user_angles: dict,
        template_angles: np.ndarray,
        joint_names: list,
    ) -> float:
        """关节角度偏差评分.

        使用指数衰减：偏离越大，扣分越重。
        """
        total_error = 0.0
        count = 0

        for i, name in enumerate(joint_names):
            if name not in user_angles or i >= len(template_angles):
                continue
            user_val = user_angles[name]
            template_val = template_angles[i]
            error = abs(user_val - template_val)
            total_error += error
            count += 1

        if count == 0:
            return 100.0

        avg_error = total_error / count
        # 指数衰减: 0°偏差→100分, 5°→90, 15°→74, 30°→55
        return float(100.0 * np.exp(-avg_error / 20.0))

    def _score_symmetry(self, angles: dict) -> float:
        """左右对称性评分."""
        pairs = [
            ("knee_angle_r", "knee_angle_l"),
            ("hip_flexion_r", "hip_flexion_l"),
            ("elbow_angle_r", "elbow_angle_l"),
            ("shoulder_angle_r", "shoulder_angle_l"),
            ("ankle_angle_r", "ankle_angle_l"),
        ]

        total_diff = 0.0
        count = 0
        for r_key, l_key in pairs:
            if r_key in angles and l_key in angles:
                total_diff += abs(angles[r_key] - angles[l_key])
                count += 1

        if count == 0:
            return 100.0

        avg_diff = total_diff / count
        # 差异 <3° 满分，>20° 0分
        return float(np.clip(100.0 - avg_diff * 5.0, 0, 100))

    def _score_stability(self) -> float:
        """关键点稳定性评分（基于抖动程度）."""
        if len(self._landmark_history) < 5:
            return 100.0

        # 计算最近N帧关键点的标准差
        recent = list(self._landmark_history)[-10:]
        stacked = np.stack([lm[:, :3] for lm in recent])  # (N, 33, 3)

        # 只关注主要关节（排除面部和手指）
        main_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        joint_std = np.std(stacked[:, main_joints, :], axis=0)  # (12, 3)
        avg_std = np.mean(joint_std)

        # 抖动越小越好: std=0→100, std=0.05→60
        return float(np.clip(100.0 - avg_std * 800, 0, 100))

    def _score_tempo(self, angles: dict) -> float:
        """节奏评分（角度变化率合理性）."""
        if len(self._angle_history) < 2:
            return 100.0

        prev = self._angle_history[-1]
        curr = angles

        # 检测是否有关节角度突变（可能表示动作过快或失控）
        max_change = 0.0
        for key in curr:
            if key in prev:
                change = abs(curr[key] - prev[key])
                max_change = max(max_change, change)

        # 单帧角度变化 > 15° 表示动作过快
        if max_change < 3:
            return 100.0
        elif max_change < 8:
            return 85.0
        elif max_change < 15:
            return 60.0
        else:
            return float(np.clip(100.0 - (max_change - 3) * 4.0, 10, 100))
