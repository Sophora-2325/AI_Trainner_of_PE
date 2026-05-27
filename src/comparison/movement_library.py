"""标准动作模板库 — 加载与管理标准动作的关节角度时间序列."""

import os
import numpy as np
import yaml
from dataclasses import dataclass, field
from typing import Optional
from src.pose.tracker import Phase


@dataclass
class MovementTemplate:
    """单个标准动作模板."""
    name: str
    joint_angle_sequence: np.ndarray    # (T, num_joints) 关节角度时间序列
    joint_names: list                   # 关节名称列表
    phase_boundaries: list              # [(start_frame, end_frame, Phase)]
    frame_rate: int = 30
    duration: float = 0.0              # 动作总时长(秒)

    @property
    def num_frames(self) -> int:
        return len(self.joint_angle_sequence)

    def get_frame_at(self, idx: int) -> np.ndarray:
        """获取指定帧的关节角度向量."""
        if idx < 0 or idx >= self.num_frames:
            idx = np.clip(idx, 0, self.num_frames - 1)
        return self.joint_angle_sequence[idx]


class MovementLibrary:
    """标准动作模板库管理器.

    模板来源优先级:
    1. 专业运动员动捕数据 (.npy)
    2. OpenSim优化模拟数据 (.mot → .npy)
    3. MediaPipe提取的教练视频数据 (.npy)
    """

    def __init__(self, data_dir: str = "movement_data"):
        self.data_dir = data_dir
        self._templates: dict[str, MovementTemplate] = {}
        self._config: dict = {}

    def load_config(self, config_path: str = "config/movements.yaml"):
        """加载动作配置文件."""
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    def load_template(self, movement: str) -> Optional[MovementTemplate]:
        """加载一个动作的标准模板.

        Args:
            movement: 动作名称 (squat, deadlift, pushup, ...)

        Returns:
            MovementTemplate 或 None
        """
        if movement in self._templates:
            return self._templates[movement]

        path = os.path.join(self.data_dir, f"{movement}_reference.npy")
        if not os.path.exists(path):
            print(f"[MovementLibrary] 未找到模板文件: {path}")
            return self._generate_synthetic_template(movement)

        # .npy 格式: 包含 joint_angles 和 joint_names
        data = np.load(path, allow_pickle=True).item()
        template = MovementTemplate(
            name=movement,
            joint_angle_sequence=data["joint_angles"],
            joint_names=data.get("joint_names", []),
            phase_boundaries=data.get("phases", []),
            frame_rate=data.get("fps", 30),
            duration=data["joint_angles"].shape[0] / data.get("fps", 30),
        )
        self._templates[movement] = template
        return template

    def _generate_synthetic_template(self, movement: str) -> Optional[MovementTemplate]:
        """当没有真实模板时，根据关节阈值生成合成模板."""
        if movement not in self._config:
            return None

        cfg = self._config[movement]
        rom = cfg.get("rom", {})

        # 为关键关节生成理想角度序列
        # 简化的阶段模型:
        # SETUP → DESCENT → BOTTOM → ASCENT → LOCKOUT
        total_frames = 90  # 3秒 @ 30fps
        joint_names = cfg.get("key_joints", [])

        if not joint_names:
            return None

        angles = np.zeros((total_frames, len(joint_names)))
        frame_rate = 30

        # 生成示范轨迹（平滑曲线）
        for i, name in enumerate(joint_names):
            angles[:, i] = self._make_trajectory(name, rom, total_frames)

        template = MovementTemplate(
            name=movement,
            joint_angle_sequence=angles,
            joint_names=joint_names,
            phase_boundaries=[
                (0, 18, Phase.SETUP),
                (18, 42, Phase.DESCENT),
                (42, 48, Phase.BOTTOM),
                (48, 72, Phase.ASCENT),
                (72, 90, Phase.LOCKOUT),
            ],
            frame_rate=frame_rate,
            duration=total_frames / frame_rate,
        )
        self._templates[movement] = template
        return template

    @staticmethod
    def _make_trajectory(name: str, rom: dict, length: int) -> np.ndarray:
        """生成单关节的理想角度轨迹."""
        t = np.linspace(0, 1, length)

        if "knee" in name:
            # 膝角: 180(直) → 90(底) → 180(直)
            min_angle = rom.get("knee_angle", {}).get("ideal_bottom", 95)
            return _smooth_valley(t, 180.0, min_angle, 0.2, 0.55)

        elif "hip" in name:
            # 髋角: 180(直) → 60(底) → 180(直)
            min_angle = rom.get("hip_angle", {}).get("min", 45)
            return _smooth_valley(t, 180.0, min_angle, 0.2, 0.55)

        elif "ankle" in name:
            # 踝角: 90(中立) → 70(背屈) → 90(中立)
            min_angle = rom.get("ankle_angle", {}).get("min", 70)
            return _smooth_valley(t, 90.0, min_angle, 0.2, 0.55)

        elif "lumbar" in name:
            # 脊柱: 0(中立) → 略前倾 → 0(中立)
            return _smooth_valley(t, 0.0, -5.0, 0.2, 0.55)

        elif "elbow" in name:
            min_angle = rom.get("elbow_angle", {}).get("ideal_bottom", 90)
            return _smooth_valley(t, 180.0, min_angle, 0.2, 0.55)

        elif "shoulder" in name:
            return _smooth_valley(t, 0.0, -40.0, 0.2, 0.55)

        elif "abduction" in name:
            return np.full(length, 0.0)  # 理想情况无外展

        else:
            return np.full(length, 0.0)  # 默认值

    def list_movements(self) -> list:
        """列出所有可用动作."""
        return list(self._config.keys())

    def get_joint_thresholds(self, movement: str) -> dict:
        """获取指定动作的关节阈值."""
        cfg = self._config.get(movement, {})
        return cfg.get("rom", {})

    def get_error_rules(self, movement: str) -> list:
        """获取指定动作的错误检测规则."""
        cfg = self._config.get(movement, {})
        return cfg.get("errors", [])


def _smooth_valley(
    t: np.ndarray,
    start_val: float,
    bottom_val: float,
    descent_pct: float = 0.25,
    bottom_pct: float = 0.5,
) -> np.ndarray:
    """生成平滑的V形山谷曲线，模拟一个完整的动作周期.

    Args:
        t: 归一化时间 [0, 1]
        start_val: 起始值（直立）
        bottom_val: 底部值
        descent_pct: 下降阶段占比
        bottom_pct: 底部位置

    Returns:
        角度数组
    """
    y = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < descent_pct:
            # 准备阶段，保持起始值
            y[i] = start_val
        elif ti < bottom_pct:
            # 下降阶段 — 平滑过渡到最低点
            progress = (ti - descent_pct) / (bottom_pct - descent_pct)
            progress = progress ** 2  # ease-in
            y[i] = start_val + (bottom_val - start_val) * progress
        elif ti < bottom_pct + 0.07:
            # 底部保持
            y[i] = bottom_val
        else:
            # 上升阶段 — 平滑回到起始值
            progress = (ti - bottom_pct - 0.07) / (1.0 - bottom_pct - 0.07)
            progress = 1 - (1 - progress) ** 2  # ease-out
            y[i] = bottom_val + (start_val - bottom_val) * progress

    return y
