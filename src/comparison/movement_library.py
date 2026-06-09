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

        加载顺序:
          1. movement_data/{movement}_reference.npy (关节角度)
          2. templates/template_{movement}.json (33关键点, 第3周格式)
          3. 合成模板 (回退)

        Args:
            movement: 动作名称 (squat, deadlift, pushup, ...)

        Returns:
            MovementTemplate 或 None
        """
        if movement in self._templates:
            return self._templates[movement]

        # 优先 .npy 格式 (关节角度)
        npy_path = os.path.join(self.data_dir, f"{movement}_reference.npy")
        if os.path.exists(npy_path):
            return self._load_npy_template(movement, npy_path)

        # 其次 JSON 格式 (第3周: 33关键点坐标)
        json_path = os.path.join("templates", f"template_{movement}.json")
        if os.path.exists(json_path):
            return self._load_json_template(movement, json_path)

        print(f"[MovementLibrary] 未找到模板文件, 使用合成模板")
        return self._generate_synthetic_template(movement)

    def _load_npy_template(self, movement: str, path: str) -> MovementTemplate:
        """从 .npy 文件加载模板."""
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
        print(f"[MovementLibrary] 已加载模板 (.npy): {path}")
        return template

    def _load_json_template(self, movement: str, path: str) -> MovementTemplate:
        """从第3周 JSON 格式 (33关键点坐标) 加载模板.

        JSON → 关节角度转换使用 GeometricIKSolver.
        """
        import json
        from src.bridge.socket_server import GeometricIKSolver

        with open(path, "r", encoding="utf-8") as f:
            pose_sequence = json.load(f)

        # 将 JSON 关键点转为 (N, 33, 4) 数组
        n_frames = len(pose_sequence)
        landmarks_array = np.zeros((n_frames, 33, 4), dtype=np.float32)

        mp_names = {
            "nose": 0, "left_eye_inner": 1, "left_eye": 2, "left_eye_outer": 3,
            "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6,
            "left_ear": 7, "right_ear": 8, "mouth_left": 9, "mouth_right": 10,
            "left_shoulder": 11, "right_shoulder": 12,
            "left_elbow": 13, "right_elbow": 14,
            "left_wrist": 15, "right_wrist": 16,
            "left_pinky": 17, "right_pinky": 18,
            "left_index": 19, "right_index": 20,
            "left_thumb": 21, "right_thumb": 22,
            "left_hip": 23, "right_hip": 24,
            "left_knee": 25, "right_knee": 26,
            "left_ankle": 27, "right_ankle": 28,
            "left_heel": 29, "right_heel": 30,
            "left_foot_index": 31, "right_foot_index": 32,
        }

        for f, frame in enumerate(pose_sequence):
            for name, idx in mp_names.items():
                landmarks_array[f, idx, 0] = frame.get(f"{name}_x", 0.0)
                landmarks_array[f, idx, 1] = frame.get(f"{name}_y", 0.0)
                landmarks_array[f, idx, 2] = frame.get(f"{name}_z", 0.0)
                landmarks_array[f, idx, 3] = 1.0  # visibility

        # 逐帧求解 IK → 关节角度
        all_angles = []
        for f in range(n_frames):
            angles = GeometricIKSolver.solve(landmarks_array[f])
            all_angles.append(angles)

        # 提取配置中的 key_joints
        cfg = self._config.get(movement, {})
        joint_names = cfg.get("key_joints", list(all_angles[0].keys()))

        # 构建 (T, J) 矩阵
        angles_matrix = np.zeros((n_frames, len(joint_names)))
        for f, angles in enumerate(all_angles):
            for j, name in enumerate(joint_names):
                angles_matrix[f, j] = angles.get(name, 0.0)

        template = MovementTemplate(
            name=movement,
            joint_angle_sequence=angles_matrix,
            joint_names=joint_names,
            phase_boundaries=[
                (0, int(n_frames * 0.2), 0),
                (int(n_frames * 0.2), int(n_frames * 0.5), 1),
                (int(n_frames * 0.5), int(n_frames * 0.55), 2),
                (int(n_frames * 0.55), int(n_frames * 0.85), 3),
                (int(n_frames * 0.85), n_frames - 1, 4),
            ],
            frame_rate=30,
            duration=n_frames / 30.0,
        )
        self._templates[movement] = template
        print(f"[MovementLibrary] 已加载模板 (JSON→IK): {path}, {n_frames}帧")
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
