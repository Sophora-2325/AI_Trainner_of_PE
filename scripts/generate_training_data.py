#!/usr/bin/env python3
"""数字孪生仿真数据生成器 — 参数化生成训练数据集。

对应研究计划书方法二（数字孪生仿真）:
  基于 config/movements.yaml 中定义的动作 ROM 范围，
  参数化生成不同速度、不同幅度、不同错误模式的合成动作序列，
  为方法三（动作识别模型）提供训练数据。

每条序列覆盖"标准动作 → 轻微偏差 → 严重错误"的渐变。
同时生成配套的虚拟IMU数据（accel_xyz + gyro_xyz）。

用法:
  python scripts/generate_training_data.py --movement squat --variations 200
  python scripts/generate_training_data.py --all --variations 100
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bridge.socket_server import GeometricIKSolver
from src.pose.virtual_imu import VirtualIMU


# ─── Configuration ───────────────────────────────────────────────

@dataclass
class VariationParams:
    """参数化变体定义."""
    speed_factor: float = 1.0       # 1.0=正常速度, 0.5=慢速, 2.0=快速
    depth_factor: float = 1.0       # 1.0=标准ROM, 0.7=浅, 1.3=深
    asymmetry: float = 0.0          # 左右不对称程度 (0~1)
    noise_std: float = 0.0          # 关键点高斯噪声标准差
    knee_valgus: float = 0.0        # 膝内扣角度增量 (degrees)
    back_rounding: float = 0.0      # 腰椎屈曲偏移 (degrees)
    heel_lift_offset: float = 0.0   # 脚跟上抬模拟 (影响踝关节)


@dataclass
class GeneratedSequence:
    """一条生成的训练序列."""
    movement: str
    variation_id: int
    params: VariationParams
    landmarks: np.ndarray          # (T, 33, 4) 合成关键点
    joint_angles: np.ndarray       # (T, N) 关节角度
    joint_names: list[str]
    phases: list[tuple]            # (start_frame, end_frame, phase)
    imu_data: np.ndarray           # (T, 6) virtual IMU
    error_labels: list[str]        # 注入的错误标签
    score: float                   # 质量评分 (100=完美)
    fps: int = 30


class TrainingDataGenerator:
    """参数化训练数据生成器。

    基于 movements.yaml 中的 ROM 定义和阶段的关节角度曲线，
    生成带标签的合成训练序列。
    """

    # 关节名称顺序（与 GeometricIKSolver 输出一致）
    JOINT_NAMES = [
        "knee_angle_r", "knee_angle_l",
        "hip_flexion_r", "hip_flexion_l",
        "ankle_angle_r", "ankle_angle_l",
        "elbow_angle_r", "elbow_angle_l",
        "shoulder_angle_r", "shoulder_angle_l",
        "lumbar_extension",
        "neck_yaw", "head_pitch",
        "shoulder_abduction_r", "shoulder_abduction_l",
        "hip_abduction_r", "hip_abduction_l",
    ]

    def __init__(self, config_path: str = "config/movements.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)
        self._virtual_imu = VirtualIMU(frame_rate=30.0)

    def generate(
        self,
        movement: str,
        num_variations: int = 100,
        output_dir: str = "dataset/processed/train",
        base_frames: int = 90,
    ) -> list[str]:
        """生成指定动作的训练数据。

        Args:
            movement: 动作名称 (squat/deadlift/pushup/pullup/plank)
            num_variations: 生成变体数量
            output_dir: 输出目录
            base_frames: 基础帧数

        Returns:
            输出文件路径列表
        """
        movement_cfg = self._config.get(movement)
        if movement_cfg is None:
            print(f"[Gen] 未找到动作配置: {movement}")
            return []

        os.makedirs(output_dir, exist_ok=True)
        output_paths = []

        for vid in range(num_variations):
            params = self._sample_params(vid, num_variations)
            seq = self._generate_sequence(movement, movement_cfg, vid, params, base_frames)

            # Save landmarks + joint angles
            out_path = os.path.join(output_dir, f"{movement}_var{vid:04d}.npy")
            np.save(out_path, {
                "landmarks": seq.landmarks,
                "joint_angles": seq.joint_angles,
                "joint_names": seq.joint_names,
                "phases": seq.phases,
                "imu_data": seq.imu_data,
                "error_labels": seq.error_labels,
                "score": seq.score,
                "label": movement,
                "variation_params": {
                    "speed_factor": params.speed_factor,
                    "depth_factor": params.depth_factor,
                    "asymmetry": params.asymmetry,
                    "noise_std": params.noise_std,
                },
                "fps": seq.fps,
            }, allow_pickle=True)
            output_paths.append(out_path)

        print(f"[Gen] {movement}: 生成 {num_variations} 条 → {output_dir}")
        return output_paths

    def _sample_params(self, vid: int, total: int) -> VariationParams:
        """采样变体参数。前20%为接近完美的标准动作，后80%逐步引入偏差。"""
        rng = np.random.default_rng(vid)

        # Progress from perfect → noisy (0 → 1)
        progress = vid / max(total - 1, 1)

        if progress < 0.2:
            # Clean variations
            return VariationParams(
                speed_factor=rng.uniform(0.85, 1.15),
                depth_factor=rng.uniform(0.90, 1.10),
                asymmetry=0.0,
                noise_std=rng.uniform(0.0, 0.005),
            )
        elif progress < 0.5:
            # Mild deviations
            err_idx = rng.integers(0, 3)
            return VariationParams(
                speed_factor=rng.uniform(0.7, 1.3),
                depth_factor=rng.uniform(0.75, 1.25),
                asymmetry=rng.uniform(0.0, 0.15),
                noise_std=rng.uniform(0.005, 0.015),
                knee_valgus=rng.uniform(0, 8) if err_idx == 0 else 0,
                back_rounding=rng.uniform(0, -8) if err_idx == 1 else 0,
                heel_lift_offset=rng.uniform(0, 0.03) if err_idx == 2 else 0,
            )
        else:
            # Significant deviations
            return VariationParams(
                speed_factor=rng.uniform(0.5, 1.5),
                depth_factor=rng.uniform(0.5, 1.5),
                asymmetry=rng.uniform(0.0, 0.35),
                noise_std=rng.uniform(0.01, 0.03),
                knee_valgus=rng.uniform(0, 20),
                back_rounding=rng.uniform(0, -15),
                heel_lift_offset=rng.uniform(0, 0.05),
            )

    def _generate_sequence(
        self,
        movement: str,
        config: dict,
        vid: int,
        params: VariationParams,
        base_frames: int,
    ) -> GeneratedSequence:
        """生成单条序列。"""
        # Adjust frame count for speed variation
        n_frames = int(base_frames / params.speed_factor)
        n_frames = max(30, min(300, n_frames))

        # Generate base joint angle curves
        joint_angles, phases = self._generate_joint_curves(movement, config, n_frames, params)

        # Convert joint angles back to approximate landmark positions
        # (inverse of GeometricIKSolver — simplified body model)
        landmarks = self._joint_angles_to_landmarks(joint_angles, n_frames, params)

        # Generate virtual IMU data
        imu_data = self._generate_imu(landmarks, n_frames)

        # Determine error labels
        error_labels = []
        if params.knee_valgus > 12:
            error_labels.append("knee_valgus")
        if params.back_rounding < -10:
            error_labels.append("back_rounding")
        if params.heel_lift_offset > 0.03:
            error_labels.append("heel_lift")
        if params.asymmetry > 0.15:
            error_labels.append("asymmetry")

        # Score: 100 minus penalty for each deviation
        score = 100.0
        score -= params.asymmetry * 50
        score -= abs(1.0 - params.depth_factor) * 30
        score -= (params.knee_valgus / 20) * 30
        score -= abs(params.back_rounding / 15) * 30
        score = max(0.0, min(100.0, score))

        joint_names = [n for n in self.JOINT_NAMES if n in joint_angles[0]]

        return GeneratedSequence(
            movement=movement,
            variation_id=vid,
            params=params,
            landmarks=np.array(landmarks, dtype=np.float32),
            joint_angles=np.array([
                [d.get(n, 0.0) for n in joint_names]
                for d in joint_angles
            ], dtype=np.float32),
            joint_names=joint_names,
            phases=phases,
            imu_data=np.array(imu_data, dtype=np.float32),
            error_labels=error_labels,
            score=score,
            fps=30,
        )

    def _generate_joint_curves(
        self, movement: str, config: dict, n_frames: int, params: VariationParams
    ) -> tuple[list[dict], list[tuple]]:
        """Generate joint angle curves with phase boundaries."""
        t = np.linspace(0, 1, n_frames)
        angles_list = []

        rom = config.get("rom", {})
        knee_rom = rom.get("knee_angle", {"min": 60, "max": 180})
        hip_rom = rom.get("hip_angle", {"min": 40, "max": 180})
        ankle_rom = rom.get("ankle_angle", {"min": 15, "max": 45})
        lumbar_rom = rom.get("lumbar_angle", {"min": -10, "max": 15})

        # Base values
        knee_top = knee_rom.get("max", 180)
        knee_bottom = knee_rom.get("ideal_bottom", knee_rom.get("min", 60))
        knee_bottom = knee_top - (knee_top - knee_bottom) * params.depth_factor

        hip_top = hip_rom.get("max", 180)
        hip_bottom = hip_rom.get("min", 40)
        hip_bottom = hip_top - (hip_top - hip_bottom) * params.depth_factor

        ankle_top = ankle_rom.get("max", 45)
        ankle_bottom = ankle_rom.get("min", 15)
        ankle_bottom = ankle_top - (ankle_top - ankle_bottom) * params.depth_factor

        lumbar_top = lumbar_rom.get("max", 15)
        lumbar_bottom = lumbar_rom.get("min", -10)

        # Phase boundaries (fraction of total time)
        phases = [
            (0,             int(n_frames * 0.15), 0),   # SETUP
            (int(n_frames * 0.15), int(n_frames * 0.45), 1),  # DESCENT
            (int(n_frames * 0.45), int(n_frames * 0.55), 2),  # BOTTOM
            (int(n_frames * 0.55), int(n_frames * 0.85), 3),  # ASCENT
            (int(n_frames * 0.85), n_frames - 1,        4),  # LOCKOUT
        ]

        for i in range(n_frames):
            ti = t[i]
            knee_r = _v_curve(ti, knee_top, knee_bottom)
            knee_l = knee_r + params.asymmetry * 15  # asymmetric knee

            hip_r = _v_curve(ti, hip_top, hip_bottom)
            hip_l = hip_r + params.asymmetry * 10

            ankle_r = _v_curve(ti, ankle_top, ankle_bottom)
            ankle_l = ankle_r + params.asymmetry * 5

            lumbar = _v_curve(ti, lumbar_top, lumbar_bottom)
            if params.back_rounding != 0:
                lumbar += params.back_rounding

            angles = {
                "knee_angle_r": knee_r,
                "knee_angle_l": knee_l,
                "hip_flexion_r": hip_r,
                "hip_flexion_l": hip_l,
                "ankle_angle_r": ankle_r,
                "ankle_angle_l": ankle_l,
                "lumbar_extension": lumbar,
                "knee_valgus_angle_r": params.knee_valgus,
                "knee_valgus_angle_l": params.knee_valgus,
                "hip_abduction_r": params.asymmetry * 5,
                "hip_abduction_l": params.asymmetry * -5,
                "elbow_angle_r": 180.0,
                "elbow_angle_l": 180.0,
                "shoulder_angle_r": 0.0,
                "shoulder_angle_l": 0.0,
                "neck_yaw": 0.0,
                "head_pitch": 0.0,
                "shoulder_abduction_r": 0.0,
                "shoulder_abduction_l": 0.0,
            }

            if movement == "pushup":
                # Override for pushup: knee stays fixed, elbow bends
                angles.update({
                    "knee_angle_r": 180.0,
                    "knee_angle_l": 180.0,
                    "hip_flexion_r": 175.0,
                    "hip_flexion_l": 175.0,
                    "elbow_angle_r": _v_curve(ti, 180, 90),
                    "elbow_angle_l": _v_curve(ti, 180, 90),
                })

            angles_list.append(angles)

        return angles_list, phases

    def _joint_angles_to_landmarks(
        self, angles_list: list[dict], n_frames: int, params: VariationParams
    ) -> list[np.ndarray]:
        """逆向生成近似 landmarks（用于 virtual IMU 计算）。

        从关节角度反推 3D 关键点位置。简化的人体运动学链。
        高度简化的模型，仅用于生成合理的 IMU 仿真数据。
        """
        landmarks_list = []
        for _, angles in enumerate(angles_list):
            # Build a simplified skeleton from joint angles
            knee_r = np.radians(angles.get("knee_angle_r", 180))
            knee_l = np.radians(angles.get("knee_angle_l", 180))
            hip_r = np.radians(angles.get("hip_flexion_r", 180))
            hip_l = np.radians(angles.get("hip_flexion_l", 180))
            ankle_r = np.radians(angles.get("ankle_angle_r", 90))
            ankle_l = np.radians(angles.get("ankle_angle_l", 90))

            # Segment lengths (meters, approximate)
            L_thigh = 0.45
            L_shank = 0.42
            L_torso = 0.50

            # Hip positions (world coordinates)
            hip_pos_r = np.array([0.09, 0.92, 0.0])
            hip_pos_l = np.array([-0.09, 0.92, 0.0])

            # Knee positions
            knee_pos_r = hip_pos_r + np.array([
                L_thigh * np.sin(np.pi - hip_r),
                -L_thigh * np.cos(np.pi - hip_r),
                0.05
            ])
            knee_pos_l = hip_pos_l + np.array([
                -L_thigh * np.sin(np.pi - hip_l),
                -L_thigh * np.cos(np.pi - hip_l),
                -0.05
            ])

            # Ankle positions
            ankle_pos_r = knee_pos_r + np.array([
                L_shank * np.sin(knee_r - hip_r + np.pi),
                -L_shank * np.cos(knee_r - hip_r + np.pi),
                0.05
            ])
            ankle_pos_l = knee_pos_l + np.array([
                -L_shank * np.sin(knee_l - hip_l + np.pi),
                -L_shank * np.cos(knee_l - hip_l + np.pi),
                -0.05
            ])

            # Build 33-keypoint array (MediaPipe format, simplified)
            lm = np.zeros((33, 3))
            lm[0] = np.array([0, 1.65, 0.05])     # nose
            lm[11] = np.array([0.12, 1.42, 0.08])  # L shoulder
            lm[12] = np.array([-0.12, 1.42, -0.08]) # R shoulder
            lm[23] = hip_pos_l                      # L hip
            lm[24] = hip_pos_r                      # R hip
            lm[25] = knee_pos_l                     # L knee
            lm[26] = knee_pos_r                     # R knee
            lm[27] = ankle_pos_l                    # L ankle
            lm[28] = ankle_pos_r                    # R ankle
            lm[31] = ankle_pos_l + np.array([0, -0.05, 0.12])  # L foot
            lm[32] = ankle_pos_r + np.array([0, -0.05, -0.12]) # R foot
            # Simple interpolation for elbows, wrists, ears
            lm[13] = lm[11] + np.array([0.05, -0.30, 0.02])
            lm[14] = lm[12] + np.array([-0.05, -0.30, -0.02])
            lm[15] = lm[13] + np.array([0.02, -0.25, 0.01])
            lm[16] = lm[14] + np.array([-0.02, -0.25, -0.01])
            lm[7] = lm[0] + np.array([-0.07, 0.05, 0.0])
            lm[8] = lm[0] + np.array([0.07, 0.05, 0.0])

            # Add noise
            if params.noise_std > 0:
                lm += np.random.randn(*lm.shape) * params.noise_std

            landmarks_list.append(lm.astype(np.float32))

        return landmarks_list

    def _generate_imu(self, landmarks_list: list[np.ndarray], n_frames: int) -> np.ndarray:
        """Generate virtual IMU data from landmark sequence."""
        imu_frames = []
        self._virtual_imu.reset()
        for t, lm in enumerate(landmarks_list):
            samples = self._virtual_imu.landmarks_to_imu(lm, t / 30.0)
            if samples:
                s = samples[0]
                imu_frames.append([s.accel_x, s.accel_y, s.accel_z,
                                   s.gyro_x, s.gyro_y, s.gyro_z])
            else:
                imu_frames.append([0.0] * 6)
        return np.array(imu_frames, dtype=np.float32)


def _v_curve(t: float, top: float, bottom: float,
             descent_pct: float = 0.15, bottom_pct: float = 0.50) -> float:
    """平滑V形曲线（一次动作周期）."""
    if t <= descent_pct:
        return float(top)
    elif descent_pct < t <= bottom_pct:
        p = (t - descent_pct) / (bottom_pct - descent_pct)
        p = p ** 2
        return float(top + (bottom - top) * p)
    elif bottom_pct < t <= bottom_pct + 0.07:
        return float(bottom)
    else:
        p = (t - bottom_pct - 0.07) / (1.0 - bottom_pct - 0.07)
        p = 1.0 - (1.0 - p) ** 2
        return float(bottom + (top - bottom) * p)


# ─── CLI ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="数字孪生仿真数据生成器"
    )
    parser.add_argument("--movement", "-m", type=str, default="squat",
                        choices=["squat", "deadlift", "pushup", "pullup", "plank"],
                        help="动作名称")
    parser.add_argument("--all", action="store_true",
                        help="为所有动作生成数据")
    parser.add_argument("--variations", "-n", type=int, default=100,
                        help="每个动作的变体数量")
    parser.add_argument("--output-dir", "-o", type=str,
                        default="dataset/processed/train",
                        help="输出目录")
    parser.add_argument("--config", "-c", type=str,
                        default="config/movements.yaml",
                        help="动作配置文件路径")
    args = parser.parse_args()

    generator = TrainingDataGenerator(config_path=args.config)

    movements = ["squat", "deadlift", "pushup", "pullup", "plank"] if args.all else [args.movement]

    total = 0
    for mv in movements:
        paths = generator.generate(
            movement=mv,
            num_variations=args.variations,
            output_dir=os.path.join(args.output_dir, mv),
        )
        total += len(paths)

    print(f"\n总计生成 {total} 条训练序列 → {args.output_dir}")
    print("可用于模型训练: python -m src.training.train_error_model --data dataset/processed/train/")
