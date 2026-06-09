"""标准动作动画播放器 — 在Webots中回放预设动作."""

import os
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class MotionClip:
    """动作片段 — 关节角度关键帧序列."""
    name: str
    joint_names: list
    keyframes: np.ndarray         # (T, num_joints) 关节角度
    frame_rate: int = 30
    loop: bool = False

    @property
    def duration(self) -> float:
        return len(self.keyframes) / self.frame_rate

    def get_at_time(self, t: float) -> dict:
        """获取指定时间的关节角度."""
        frame = int(t * self.frame_rate) % len(self.keyframes)
        return {name: self.keyframes[frame, i]
                for i, name in enumerate(self.joint_names)}


class MotionPlayer:
    """Webots标准动作动画播放器.

    管理多个 MotionClip，按时间轴播放。
    用于在Webots中展示标准动作（并排对比用户实际动作）。
    """

    def __init__(self):
        self._clips: dict[str, MotionClip] = {}
        self._active_clip: Optional[MotionClip] = None
        self._play_time: float = 0.0
        self._playing: bool = False
        self._speed: float = 1.0

    def add_clip(self, clip: MotionClip):
        self._clips[clip.name] = clip

    def play(self, name: str, loop: bool = True):
        """播放指定动作."""
        clip = self._clips.get(name)
        if clip is None:
            print(f"[MotionPlayer] 未找到动作: {name}")
            return
        clip.loop = loop
        self._active_clip = clip
        self._play_time = 0.0
        self._playing = True

    def stop(self):
        self._playing = False
        self._play_time = 0.0

    def pause(self):
        self._playing = False

    def resume(self):
        self._playing = True

    def set_speed(self, speed: float):
        """设置播放速度倍率."""
        self._speed = max(0.1, min(3.0, speed))

    def update(self, dt: float) -> Optional[dict]:
        """更新播放时间，返回当前帧关节角度.

        Args:
            dt: 时间步长 (秒)

        Returns:
            {joint_name: angle} 或 None
        """
        if not self._playing or self._active_clip is None:
            return None

        self._play_time += dt * self._speed

        clip = self._active_clip
        total_duration = clip.duration

        if self._play_time > total_duration:
            if clip.loop:
                self._play_time %= total_duration
            else:
                self._playing = False
                self._play_time = total_duration

        return clip.get_at_time(self._play_time)

    @property
    def progress(self) -> float:
        """当前播放进度 0.0-1.0."""
        if self._active_clip is None or self._active_clip.duration < 1e-9:
            return 0.0
        return min(self._play_time / self._active_clip.duration, 1.0)

    @classmethod
    def from_movement_template(cls, template) -> "MotionPlayer":
        """从 MovementTemplate 创建播放器."""
        player = cls()
        clip = MotionClip(
            name=template.name,
            joint_names=template.joint_names,
            keyframes=template.joint_angle_sequence,
            frame_rate=template.frame_rate,
        )
        player.add_clip(clip)
        return player

    @classmethod
    def from_opensim_mot(cls, mot_path: str, name: str = "") -> "MotionPlayer":
        """从 OpenSim .mot 文件创建播放器。

        解析 OpenSim IK/Forward Dynamics 输出的 .mot 文件，
        提取关节角度时间序列，构建 MotionPlayer。

        Args:
            mot_path: OpenSim .mot 运动文件路径
            name: 动作名称（默认取文件名）

        Returns:
            包含该运动数据的 MotionPlayer
        """
        from collections import OrderedDict

        if not name:
            name = os.path.splitext(os.path.basename(mot_path))[0]

        # Parse .mot file
        with open(mot_path) as f:
            lines = f.readlines()

        # Find data section
        header_start = -1
        data_start = -1
        in_header = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.lower() == "endheader":
                header_start = i + 1
                break

        if header_start < 0:
            raise ValueError(f"Invalid .mot file (no endheader): {mot_path}")

        # Column names line
        headers = lines[header_start].strip().split()
        # Next line is data
        data_lines = []
        for i in range(header_start + 1, len(lines)):
            if lines[i].strip():
                data_lines.append(lines[i].strip().split())

        # Extract time and joint angles
        n_frames = len(data_lines)
        joint_names = [h for h in headers if h.lower() != "time"]
        n_joints = len(joint_names)

        keyframes = np.zeros((n_frames, n_joints))
        for fi, row in enumerate(data_lines):
            col_offset = 0
            for ji, name in enumerate(headers):
                if name.lower() == "time":
                    col_offset += 1
                    continue
                val = float(row[ji])
                # OpenSim uses radians, convert to degrees for consistency
                keyframes[fi, ji - col_offset + 1] = np.degrees(val)

        # Determine frame rate from time column
        frame_rate = 30
        time_idx = next((j for j, h in enumerate(headers) if h.lower() == "time"), -1)
        if time_idx >= 0 and n_frames >= 2:
            t0 = float(data_lines[0][time_idx])
            t1 = float(data_lines[-1][time_idx])
            if t1 > t0:
                frame_rate = int((n_frames - 1) / (t1 - t0))

        clip = MotionClip(
            name=name,
            joint_names=joint_names,
            keyframes=keyframes,
            frame_rate=frame_rate,
            loop=True,
        )
        player = cls()
        player.add_clip(clip)
        return player

    @classmethod
    def from_npy_template(cls, npy_path: str, name: str = "") -> "MotionPlayer":
        """从 .npy 模板文件创建播放器。

        兼容 MovementLibrary 生成的参考模板格式:
          {joint_angles: (T, N), joint_names: [...], fps: 30}

        Args:
            npy_path: .npy 模板文件路径
            name: 动作名称

        Returns:
            MotionPlayer
        """
        if not name:
            name = os.path.splitext(os.path.basename(npy_path))[0]

        data = np.load(npy_path, allow_pickle=True)
        if isinstance(data, np.ndarray):
            # Bare array: assume (T, N) with no joint name info
            keyframes = data
            joint_names = [f"joint_{i}" for i in range(data.shape[1])]
            frame_rate = 30
        else:
            data = data.item()
            joint_angles = data.get("joint_angles", data.get("landmarks"))
            joint_names = list(data.get("joint_names", []))
            frame_rate = data.get("fps", 30)

            if joint_angles.ndim == 3:
                # (T, 33, 4) landmarks format — need to extract angles
                # Use geometric IK to convert
                from src.bridge.socket_server import GeometricIKSolver
                keyframes_list = []
                for t in range(joint_angles.shape[0]):
                    angles = GeometricIKSolver.solve(joint_angles[t])
                    keyframes_list.append(angles)
                joint_names = list(keyframes_list[0].keys())
                keyframes = np.array([
                    [frame[name] for name in joint_names]
                    for frame in keyframes_list
                ])
            else:
                keyframes = joint_angles

        clip = MotionClip(
            name=name,
            joint_names=joint_names,
            keyframes=keyframes.astype(np.float32),
            frame_rate=frame_rate,
            loop=True,
        )
        player = cls()
        player.add_clip(clip)
        return player




def create_squat_reference_motion() -> MotionClip:
    """创建深蹲标准参考动作."""
    joint_names = [
        "hip_flexion_r", "hip_flexion_l",
        "knee_angle_r", "knee_angle_l",
        "ankle_angle_r", "ankle_angle_l",
        "lumbar_extension",
    ]
    total_frames = 90  # 3秒
    t = np.linspace(0, 1, total_frames)

    n_joints = len(joint_names)
    keyframes = np.zeros((total_frames, n_joints))

    for i, name in enumerate(joint_names):
        if "knee" in name:
            keyframes[:, i] = _make_smooth_curve(t, 180, 90, 0.25, 0.5)
        elif "hip" in name:
            keyframes[:, i] = _make_smooth_curve(t, 180, 60, 0.25, 0.5)
        elif "ankle" in name:
            keyframes[:, i] = _make_smooth_curve(t, 90, 70, 0.25, 0.5)
        elif "lumbar" in name:
            keyframes[:, i] = _make_smooth_curve(t, 0, -5, 0.25, 0.5)

    return MotionClip(
        name="squat_reference",
        joint_names=joint_names,
        keyframes=keyframes,
        frame_rate=30,
    )


def _make_smooth_curve(
    t: np.ndarray,
    start: float,
    bottom: float,
    descent_pct: float = 0.25,
    bottom_pct: float = 0.5,
) -> np.ndarray:
    """生成平滑V形曲线."""
    y = np.full_like(t, start)
    for i in range(len(t)):
        ti = t[i]
        if descent_pct < ti <= bottom_pct:
            p = ((ti - descent_pct) / (bottom_pct - descent_pct)) ** 2
            y[i] = start + (bottom - start) * p
        elif bottom_pct < ti <= bottom_pct + 0.07:
            y[i] = bottom
        elif ti > bottom_pct + 0.07:
            p = (ti - bottom_pct - 0.07) / (1.0 - bottom_pct - 0.07)
            p = 1 - (1 - p) ** 2
            y[i] = bottom + (start - bottom) * p
    return y
