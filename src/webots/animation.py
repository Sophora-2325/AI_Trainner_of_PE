"""标准动作动画播放器 — 在Webots中回放预设动作."""

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
