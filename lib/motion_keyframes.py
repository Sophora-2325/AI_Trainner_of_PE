"""Motion keyframes — motor position sequences for OP2 playback.

Stores per-motor position arrays and supports JSON serialization,
interpolation between sparse keyframes, and time scaling.
"""

import json
import math
from dataclasses import dataclass, field

from .joint_config_op2 import MOTOR_NAMES


@dataclass
class MotionKeyframes:
    """A sequence of motor positions for all 20 OP2 motors.

    Args:
        name: Motion name (e.g. "squat", "deadlift")
        motor_positions: {motor_name: [rad_frame0, rad_frame1, ...]}
        frame_rate: Frames per second of the original data
        loop: Whether to loop during playback
    """
    name: str
    motor_positions: dict[str, list[float]] = field(default_factory=dict)
    frame_rate: int = 30
    loop: bool = False

    def __post_init__(self):
        if not self.motor_positions:
            self.motor_positions = {name: [] for name in MOTOR_NAMES}

    # ---- Properties ----

    @property
    def num_frames(self) -> int:
        if not self.motor_positions:
            return 0
        return len(next(iter(self.motor_positions.values())))

    @property
    def duration(self) -> float:
        if self.frame_rate <= 0:
            return 0.0
        return self.num_frames / self.frame_rate

    @property
    def motor_names(self) -> list[str]:
        return list(self.motor_positions.keys())

    # ---- Frame access ----

    def get_frame(self, idx: int) -> dict[str, float]:
        """Get a single frame's motor positions.

        Args:
            idx: Frame index (clamped to valid range)

        Returns:
            {motor_name: position_rad}
        """
        idx = max(0, min(idx, self.num_frames - 1)) if self.num_frames > 0 else 0
        return {name: seq[idx] for name, seq in self.motor_positions.items()}

    def get_at_time(self, t: float) -> dict[str, float]:
        """Get interpolated motor positions at a given time.

        Args:
            t: Time in seconds. Loops if loop=True.

        Returns:
            {motor_name: interpolated_position_rad}
        """
        if self.num_frames < 2:
            return self.get_frame(0)

        total = self.duration
        if total <= 0:
            return self.get_frame(0)

        if self.loop:
            t = t % total

        # Clamp to [0, total]
        t = max(0.0, min(t, total))

        # Frame index as float
        frame_float = t * self.frame_rate
        idx_lo = int(math.floor(frame_float))
        idx_hi = idx_lo + 1
        frac = frame_float - idx_lo

        idx_lo = max(0, min(idx_lo, self.num_frames - 1))
        idx_hi = max(0, min(idx_hi, self.num_frames - 1))

        result = {}
        for name in self.motor_names:
            seq = self.motor_positions[name]
            if idx_lo >= len(seq) or idx_hi >= len(seq):
                result[name] = 0.0
            else:
                result[name] = seq[idx_lo] + (seq[idx_hi] - seq[idx_lo]) * frac

        return result

    # ---- Time scaling ----

    def time_scale(self, factor: float) -> "MotionKeyframes":
        """Create a time-stretched copy by resampling.

        Args:
            factor: >1.0 = slower playback (more frames), <1.0 = faster.

        Returns:
            New MotionKeyframes with resampled sequence
        """
        if factor <= 0:
            raise ValueError("Time scale factor must be positive")

        new_num_frames = int(self.num_frames * factor)
        if new_num_frames < 2:
            return self  # can't stretch a single frame

        new_positions = {}
        for name in self.motor_names:
            old_seq = self.motor_positions[name]
            new_seq = []
            for i in range(new_num_frames):
                old_idx = i / factor
                idx_lo = int(math.floor(old_idx))
                idx_hi = min(idx_lo + 1, len(old_seq) - 1)
                frac = old_idx - idx_lo
                val = old_seq[idx_lo] + (old_seq[idx_hi] - old_seq[idx_lo]) * frac
                new_seq.append(val)
            new_positions[name] = new_seq

        return MotionKeyframes(
            name=self.name,
            motor_positions=new_positions,
            frame_rate=self.frame_rate,
            loop=self.loop,
        )

    # ---- Serialization ----

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "motor_positions": self.motor_positions,
            "frame_rate": self.frame_rate,
            "loop": self.loop,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MotionKeyframes":
        return cls(
            name=data["name"],
            motor_positions=data.get("motor_positions", {}),
            frame_rate=data.get("frame_rate", 30),
            loop=data.get("loop", False),
        )

    def save(self, path: str):
        """Save to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MotionKeyframes":
        """Load from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_motor_sequences(
        cls,
        name: str,
        sequences: dict[str, list[float]],
        frame_rate: int = 30,
        loop: bool = False,
    ) -> "MotionKeyframes":
        """Create from {motor_name: [rad_seq]} dict."""
        return cls(
            name=name,
            motor_positions=sequences,
            frame_rate=frame_rate,
            loop=loop,
        )
