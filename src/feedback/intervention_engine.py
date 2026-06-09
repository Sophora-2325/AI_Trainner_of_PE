"""Graded intervention engine for real-time movement correction.

Implements the 3-level intervention strategy from the research proposal (Method 4):
  Level 1 (提示/Hint): angle approaching 80% of safety limit → voice prompt
  Level 2 (警告/Warning): angle exceeding threshold or abnormal angular velocity → voice + visual alert
  Level 3 (紧急/Emergency): sustained severe deviation → voice + OP2 auto-recovery

Safety thresholds are read from config/movements.yaml per-movement ROM ranges.
"""

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


class InterventionLevel(IntEnum):
    NONE = 0
    HINT = 1       # 80% of safety boundary
    WARNING = 2    # beyond threshold or angular velocity anomaly
    EMERGENCY = 3  # sustained severe deviation


@dataclass
class InterventionResult:
    """Result of one evaluation frame."""
    level: InterventionLevel = InterventionLevel.NONE
    joint_name: str = ""
    message: str = ""
    current_value: float = 0.0
    threshold: float = 0.0
    timestamp: float = 0.0


@dataclass
class JointSafetyBoundary:
    """Safety boundary for a single joint."""
    name: str
    min_val: float        # minimum safe angle (degrees)
    max_val: float        # maximum safe angle (degrees)
    hint_ratio: float = 0.80   # hint when reaching 80% of boundary

    def evaluate(self, value: float) -> tuple[InterventionLevel, str]:
        """Check value against boundary, return (level, message)."""
        range_span = self.max_val - self.min_val
        if range_span <= 0:
            return InterventionLevel.NONE, ""

        hint_margin = range_span * (1.0 - self.hint_ratio) / 2.0
        hint_lo = self.min_val + hint_margin
        hint_hi = self.max_val - hint_margin

        if value < self.min_val or value > self.max_val:
            return InterventionLevel.WARNING, f"{self.name}超出安全范围"
        elif value < hint_lo:
            return InterventionLevel.HINT, f"注意{self.name}角度偏小"
        elif value > hint_hi:
            return InterventionLevel.HINT, f"注意{self.name}角度偏大"
        return InterventionLevel.NONE, ""


class InterventionEngine:
    """Graded intervention engine with configurable safety boundaries.

    Usage:
        engine = InterventionEngine(movement="squat", rom_config=...)
        result = engine.evaluate(joint_angles, phase, landmarks)
        if result.level >= InterventionLevel.WARNING:
            tts.speak(result.message)
    """

    # Angular velocity thresholds (degrees per second) — tuned for fitness movements
    DEFAULT_ANGULAR_VELOCITY_WARN = 300.0   # deg/s, e.g., explosive squat ascent
    SUSTAINED_DEVIATION_FRAMES = 15         # consecutive frames for emergency

    def __init__(
        self,
        movement: str = "squat",
        rom_config: dict | None = None,
        angular_velocity_warn: float = 300.0,
    ):
        self.movement = movement
        self.angular_velocity_warn = angular_velocity_warn
        self._boundaries: list[JointSafetyBoundary] = []
        self._prev_angles: dict[str, float] = {}
        self._prev_timestamp: float = 0.0
        self._deviation_counter: dict[str, int] = {}  # joint → consecutive deviation frames

        if rom_config:
            self._load_from_config(rom_config)

    def _load_from_config(self, rom_config: dict):
        """Parse ROM ranges from movements.yaml format."""
        rom = rom_config.get("rom", {})
        name_map = {
            "knee_angle": "膝关节",
            "hip_angle": "髋关节",
            "ankle_angle": "踝关节",
            "lumbar_angle": "腰椎",
            "elbow_angle": "肘关节",
            "shoulder_angle": "肩关节",
            "torso_lean": "躯干前倾",
        }
        for key, ranges in rom.items():
            if isinstance(ranges, dict) and "min" in ranges and "max" in ranges:
                display = name_map.get(key, key)
                self._boundaries.append(JointSafetyBoundary(
                    name=display,
                    min_val=float(ranges["min"]),
                    max_val=float(ranges["max"]),
                ))

    def evaluate(
        self,
        joint_angles: dict[str, float],
        phase,
        landmarks: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> InterventionResult:
        """Evaluate one frame of joint angles.

        Args:
            joint_angles: {angle_name: value_deg} from IK solver
            phase: current movement phase
            landmarks: optional (33, 3) for additional checks
            timestamp: frame timestamp for angular velocity calc

        Returns:
            InterventionResult with highest-priority finding
        """
        ts = timestamp or time.time()
        results: list[InterventionResult] = []

        # 1. Check angular velocity (landing impact / explosive motion)
        if self._prev_timestamp > 0 and self._prev_angles:
            dt = ts - self._prev_timestamp
            if dt > 0:
                for name, val in joint_angles.items():
                    prev = self._prev_angles.get(name)
                    if prev is not None and isinstance(val, (int, float)):
                        vel = abs(val - prev) / dt
                        if vel > self.angular_velocity_warn:
                            results.append(InterventionResult(
                                level=InterventionLevel.WARNING,
                                joint_name=name,
                                message=f"动作过快，请控制速度",
                                current_value=vel,
                                threshold=self.angular_velocity_warn,
                                timestamp=ts,
                            ))

        # 2. Check joint safety boundaries
        rom_key_map = {
            "knee_angle_r": "膝关节", "knee_angle_l": "膝关节",
            "hip_flexion_r": "髋关节", "hip_flexion_l": "髋关节",
            "ankle_angle_r": "踝关节", "ankle_angle_l": "踝关节",
            "lumbar_extension": "腰椎",
            "elbow_angle_r": "肘关节", "elbow_angle_l": "肘关节",
        }

        for joint_name, value in joint_angles.items():
            boundary_key = rom_key_map.get(joint_name)
            if boundary_key is None:
                continue
            for b in self._boundaries:
                if b.name == boundary_key:
                    level, msg = b.evaluate(float(value))
                    if level > InterventionLevel.NONE:
                        results.append(InterventionResult(
                            level=level,
                            joint_name=joint_name,
                            message=msg,
                            current_value=float(value),
                            threshold=b.max_val,
                            timestamp=ts,
                        ))
                    break

        # 3. Check sustained deviation → emergency
        for r in results:
            key = r.joint_name
            count = self._deviation_counter.get(key, 0)
            if r.level >= InterventionLevel.WARNING:
                count += 1
            else:
                count = max(0, count - 1)
            self._deviation_counter[key] = count

            if count >= self.SUSTAINED_DEVIATION_FRAMES:
                r.level = InterventionLevel.EMERGENCY
                r.message = f"严重: {r.message}，请立即停止"

        # Store state for next frame
        self._prev_angles = dict(joint_angles)
        self._prev_timestamp = ts

        # Return highest severity result
        if not results:
            return InterventionResult(timestamp=ts)

        return sorted(results, key=lambda r: r.level, reverse=True)[0]

    def reset(self):
        self._prev_angles.clear()
        self._prev_timestamp = 0.0
        self._deviation_counter.clear()


# ─── Vibration command stub (reserved for hardware integration) ──

def send_vibration_command(intensity: float, duration_ms: int = 200):
    """Send vibration command to wearable node.

    Stub — implement when hardware is connected.
    intensity: 0.0 (off) ~ 1.0 (max)
    """
    # TODO: integrate with IMU node's vibration motor via BLE/UART
    pass
