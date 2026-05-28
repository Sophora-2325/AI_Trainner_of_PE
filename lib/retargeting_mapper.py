"""Human biomechanical angle → ROBOTIS OP2 motor position retargeting.

Consumes the output of GeometricIKSolver.solve() — a dict of human joint angles
in degrees — and produces a dict of 20 OP2 motor target positions in radians.

Core reference: Symmetry.cpp shows left_motor = clamp(-right_sensor_value, min, max)
"""

import math
from typing import Optional

from .joint_config_op2 import (
    MOTOR_NAMES,
    HUMAN_TO_OP2_MAP,
    NEUTRAL_MOTORS,
    BALANCE_MOTORS,
    RETARGET_GAINS,
    LUMBAR_HIP_RATIO,
    LUMBAR_SHOULDER_RATIO,
    STANDING_OFFSET_RAD,
)


class HumanAngleToOP2Mapper:
    """Convert human biomechanical joint angles to ROBOTIS OP2 motor positions."""

    def __init__(self, motor_limits: Optional[dict[str, tuple[float, float]]] = None):
        """
        Args:
            motor_limits: {motor_name: (min_rad, max_rad)} from Webots motor devices.
                          If None, uses DEFAULT_MOTOR_LIMITS_RAD from joint_config_op2.
        """
        from .joint_config_op2 import DEFAULT_MOTOR_LIMITS_RAD
        self._limits = motor_limits or DEFAULT_MOTOR_LIMITS_RAD

    # ---- Public API ----

    def map_frame(self, human_angles: dict[str, float]) -> dict[str, float]:
        """Convert a single frame of human angles to OP2 motor positions.

        Args:
            human_angles: dict from GeometricIKSolver.solve(),
                          e.g. {"knee_angle_r": 150.0, "hip_flexion_r": 140.0, ...}

        Returns:
            {motor_name: position_rad} for all 20 motors
        """
        positions: dict[str, float] = {}

        # 1. Initialize all motors to neutral (0 rad)
        for name in MOTOR_NAMES:
            positions[name] = 0.0

        # 2. Map each human angle to its OP2 motor
        for angle_name, motor_name in HUMAN_TO_OP2_MAP.items():
            if angle_name not in human_angles:
                continue
            human_deg = human_angles[angle_name]

            raw_rad = self._convert(angle_name, human_deg)

            # Symmetry: left-side motors get negated values (Symmetry.cpp)
            if angle_name.endswith("_l"):
                raw_rad = -raw_rad

            positions[motor_name] = raw_rad

        # 3. Distribute lumbar extension across hip + shoulder pitch
        if "lumbar_extension" in human_angles:
            lumbar = human_angles["lumbar_extension"]
            hip_offset = math.radians(lumbar * LUMBAR_HIP_RATIO)
            shoulder_offset = math.radians(lumbar * LUMBAR_SHOULDER_RATIO)

            # Apply to hip pitch motors
            positions["LegUpperR"] += hip_offset
            positions["LegUpperL"] -= hip_offset
            # Apply to shoulder pitch motors
            positions["ShoulderR"] += shoulder_offset
            positions["ShoulderL"] -= shoulder_offset

        # 4. Keep neutral motors at zero
        for name in NEUTRAL_MOTORS:
            positions[name] = 0.0

        # 5. Apply standing offset — gives slight knee bend etc. for stability
        for name, offset in STANDING_OFFSET_RAD.items():
            positions[name] = positions.get(name, 0.0) + offset

        # 6. Apply balance motors default neutral (will be overridden by BalanceController)
        for name in BALANCE_MOTORS:
            if name not in HUMAN_TO_OP2_MAP.values():
                positions[name] = 0.0

        # 7. Clamp all positions to motor limits
        positions = self._clamp_all(positions)

        return positions

    def map_sequence(
        self, human_angle_sequence: list[dict[str, float]]
    ) -> dict[str, list[float]]:
        """Convert a sequence of human angle frames to OP2 motor keyframe sequences.

        Args:
            human_angle_sequence: list of per-frame human angle dicts

        Returns:
            {motor_name: [rad_frame0, rad_frame1, ...]} for all 20 motors
        """
        num_frames = len(human_angle_sequence)
        result: dict[str, list[float]] = {name: [0.0] * num_frames for name in MOTOR_NAMES}

        for frame_idx, human_angles in enumerate(human_angle_sequence):
            positions = self.map_frame(human_angles)
            for name in MOTOR_NAMES:
                result[name][frame_idx] = positions[name]

        return result

    # ---- Conversion functions ----

    def _convert(self, angle_name: str, human_deg: float) -> float:
        """Dispatch to the correct conversion function based on angle type."""
        base = angle_name.rstrip("_r").rstrip("_l")

        if base == "knee_angle":
            return self._f_knee(human_deg)
        elif base == "hip_flexion":
            return self._f_hip(human_deg)
        elif base == "ankle_angle":
            return self._f_ankle(human_deg)
        elif base == "elbow_angle":
            return self._f_elbow(human_deg)
        elif base == "shoulder_angle":
            return self._f_shoulder(human_deg)
        elif base == "hip_abduction":
            return self._f_abduction(human_deg)
        else:
            return 0.0

    @staticmethod
    def _f_knee(human_deg: float) -> float:
        """Knee: 180°=straight → 0 rad, 60°=deep bend → ~2.0 rad."""
        g = RETARGET_GAINS["knee"]
        # deviation from straight: 0 when standing, ~120 when deep squatting
        deviation = g["human_neutral_deg"] - human_deg
        deviation = max(0.0, min(120.0, deviation))
        return (deviation / 120.0) * g["op2_max_bend_rad"]

    @staticmethod
    def _f_hip(human_deg: float) -> float:
        """Hip: 180°=upright → 0 rad, 45°=deep forward lean → ~1.5 rad."""
        g = RETARGET_GAINS["hip"]
        deviation = g["human_neutral_deg"] - human_deg
        deviation = max(0.0, min(135.0, deviation))
        return (deviation / 135.0) * g["op2_max_flex_rad"]

    @staticmethod
    def _f_ankle(human_deg: float) -> float:
        """Ankle: 90°=neutral → 0 rad, 70°=dorsiflexion → ~0.5 rad."""
        g = RETARGET_GAINS["ankle"]
        deviation = g["human_neutral_deg"] - human_deg
        deviation = max(-20.0, min(20.0, deviation))
        return (deviation / 20.0) * g["op2_max_dorsi_rad"]

    @staticmethod
    def _f_elbow(human_deg: float) -> float:
        """Elbow: 180°=straight → 0 rad, 45°=full bend → ~2.5 rad."""
        g = RETARGET_GAINS["elbow"]
        deviation = g["human_neutral_deg"] - human_deg
        deviation = max(0.0, min(135.0, deviation))
        return (deviation / 135.0) * g["op2_max_bend_rad"]

    @staticmethod
    def _f_shoulder(human_deg: float) -> float:
        """Shoulder: degree input scaled gently to radians."""
        g = RETARGET_GAINS["shoulder"]
        return g["op2_neutral_rad"] + human_deg * g["gain"]

    @staticmethod
    def _f_abduction(human_deg: float) -> float:
        """Hip abduction: directly scaled to radians."""
        g = RETARGET_GAINS["hip_abduction"]
        return human_deg * g["gain"]

    # ---- Clamping ----

    def _clamp_all(self, positions: dict[str, float]) -> dict[str, float]:
        """Clamp all motor positions to their configured limits."""
        clamped = {}
        for name, value in positions.items():
            lo, hi = self._limits.get(name, (-math.pi, math.pi))
            clamped[name] = max(lo, min(hi, value))
        return clamped
