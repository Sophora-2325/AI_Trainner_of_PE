"""Balance controller for ROBOTIS OP3 during imitation playback.

Uses simplified COM projection and ankle strategy to keep the robot
stable during motions like squats, where the COM shifts rearward.

Fall detection mirrors the Walk.cpp accelerometer-based approach.
"""

import math
from typing import Optional

from .joint_config_op3 import MOTOR_NAMES


# ---- Simplified 5-link humanoid body model segment masses (fraction of total) ----
# Based on typical humanoid anthropometry
SEGMENT_MASS = {
    "torso":  0.50,
    "thigh_r": 0.10,
    "thigh_l": 0.10,
    "shank_r": 0.06,
    "shank_l": 0.06,
    "upper_arm_r": 0.03,
    "upper_arm_l": 0.03,
    "forearm_r": 0.02,
    "forearm_l": 0.02,
    "head": 0.08,
}

# Approximate segment lengths (meters) for COM calculation
SEGMENT_LENGTH = {
    "thigh": 0.35,
    "shank": 0.35,
    "torso": 0.40,
    "upper_arm": 0.25,
    "forearm": 0.25,
}

# COM position within each segment as fraction from proximal end
SEGMENT_COM_RATIO = {
    "thigh": 0.43,
    "shank": 0.43,
    "torso": 0.50,
    "upper_arm": 0.44,
    "forearm": 0.43,
}


class BalanceController:
    """Quasi-static balance controller for OP3."""

    def __init__(
        self,
        com_safety_margin: float = 0.02,    # meters inside support polygon
        ankle_gain: float = 0.3,             # rad per meter COM offset
        hip_gain: float = 0.15,              # rad per meter COM offset
    ):
        self.com_safety_margin = com_safety_margin
        self.ankle_gain = ankle_gain
        self.hip_gain = hip_gain

    def process_keyframes(
        self, motor_sequences: dict[str, list[float]], frame_rate: int = 30
    ) -> dict[str, list[float]]:
        """Apply balance corrections to an entire keyframe sequence.

        Args:
            motor_sequences: {motor_name: [rad_per_frame]}
            frame_rate: FPS of the sequence

        Returns:
            Adjusted motor_sequences with ankle/foot/hip corrections
        """
        num_frames = len(next(iter(motor_sequences.values())))
        adjusted = {name: list(seq) for name, seq in motor_sequences.items()}

        for frame_idx in range(num_frames):
            frame = {name: adjusted[name][frame_idx]
                     for name in motor_sequences}
            corrections = self.compute_corrections(frame)
            for motor_name, delta in corrections.items():
                adjusted[motor_name][frame_idx] += delta

        return adjusted

    def compute_corrections(self, motor_positions: dict[str, float]) -> dict[str, float]:
        """Compute balance corrections for one frame.

        Args:
            motor_positions: {motor_name: position_rad} for current frame

        Returns:
            {motor_name: delta_rad} — corrections to ADD
        """
        corrections: dict[str, float] = {name: 0.0 for name in MOTOR_NAMES}

        # Simplified COM forward/backward (x-axis) estimation
        com_x = self._estimate_com_x(motor_positions)

        # Ankle strategy: if COM too far back (x < 0 = behind), add forward ankle pitch
        if com_x < -self.com_safety_margin:
            ankle_correction = (abs(com_x) - self.com_safety_margin) * self.ankle_gain
            corrections["AnkleR"] = ankle_correction
            corrections["AnkleL"] = -ankle_correction  # mirrored
            hip_correction = (abs(com_x) - self.com_safety_margin) * self.hip_gain
            corrections["LegUpperR"] += hip_correction
            corrections["LegUpperL"] -= hip_correction
        elif com_x > self.com_safety_margin:
            # COM too far forward — lean back
            ankle_correction = (com_x - self.com_safety_margin) * self.ankle_gain
            corrections["AnkleR"] = -ankle_correction
            corrections["AnkleL"] = ankle_correction
            hip_correction = (com_x - self.com_safety_margin) * self.hip_gain
            corrections["LegUpperR"] -= hip_correction
            corrections["LegUpperL"] += hip_correction

        return corrections

    def _estimate_com_x(self, motor_positions: dict[str, float]) -> float:
        """Estimate COM x-position (forward/backward) from motor angles.

        Uses a simplified 5-link sagittal-plane model:
          - Foot (base) → shank → thigh → torso
        Positive x = forward of ankle joint.

        Returns:
            com_x in meters (relative to ankle joint center)
        """
        # Get the relevant joint angles
        ankle_r = motor_positions.get("AnkleR", 0.0)
        ankle_l = motor_positions.get("AnkleL", 0.0)
        knee_r = motor_positions.get("LegLowerR", 0.0)
        knee_l = motor_positions.get("LegLowerL", 0.0)
        hip_r = motor_positions.get("LegUpperR", 0.0)
        hip_l = motor_positions.get("LegUpperL", 0.0)

        # Use average of left/right for sagittal plane
        ankle = (ankle_r - ankle_l) / 2.0   # negate left due to symmetry
        knee = (knee_r - knee_l) / 2.0
        hip = (hip_r - hip_l) / 2.0

        # Forward kinematics (sagittal plane, right-handed: x=forward, y=up)
        # Ankle joint at origin (0, 0)
        # Shank: length L_s, angle = ankle (from vertical)
        shank_angle = ankle  # ankle pitch relative to vertical
        shank_x = SEGMENT_LENGTH["shank"] * math.sin(shank_angle)
        shank_com_x = shank_x * SEGMENT_COM_RATIO["shank"]

        # Knee joint
        knee_angle_abs = shank_angle + knee
        knee_x = shank_x + SEGMENT_LENGTH["shank"] * 0.05 * math.sin(knee_angle_abs)

        # Thigh
        thigh_angle = knee_angle_abs + hip
        thigh_x = knee_x + SEGMENT_LENGTH["thigh"] * math.sin(thigh_angle)
        thigh_com_x = knee_x + SEGMENT_LENGTH["thigh"] * SEGMENT_COM_RATIO["thigh"] * math.sin(thigh_angle)

        # Torso (extends upward from hip)
        torso_angle = thigh_angle  # torso continues from hip
        hip_x = knee_x + SEGMENT_LENGTH["thigh"] * math.sin(thigh_angle)
        torso_com_x = hip_x + SEGMENT_LENGTH["torso"] * SEGMENT_COM_RATIO["torso"] * math.sin(torso_angle)

        # Weighted COM
        com_x = (
            shank_com_x * (SEGMENT_MASS["shank_r"] + SEGMENT_MASS["shank_l"]) +
            thigh_com_x * (SEGMENT_MASS["thigh_r"] + SEGMENT_MASS["thigh_l"]) +
            torso_com_x * SEGMENT_MASS["torso"]
        )

        return com_x
