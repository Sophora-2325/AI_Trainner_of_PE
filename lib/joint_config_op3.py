"""ROBOTIS OP3 joint/motor configuration constants.

Based on Webots Symmetry.cpp motor ordering and the robot's 20-DOF layout.
"""

# --- Motor names in Webots device order (index 0-19) ---
MOTOR_NAMES = [
    "ShoulderR",   # ID  1 - R_Shoulder_Pitch
    "ShoulderL",   # ID  2 - L_Shoulder_Pitch
    "ArmUpperR",   # ID  3 - R_Shoulder_Roll
    "ArmUpperL",   # ID  4 - L_Shoulder_Roll
    "ArmLowerR",   # ID  5 - R_Elbow
    "ArmLowerL",   # ID  6 - L_Elbow
    "PelvYR",      # ID  7 - R_Hip_Yaw
    "PelvYL",      # ID  8 - L_Hip_Yaw
    "PelvR",       # ID  9 - R_Hip_Roll
    "PelvL",       # ID 10 - L_Hip_Roll
    "LegUpperR",   # ID 11 - R_Hip_Pitch
    "LegUpperL",   # ID 12 - L_Hip_Pitch
    "LegLowerR",   # ID 13 - R_Knee
    "LegLowerL",   # ID 14 - L_Knee
    "AnkleR",      # ID 15 - R_Ankle_Pitch
    "AnkleL",      # ID 16 - L_Ankle_Pitch
    "FootR",       # ID 17 - R_Ankle_Roll
    "FootL",       # ID 18 - L_Ankle_Roll
    "Neck",        # ID 19 - Head_Pan
    "Head",        # ID 20 - Head_Tilt
]

MOTOR_COUNT = len(MOTOR_NAMES)

# Index lookup
MOTOR_INDEX = {name: i for i, name in enumerate(MOTOR_NAMES)}

# Motor IDs (1-indexed as in ROBOTIS OP2 framework)
MOTOR_ID = {name: i + 1 for i, name in enumerate(MOTOR_NAMES)}

# --- Human biomechanical angle → OP3 motor mapping ---
# Each human angle maps to the motor on the same side.
# Left-side motors are negated in map_frame() (symmetry from Symmetry.cpp).
HUMAN_TO_OP3_MAP = {
    "knee_angle_r":   "LegLowerR",
    "knee_angle_l":   "LegLowerL",
    "hip_flexion_r":  "LegUpperR",
    "hip_flexion_l":  "LegUpperL",
    "ankle_angle_r":  "AnkleR",
    "ankle_angle_l":  "AnkleL",
    "elbow_angle_r":  "ArmLowerR",
    "elbow_angle_l":  "ArmLowerL",
    "shoulder_angle_r": "ShoulderR",
    "shoulder_angle_l": "ShoulderL",
    "hip_abduction_r":  "PelvR",
    "hip_abduction_l":  "PelvL",
}

# Motors that are always kept at neutral during imitation
NEUTRAL_MOTORS = [
    "PelvYR", "PelvYL",   # hip yaw — no human analog for gym exercises
    "ArmUpperR", "ArmUpperL",  # shoulder roll — keep neutral for safety
    "Neck", "Head",        # head — keep looking forward
]

# Motors reserved for balance control (set by BalanceController, not retargeting)
BALANCE_MOTORS = ["AnkleR", "AnkleL", "FootR", "FootL"]

# --- Standing offset: added to motor positions so "human standing" → stable OP3 crouch ---
# Without this, f_knee(180°) = 0 rad = fully extended knee (unstable).
# Values in radians, applied per-motor (right positive, left negative for symmetry).
STANDING_OFFSET_RAD = {
    "LegLowerR":  0.30,   # slight knee bend for stability
    "LegLowerL": -0.30,
    "LegUpperR":  0.10,   # slight hip forward lean
    "LegUpperL": -0.10,
    "AnkleR":    -0.10,   # slight ankle dorsiflexion
    "AnkleL":     0.10,
}

# --- ROM limits (approximate, calibrated from MX-28 specs) ---
# These are fallback defaults; actual limits are read from motor.getMin/MaxPosition()
# All values in radians.
DEFAULT_MOTOR_LIMITS_RAD = {
    "ShoulderR":  (-2.0, 2.0),
    "ShoulderL":  (-2.0, 2.0),
    "ArmUpperR":  (-1.5, 1.5),
    "ArmUpperL":  (-1.5, 1.5),
    "ArmLowerR":  (-2.5, 2.5),   # wide symmetric for safety
    "ArmLowerL":  (-2.5, 2.5),
    "PelvYR":     (-1.0, 1.0),
    "PelvYL":     (-1.0, 1.0),
    "PelvR":      (-0.5, 0.5),
    "PelvL":      (-0.5, 0.5),
    "LegUpperR":  (-1.5, 1.5),   # wide symmetric for safety
    "LegUpperL":  (-1.5, 1.5),
    "LegLowerR":  (-2.5, 2.5),   # wide symmetric — left knee uses negative values
    "LegLowerL":  (-2.5, 2.5),
    "AnkleR":     (-1.0, 1.0),
    "AnkleL":     (-1.0, 1.0),
    "FootR":      (-0.5, 0.5),
    "FootL":      (-0.5, 0.5),
    "Neck":       (-1.5, 1.5),
    "Head":       (-0.8, 0.8),
}

# --- Retargeting gains (human degrees → OP3 radians) ---
# human_knee: 180°=straight, ~60°=full bend
# OP3 LegLower: 0 rad = straight, ~2.0 rad = full bend
RETARGET_GAINS = {
    "knee": {
        "human_neutral_deg": 180.0,   # straight leg
        "human_max_bend_deg": 60.0,   # max human knee bend (from movements.yaml)
        "op3_neutral_rad": 0.0,
        "op3_max_bend_rad": 2.0,      # approximate MX-28 range
    },
    "hip": {
        "human_neutral_deg": 180.0,   # upright
        "human_max_flex_deg": 45.0,   # deep squat hip angle (from movements.yaml min=40)
        "op3_neutral_rad": 0.0,
        "op3_max_flex_rad": 1.5,      # forward hip pitch
    },
    "ankle": {
        "human_neutral_deg": 90.0,    # standing ankle
        "human_max_dorsi_deg": 70.0,  # max dorsiflexion
        "op3_neutral_rad": 0.0,
        "op3_max_dorsi_rad": 0.5,
    },
    "elbow": {
        "human_neutral_deg": 180.0,   # straight arm
        "human_max_bend_deg": 45.0,   # max elbow bend (pushup bottom)
        "op3_neutral_rad": 0.0,
        "op3_max_bend_rad": 2.5,
    },
    "shoulder": {
        "human_neutral_deg": 0.0,     # arm at side
        "op3_neutral_rad": 0.0,
        "gain": 0.01,                 # rad per degree (gentle scaling)
    },
    "hip_abduction": {
        "human_neutral_deg": 0.0,
        "op3_neutral_rad": 0.0,
        "gain": 0.005,
    },
}

# Lumbar extension distribution: how to split spine angle across OP3 joints
# 60% to hip pitch, 40% to shoulder pitch
LUMBAR_HIP_RATIO = 0.6
LUMBAR_SHOULDER_RATIO = 0.4
