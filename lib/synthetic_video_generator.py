"""Generate synthetic training videos for pipeline testing.

Draws an animated stick figure performing exercises (squat, pushup, etc.)
using OpenCV. This removes the dependency on real training videos for
development and testing of the retargeting pipeline.
"""

import os
import math
import numpy as np


def generate_squat_video(
    output_path: str = "test_squat.mp4",
    duration: float = 3.0,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
):
    """Generate a synthetic squat video with a stick figure.

    The stick figure joint angles follow a realistic squat trajectory:
    - Start standing upright
    - Descend into a deep squat
    - Hold at the bottom
    - Ascend back to standing

    Args:
        output_path: Output .mp4 file path
        duration: Video duration in seconds
        fps: Frames per second
        width: Video width
        height: Video height
    """
    import cv2

    total_frames = int(duration * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Body segment proportions (pixels at default scale)
    segment_lengths = {
        "head_to_shoulder": 40,
        "shoulder_to_hip": 100,
        "hip_to_knee": 100,
        "knee_to_ankle": 100,
        "shoulder_to_elbow": 70,
        "elbow_to_wrist": 70,
        "shoulder_width": 60,
        "hip_width": 50,
    }

    # Root position (hip center) — fixed on screen
    root_x = width // 2
    root_y = height // 2 - 20

    print(f"Generating synthetic squat video: {output_path}")
    print(f"  {total_frames} frames @ {fps} FPS, {duration}s")

    for frame_idx in range(total_frames):
        t = frame_idx / total_frames  # normalized time [0, 1]

        # Compute joint angles for this frame
        knee_angle = _squat_knee_angle(t)
        hip_angle = _squat_hip_angle(t)
        ankle_angle = _squat_ankle_angle(t)
        shoulder_angle = _squat_shoulder_angle(t)
        elbow_angle = _squat_elbow_angle(t)

        # Build stick figure landmarks
        # Using simplified 2D forward kinematics from hip center
        landmarks = _build_stick_figure_2d(
            root_x, root_y,
            knee_angle, hip_angle, ankle_angle,
            shoulder_angle, elbow_angle,
            segment_lengths,
        )

        # Draw on blank canvas
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 240  # light gray

        # Draw bones
        bone_pairs = [
            (0, 1),   # head → neck
            (1, 2),   # neck → hip
            (2, 3), (2, 4),   # hip → left/right hip
            (3, 5), (4, 6),   # hip → knee
            (5, 7), (6, 8),   # knee → ankle
            (1, 9), (1, 10),  # neck → shoulders
            (9, 11), (10, 12),  # shoulder → elbow
            (11, 13), (12, 14),  # elbow → wrist
        ]
        for i, j in bone_pairs:
            if i < len(landmarks) and j < len(landmarks):
                pt1 = (int(landmarks[i][0]), int(landmarks[i][1]))
                pt2 = (int(landmarks[j][0]), int(landmarks[j][1]))
                cv2.line(canvas, pt1, pt2, (50, 50, 200), 3)

        # Draw joints
        for lm in landmarks:
            cv2.circle(canvas, (int(lm[0]), int(lm[1])), 5, (200, 50, 50), -1)

        # Phase indicator
        phase = _get_phase(t)
        cv2.putText(canvas, phase, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 0), 2)
        cv2.putText(canvas, f"Frame: {frame_idx}/{total_frames}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

        out.write(canvas)

    out.release()
    print(f"  Saved to {output_path}")
    return output_path


def _squat_knee_angle(t: float) -> float:
    """Knee angle trajectory: 180° (straight) → 90° (deep) → 180°."""
    if t < 0.15:
        return 180.0
    elif t < 0.45:
        p = (t - 0.15) / 0.30
        return 180.0 - 90.0 * (p ** 2)
    elif t < 0.55:
        return 90.0
    elif t < 0.85:
        p = (t - 0.55) / 0.30
        p = 1.0 - (1.0 - p) ** 2
        return 90.0 + 90.0 * p
    else:
        return 180.0


def _squat_hip_angle(t: float) -> float:
    """Hip angle trajectory: 180° (upright) → ~60° (deep) → 180°."""
    if t < 0.15:
        return 180.0
    elif t < 0.45:
        p = (t - 0.15) / 0.30
        return 180.0 - 120.0 * (p ** 2)
    elif t < 0.55:
        return 60.0
    elif t < 0.85:
        p = (t - 0.55) / 0.30
        p = 1.0 - (1.0 - p) ** 2
        return 60.0 + 120.0 * p
    else:
        return 180.0


def _squat_ankle_angle(t: float) -> float:
    """Ankle dorsiflexion: 90° (neutral) → 70° → 90°."""
    if t < 0.15:
        return 90.0
    elif t < 0.45:
        p = (t - 0.15) / 0.30
        return 90.0 - 20.0 * (p ** 2)
    elif t < 0.55:
        return 70.0
    elif t < 0.85:
        p = (t - 0.55) / 0.30
        p = 1.0 - (1.0 - p) ** 2
        return 70.0 + 20.0 * p
    else:
        return 90.0


def _squat_shoulder_angle(t: float) -> float:
    """Shoulder: 0° at rest, ~30° forward during squat descent."""
    if t < 0.15:
        return 0.0
    elif t < 0.45:
        p = (t - 0.15) / 0.30
        return 30.0 * (p ** 2)
    elif t < 0.55:
        return 30.0
    elif t < 0.85:
        p = (t - 0.55) / 0.30
        p = 1.0 - (1.0 - p) ** 2
        return 30.0 * (1.0 - p)
    else:
        return 0.0


def _squat_elbow_angle(t: float) -> float:
    """Elbow: 180° (straight) throughout, slight bend at bottom."""
    if t < 0.40:
        return 180.0
    elif t < 0.60:
        return 160.0
    else:
        return 180.0


def _get_phase(t: float) -> str:
    if t < 0.15:
        return "SETUP"
    elif t < 0.45:
        return "DESCENT"
    elif t < 0.55:
        return "BOTTOM"
    elif t < 0.85:
        return "ASCENT"
    else:
        return "LOCKOUT"


def _build_stick_figure_2d(
    root_x, root_y,
    knee_angle_deg, hip_angle_deg, ankle_angle_deg,
    shoulder_angle_deg, elbow_angle_deg,
    seg,
):
    """Build 2D landmark positions from joint angles.

    Returns list of (x, y) pixel coordinates.
    Indices: 0=head, 1=neck, 2=hip_center, 3=l_hip, 4=r_hip,
             5=l_knee, 6=r_knee, 7=l_ankle, 8=r_ankle,
             9=l_shoulder, 10=r_shoulder, 11=l_elbow, 12=r_elbow,
             13=l_wrist, 14=r_wrist
    """
    landmarks = []

    # Convert degrees to radians, adjust to screen coordinate system (y down)
    hip_rad = math.radians(hip_angle_deg - 90)   # 0 = pointing right, adjust
    knee_rad = math.radians(knee_angle_deg - 90)
    ankle_rad = math.radians(ankle_angle_deg - 90)

    # Hip center
    hip_cx, hip_cy = root_x, root_y

    # Neck (above hip center)
    neck_x = hip_cx
    neck_y = hip_cy - seg["shoulder_to_hip"]
    landmarks.append((neck_x, neck_y - seg["head_to_shoulder"]))  # 0: head
    landmarks.append((neck_x, neck_y))                             # 1: neck
    landmarks.append((hip_cx, hip_cy))                             # 2: hip_center

    # Hips (left/right)
    l_hip_x = hip_cx - seg["hip_width"] // 2
    l_hip_y = hip_cy
    r_hip_x = hip_cx + seg["hip_width"] // 2
    r_hip_y = hip_cy
    landmarks.append((l_hip_x, l_hip_y))  # 3: left hip
    landmarks.append((r_hip_x, r_hip_y))  # 4: right hip

    # Thigh angle: hip_angle determines how much thigh angles forward
    thigh_angle = math.radians(180.0 - hip_angle_deg)  # 0=down, positive=forward
    l_knee_x = l_hip_x + seg["hip_to_knee"] * math.sin(thigh_angle)
    l_knee_y = l_hip_y + seg["hip_to_knee"] * math.cos(thigh_angle)
    r_knee_x = r_hip_x + seg["hip_to_knee"] * math.sin(thigh_angle)
    r_knee_y = r_hip_y + seg["hip_to_knee"] * math.cos(thigh_angle)
    landmarks.append((l_knee_x, l_knee_y))  # 5: left knee
    landmarks.append((r_knee_x, r_knee_y))  # 6: right knee

    # Shank angle: knee_angle relative to thigh
    shank_angle = thigh_angle + math.radians(180.0 - knee_angle_deg)
    l_ankle_x = l_knee_x + seg["knee_to_ankle"] * math.sin(shank_angle)
    l_ankle_y = l_knee_y + seg["knee_to_ankle"] * math.cos(shank_angle)
    r_ankle_x = r_knee_x + seg["knee_to_ankle"] * math.sin(shank_angle)
    r_ankle_y = r_knee_y + seg["knee_to_ankle"] * math.cos(shank_angle)
    landmarks.append((l_ankle_x, l_ankle_y))  # 7: left ankle
    landmarks.append((r_ankle_x, r_ankle_y))  # 8: right ankle

    # Shoulders
    l_shoulder_x = neck_x - seg["shoulder_width"] // 2
    l_shoulder_y = neck_y
    r_shoulder_x = neck_x + seg["shoulder_width"] // 2
    r_shoulder_y = neck_y
    landmarks.append((l_shoulder_x, l_shoulder_y))  # 9: left shoulder
    landmarks.append((r_shoulder_x, r_shoulder_y))  # 10: right shoulder

    # Arms (shoulder_angle: forward = positive)
    arm_angle = math.radians(shoulder_angle_deg)
    l_elbow_x = l_shoulder_x + seg["shoulder_to_elbow"] * math.sin(arm_angle)
    l_elbow_y = l_shoulder_y + seg["shoulder_to_elbow"] * math.cos(arm_angle)
    r_elbow_x = r_shoulder_x + seg["shoulder_to_elbow"] * math.sin(arm_angle)
    r_elbow_y = r_shoulder_y + seg["shoulder_to_elbow"] * math.cos(arm_angle)
    landmarks.append((l_elbow_x, l_elbow_y))  # 11: left elbow
    landmarks.append((r_elbow_x, r_elbow_y))  # 12: right elbow

    # Forearms
    forearm_angle = arm_angle + math.radians(180.0 - elbow_angle_deg)
    l_wrist_x = l_elbow_x + seg["elbow_to_wrist"] * math.sin(forearm_angle)
    l_wrist_y = l_elbow_y + seg["elbow_to_wrist"] * math.cos(forearm_angle)
    r_wrist_x = r_elbow_x + seg["elbow_to_wrist"] * math.sin(forearm_angle)
    r_wrist_y = r_elbow_y + seg["elbow_to_wrist"] * math.cos(forearm_angle)
    landmarks.append((l_wrist_x, l_wrist_y))  # 13: left wrist
    landmarks.append((r_wrist_x, r_wrist_y))  # 14: right wrist

    return landmarks


# ---- CLI entry point ----

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic exercise video")
    parser.add_argument("--output", default="test_squat.mp4", help="Output .mp4 path")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = parser.parse_args()

    generate_squat_video(args.output, args.duration, args.fps)
