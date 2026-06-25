"""标准动作模板生成器 — 生成理想动作的 33 关键点 JSON 序列.
第3周：生成 template_squat.json

运行方式:
  python scripts/generate_template.py                   # 生成深蹲模板到 templates/
  python scripts/generate_template.py --movement pushup # 生成俯卧撑模板

原理:
  使用正向运动学生成理想动作的 3D 关键点轨迹，
  模拟标准深蹲/俯卧撑等动作的完整周期。
  输出格式与 extract_pose.py 一致。
"""

import json
import math
import os
import argparse

import numpy as np

# MediaPipe Pose 33 关键点名称 (按索引顺序)
LANDMARK_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# 默认人体比例 (归一化单位)
BODY_PROPS = {
    "head_radius": 0.18,
    "neck_length": 0.10,
    "shoulder_width": 1.00,
    "torso_length": 0.85,
    "hip_width": 0.70,
    "upper_leg": 0.90,
    "lower_leg": 0.85,
    "upper_arm": 0.60,
    "lower_arm": 0.55,
    "foot_length": 0.25,
}


def _smooth_valley(t: np.ndarray, start_val: float, bottom_val: float,
                   descent_pct: float = 0.15, bottom_pct: float = 0.50) -> np.ndarray:
    """生成平滑的山谷曲线，模拟一次完整的动作周期."""
    y = np.full_like(t, start_val, dtype=np.float64)
    hold_end = bottom_pct + 0.07
    for i in range(len(t)):
        ti = t[i]
        if descent_pct < ti <= bottom_pct:
            p = ((ti - descent_pct) / (bottom_pct - descent_pct)) ** 2
            y[i] = start_val + (bottom_val - start_val) * p
        elif bottom_pct < ti <= hold_end:
            y[i] = bottom_val
        elif ti > hold_end:
            p = (ti - hold_end) / (1.0 - hold_end)
            p = 1.0 - (1.0 - p) ** 2
            y[i] = bottom_val + (start_val - bottom_val) * p
    return y


def _vec(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, y, z], dtype=np.float64)


def _build_from_keyframes(
    stand: dict[str, tuple[float, float, float]],
    bottom: dict[str, tuple[float, float, float]],
    num_frames: int = 90,
) -> list[dict]:
    """在两个关键姿态之间插值，生成完整周期 (MediaPipe world 坐标)."""
    t = np.linspace(0, 1, num_frames)
    alpha = _smooth_valley(t, 0.0, 1.0)

    sequence = []
    for f in range(num_frames):
        a = float(alpha[f])
        frame_data = {"frame": f}
        for name in LANDMARK_NAMES:
            sx, sy, sz = stand[name]
            bx, by, bz = bottom[name]
            frame_data[f"{name}_x"] = round(sx + a * (bx - sx), 6)
            frame_data[f"{name}_y"] = round(sy + a * (by - sy), 6)
            frame_data[f"{name}_z"] = round(sz + a * (bz - sz), 6)
        sequence.append(frame_data)
    return sequence


def _lerp_pose(
    pose_a: dict[str, tuple[float, float, float]],
    pose_b: dict[str, tuple[float, float, float]],
    alpha: float,
) -> dict[str, tuple[float, float, float]]:
    """线性插值两个关键姿态."""
    pose = {}
    for name in LANDMARK_NAMES:
        ax, ay, az = pose_a[name]
        bx, by, bz = pose_b[name]
        pose[name] = (
            ax + alpha * (bx - ax),
            ay + alpha * (by - ay),
            az + alpha * (bz - az),
        )
    return pose


def _sequence_from_keyposes(
    keyposes: list[tuple[float, dict[str, tuple[float, float, float]]]],
    num_frames: int = 90,
) -> list[dict]:
    """按关键时刻插值生成完整模板，输入坐标为 MediaPipe world."""
    keyposes = sorted(keyposes, key=lambda item: item[0])
    sequence = []
    for f in range(num_frames):
        t = f / max(num_frames - 1, 1)
        left_t, left_pose = keyposes[0]
        right_t, right_pose = keyposes[-1]
        for i in range(len(keyposes) - 1):
            if keyposes[i][0] <= t <= keyposes[i + 1][0]:
                left_t, left_pose = keyposes[i]
                right_t, right_pose = keyposes[i + 1]
                break
        span = max(right_t - left_t, 1e-6)
        local = (t - left_t) / span
        # smootherstep：关键帧之间速度更自然，避免机械匀速。
        local = local * local * local * (local * (local * 6 - 15) + 10)
        pose = _lerp_pose(left_pose, right_pose, local)
        frame = {"frame": f}
        for name in LANDMARK_NAMES:
            x, y, z = pose[name]
            frame[f"{name}_x"] = round(float(x), 6)
            frame[f"{name}_y"] = round(float(y), 6)
            frame[f"{name}_z"] = round(float(z), 6)
        sequence.append(frame)
    return sequence


def _pose_from_major_points(
    major: dict[str, tuple[float, float, float]],
) -> dict[str, tuple[float, float, float]]:
    """由 12 个主关节补全 MediaPipe 33 点，供标准模板播放使用."""
    pose = {}

    l_sh = np.array(major["left_shoulder"])
    r_sh = np.array(major["right_shoulder"])
    l_el = np.array(major["left_elbow"])
    r_el = np.array(major["right_elbow"])
    l_wr = np.array(major["left_wrist"])
    r_wr = np.array(major["right_wrist"])
    l_hp = np.array(major["left_hip"])
    r_hp = np.array(major["right_hip"])
    l_kn = np.array(major["left_knee"])
    r_kn = np.array(major["right_knee"])
    l_an = np.array(major["left_ankle"])
    r_an = np.array(major["right_ankle"])

    shoulder_mid = (l_sh + r_sh) * 0.5
    hip_mid = (l_hp + r_hp) * 0.5
    trunk = shoulder_mid - hip_mid
    trunk_norm = trunk / max(np.linalg.norm(trunk), 1e-6)
    face_dir = np.array([0.0, -0.05, 0.16])
    head_base = shoulder_mid + trunk_norm * 0.17

    pose["nose"] = tuple(head_base + face_dir)
    pose["left_eye_inner"] = tuple(head_base + np.array([0.025, -0.015, 0.13]))
    pose["left_eye"] = tuple(head_base + np.array([0.04, -0.015, 0.13]))
    pose["left_eye_outer"] = tuple(head_base + np.array([0.055, -0.015, 0.12]))
    pose["right_eye_inner"] = tuple(head_base + np.array([-0.025, -0.015, 0.13]))
    pose["right_eye"] = tuple(head_base + np.array([-0.04, -0.015, 0.13]))
    pose["right_eye_outer"] = tuple(head_base + np.array([-0.055, -0.015, 0.12]))
    pose["left_ear"] = tuple(head_base + np.array([0.09, 0.0, 0.02]))
    pose["right_ear"] = tuple(head_base + np.array([-0.09, 0.0, 0.02]))
    pose["mouth_left"] = tuple(head_base + np.array([0.03, 0.02, 0.14]))
    pose["mouth_right"] = tuple(head_base + np.array([-0.03, 0.02, 0.14]))

    for name, value in major.items():
        pose[name] = value

    def hand_points(wrist: np.ndarray, side: float):
        return {
            "pinky": tuple(wrist + np.array([0.035 * side, 0.01, 0.01])),
            "index": tuple(wrist + np.array([0.01 * side, 0.00, 0.045])),
            "thumb": tuple(wrist + np.array([-0.03 * side, 0.01, 0.02])),
        }

    left_hand = hand_points(l_wr, 1.0)
    right_hand = hand_points(r_wr, -1.0)
    pose["left_pinky"] = left_hand["pinky"]
    pose["left_index"] = left_hand["index"]
    pose["left_thumb"] = left_hand["thumb"]
    pose["right_pinky"] = right_hand["pinky"]
    pose["right_index"] = right_hand["index"]
    pose["right_thumb"] = right_hand["thumb"]

    pose["left_heel"] = tuple(l_an + np.array([0.0, 0.045, -0.055]))
    pose["right_heel"] = tuple(r_an + np.array([0.0, 0.045, -0.055]))
    pose["left_foot_index"] = tuple(l_an + np.array([0.035, 0.085, 0.11]))
    pose["right_foot_index"] = tuple(r_an + np.array([-0.035, 0.085, 0.11]))

    return pose


def _build_squat_landmarks(num_frames: int = 90) -> list[dict]:
    """基于实测关键帧的标准深蹲 (比纯 FK 更接近真实 MediaPipe 比例)."""
    from scripts.squat_keyframes import SQUAT_STAND, SQUAT_BOTTOM
    return _build_from_keyframes(SQUAT_STAND, SQUAT_BOTTOM, num_frames)


def _build_pushup_landmarks(num_frames: int = 90) -> list[dict]:
    """关键帧标准俯卧撑：四肢着地、身体卧倒，前肢屈伸上下运动."""
    top = _pose_from_major_points({
        "left_shoulder": (0.18, -0.18, 0.30),
        "right_shoulder": (-0.18, -0.18, 0.30),
        "left_elbow": (0.23, -0.07, 0.23),
        "right_elbow": (-0.23, -0.07, 0.23),
        "left_wrist": (0.25, 0.08, 0.24),
        "right_wrist": (-0.25, 0.08, 0.24),
        "left_hip": (0.13, -0.12, -0.02),
        "right_hip": (-0.13, -0.12, -0.02),
        "left_knee": (0.12, -0.03, -0.30),
        "right_knee": (-0.12, -0.03, -0.30),
        "left_ankle": (0.10, 0.08, -0.58),
        "right_ankle": (-0.10, 0.08, -0.58),
    })
    bottom = _pose_from_major_points({
        "left_shoulder": (0.18, 0.00, 0.30),
        "right_shoulder": (-0.18, 0.00, 0.30),
        "left_elbow": (0.34, 0.02, 0.24),
        "right_elbow": (-0.34, 0.02, 0.24),
        "left_wrist": (0.25, 0.08, 0.24),
        "right_wrist": (-0.25, 0.08, 0.24),
        "left_hip": (0.13, -0.02, -0.02),
        "right_hip": (-0.13, -0.02, -0.02),
        "left_knee": (0.12, 0.02, -0.30),
        "right_knee": (-0.12, 0.02, -0.30),
        "left_ankle": (0.10, 0.08, -0.58),
        "right_ankle": (-0.10, 0.08, -0.58),
    })
    return _sequence_from_keyposes([(0.0, top), (0.48, bottom), (0.58, bottom), (1.0, top)], num_frames)


def _build_deadlift_landmarks(num_frames: int = 90) -> list[dict]:
    """关键帧标准硬拉：髋主导，上背稳定，杠铃路径贴近小腿."""
    setup = _pose_from_major_points({
        "left_shoulder": (0.18, -0.38, 0.14),
        "right_shoulder": (-0.18, -0.38, 0.14),
        "left_elbow": (0.20, -0.10, 0.12),
        "right_elbow": (-0.20, -0.10, 0.12),
        "left_wrist": (0.22, 0.18, 0.10),
        "right_wrist": (-0.22, 0.18, 0.10),
        "left_hip": (0.13, 0.0, -0.02),
        "right_hip": (-0.13, 0.0, -0.02),
        "left_knee": (0.16, 0.34, 0.03),
        "right_knee": (-0.16, 0.34, 0.03),
        "left_ankle": (0.14, 0.68, 0.02),
        "right_ankle": (-0.14, 0.68, 0.02),
    })
    lockout = _pose_from_major_points({
        "left_shoulder": (0.18, -0.55, 0.08),
        "right_shoulder": (-0.18, -0.55, 0.08),
        "left_elbow": (0.20, -0.25, 0.07),
        "right_elbow": (-0.20, -0.25, 0.07),
        "left_wrist": (0.22, 0.02, 0.06),
        "right_wrist": (-0.22, 0.02, 0.06),
        "left_hip": (0.13, 0.0, -0.02),
        "right_hip": (-0.13, 0.0, -0.02),
        "left_knee": (0.13, 0.40, 0.00),
        "right_knee": (-0.13, 0.40, 0.00),
        "left_ankle": (0.13, 0.73, 0.00),
        "right_ankle": (-0.13, 0.73, 0.00),
    })
    hinge = _pose_from_major_points({
        "left_shoulder": (0.18, -0.44, 0.12),
        "right_shoulder": (-0.18, -0.44, 0.12),
        "left_elbow": (0.20, -0.16, 0.10),
        "right_elbow": (-0.20, -0.16, 0.10),
        "left_wrist": (0.22, 0.10, 0.08),
        "right_wrist": (-0.22, 0.10, 0.08),
        "left_hip": (0.13, 0.0, -0.02),
        "right_hip": (-0.13, 0.0, -0.02),
        "left_knee": (0.15, 0.37, 0.02),
        "right_knee": (-0.15, 0.37, 0.02),
        "left_ankle": (0.13, 0.71, 0.01),
        "right_ankle": (-0.13, 0.71, 0.01),
    })
    return _sequence_from_keyposes(
        [(0.0, setup), (0.46, lockout), (0.58, lockout), (0.82, hinge), (1.0, setup)],
        num_frames,
    )


def _build_pullup_landmarks(num_frames: int = 90) -> list[dict]:
    """正向运动学构建标准引体向上的 33 关键点.

    核心关节角度:
      - 肘角: 175°(悬垂) → 22°(下巴过杠) → 175°(下放)
      - 肩角: 170°(过头) → 5°(肩收) → 170°
      双手固定于单杠，身体通过屈肘上拉
    """
    t = np.linspace(0, 1, num_frames)
    p = BODY_PROPS

    elbow_deg = _smooth_valley(t, 175.0, 22.0)
    shoulder_deg = _smooth_valley(t, 170.0, 5.0)

    sequence = []
    bar_y = 2.4
    half_sw = p["shoulder_width"] / 2.0
    grip_offset = 0.06

    for f in range(num_frames):
        ed = math.radians(elbow_deg[f])
        sd = math.radians(shoulder_deg[f])

        # 双手固定于单杠
        LWrist = _vec( half_sw + grip_offset, bar_y, 0.0)
        RWrist = _vec(-half_sw - grip_offset, bar_y, 0.0)

        # 前臂: 从手腕指向肘部 (大致向下/向外)
        # 肘角 = 前臂与上臂夹角, ed ≈ π 时臂伸直(肘在腕正下方)
        forearm_len = p["lower_arm"]
        upperarm_len = p["upper_arm"]

        # 当肘角接近180°(伸直悬垂)，肘在腕下方；当肘角接近0°(屈曲)，肘在腕侧上方
        elbow_drop = math.cos(ed - math.pi) * forearm_len
        elbow_out = math.sin(ed - math.pi) * forearm_len * 0.15
        LElbow = LWrist + _vec( elbow_out, -abs(elbow_drop) - 0.02, -0.05)
        RElbow = RWrist + _vec(-elbow_out, -abs(elbow_drop) - 0.02, -0.05)

        # 上臂: 从肘到肩，肩角决定肩相对于肘的位置
        shoulder_drop = math.cos(sd - math.pi) * upperarm_len
        shoulder_out = math.sin(sd - math.pi) * upperarm_len * 0.2
        LShoulder = LElbow + _vec( shoulder_out, -abs(shoulder_drop), 0.1)
        RShoulder = RElbow + _vec(-shoulder_out, -abs(shoulder_drop), 0.1)

        # 身体位置从肩部推导
        shoulder_mid_y = (LShoulder[1] + RShoulder[1]) / 2.0
        spine_top = _vec(0.0, shoulder_mid_y, 0.0)

        # 头颈
        neck = spine_top + _vec(0.0, p["neck_length"] * 0.5, 0.15)
        head_center = neck + _vec(0.0, p["head_radius"] * 1.2, 0.15)
        nose = head_center + _vec(0.0, p["head_radius"] * 0.6, p["head_radius"] * 0.3)

        le = head_center + _vec( p["head_radius"] * 0.3, p["head_radius"] * 0.8, p["head_radius"] * 0.6)
        re = head_center + _vec(-p["head_radius"] * 0.3, p["head_radius"] * 0.8, p["head_radius"] * 0.6)
        le_i = head_center + _vec( p["head_radius"] * 0.1, p["head_radius"] * 0.8, p["head_radius"] * 0.6)
        re_i = head_center + _vec(-p["head_radius"] * 0.1, p["head_radius"] * 0.8, p["head_radius"] * 0.6)
        le_o = head_center + _vec( p["head_radius"] * 0.5, p["head_radius"] * 0.8, p["head_radius"] * 0.5)
        re_o = head_center + _vec(-p["head_radius"] * 0.5, p["head_radius"] * 0.8, p["head_radius"] * 0.5)
        lear = head_center + _vec( p["head_radius"] * 0.9, p["head_radius"] * 0.3, 0.0)
        rear = head_center + _vec(-p["head_radius"] * 0.9, p["head_radius"] * 0.3, 0.0)
        ml = head_center + _vec( p["head_radius"] * 0.2, p["head_radius"] * 0.4, p["head_radius"] * 0.7)
        mr = head_center + _vec(-p["head_radius"] * 0.2, p["head_radius"] * 0.4, p["head_radius"] * 0.7)

        # 手部细节
        LPinky = LWrist + _vec( 0.02, 0.01, -0.02)
        RPinky = RWrist + _vec(-0.02, 0.01, -0.02)
        LIndex = LWrist + _vec( 0.005, 0.0, -0.03)
        RIndex = RWrist + _vec(-0.005, 0.0, -0.03)
        LThumb = LWrist + _vec(-0.02, 0.0, -0.01)
        RThumb = RWrist + _vec( 0.02, 0.0, -0.01)

        # 髋部
        pelvis_y = shoulder_mid_y - p["torso_length"]
        half_hw = p["hip_width"] / 2.0
        LHip = _vec( half_hw, pelvis_y, -0.02)
        RHip = _vec(-half_hw, pelvis_y, -0.02)

        # 腿
        LKnee = LHip + _vec(0.0, -p["upper_leg"], 0.05)
        RKnee = RHip + _vec(0.0, -p["upper_leg"], 0.05)
        LAnkle = LKnee + _vec(0.0, -p["lower_leg"], 0.02)
        RAnkle = RKnee + _vec(0.0, -p["lower_leg"], 0.02)
        LHeel = LAnkle + _vec(0.0, -p["foot_length"] * 0.2, -p["foot_length"] * 0.3)
        RHeel = RAnkle + _vec(0.0, -p["foot_length"] * 0.2, -p["foot_length"] * 0.3)
        LFootIdx = LAnkle + _vec(0.0, -p["foot_length"] * 0.05, p["foot_length"] * 0.5)
        RFootIdx = RAnkle + _vec(0.0, -p["foot_length"] * 0.05, p["foot_length"] * 0.5)

        pts = [
            nose, le_i, le, le_o, re_i, re, re_o, lear, rear, ml, mr,
            LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist,
            LPinky, RPinky, LIndex, RIndex, LThumb, RThumb,
            LHip, RHip, LKnee, RKnee, LAnkle, RAnkle,
            LHeel, RHeel, LFootIdx, RFootIdx,
        ]

        frame_data = {"frame": f}
        for i, name in enumerate(LANDMARK_NAMES):
            pt = pts[i]
            frame_data[f"{name}_x"] = round(float(pt[0]), 6)
            frame_data[f"{name}_y"] = round(float(pt[1]), 6)
            frame_data[f"{name}_z"] = round(float(pt[2]), 6)
        sequence.append(frame_data)

    return sequence


def _build_plank_landmarks(num_frames: int = 90) -> list[dict]:
    """关键帧标准平板支撑：四肢撑地、身体卧倒、静止保持."""
    hold = _pose_from_major_points({
        "left_shoulder": (0.18, -0.13, 0.30),
        "right_shoulder": (-0.18, -0.13, 0.30),
        "left_elbow": (0.24, 0.07, 0.24),
        "right_elbow": (-0.24, 0.07, 0.24),
        "left_wrist": (0.24, 0.08, 0.10),
        "right_wrist": (-0.24, 0.08, 0.10),
        "left_hip": (0.13, -0.12, -0.02),
        "right_hip": (-0.13, -0.12, -0.02),
        "left_knee": (0.12, -0.03, -0.30),
        "right_knee": (-0.12, -0.03, -0.30),
        "left_ankle": (0.10, 0.08, -0.58),
        "right_ankle": (-0.10, 0.08, -0.58),
    })
    return _sequence_from_keyposes([(0.0, hold), (1.0, hold)], num_frames)


def _build_shooting_landmarks(num_frames: int = 90) -> list[dict]:
    """正向运动学构建标准投篮动作的 33 关键点.

    核心关节角度:
      - 膝角: 120°(微屈) → 80°(深屈蓄力) → 180°(蹬伸)
      - 右肘角: 60°(持球) → 45°(举球) → 180°(出手)
      - 右肩角: 30°(准备) → 135°(举球) → 165°(出手跟随)
    """
    t = np.linspace(0, 1, num_frames)
    p = BODY_PROPS

    knee_deg = _smooth_valley(t, 160.0, 75.0, descent_pct=0.10, bottom_pct=0.38)
    hip_deg = _smooth_valley(t, 160.0, 70.0, descent_pct=0.10, bottom_pct=0.38)
    elbow_deg = _smooth_valley(t, 55.0, 42.0, descent_pct=0.05, bottom_pct=0.40)
    for i in range(len(t)):
        if t[i] > 0.42:
            frac = (t[i] - 0.42) / 0.25
            elbow_deg[i] = 42.0 + 138.0 * min(frac ** 2, 1.0)
    shoulder_deg = _smooth_valley(t, 35.0, 130.0, descent_pct=0.05, bottom_pct=0.42)
    # 出手后肩角继续伸展至165°，保持跟随动作
    for i in range(len(t)):
        if t[i] > 0.42:
            frac = (t[i] - 0.42) / 0.30
            shoulder_deg[i] = 130.0 + 35.0 * min(frac, 1.0)

    sequence = []
    for f in range(num_frames):
        kd = math.radians(knee_deg[f])
        hd = math.radians(hip_deg[f])
        ed = math.radians(elbow_deg[f])
        sd = math.radians(shoulder_deg[f])

        squat_frac = 1.0 - (hip_deg[f] / 160.0)
        pelvis_y = 0.95 - squat_frac * 0.25
        pelvis = _vec(0.0, pelvis_y, 0.0)

        torso_lean = math.radians((160.0 - hip_deg[f]) * 0.15)
        spine_top = pelvis + _vec(
            math.sin(torso_lean) * p["torso_length"],
            math.cos(torso_lean) * p["torso_length"],
            0.0,
        )

        neck = spine_top + _vec(0.0, p["neck_length"], 0.0)
        head_center = neck + _vec(0.0, p["head_radius"] * 1.5, 0.0)
        nose = head_center + _vec(0.0, p["head_radius"], p["head_radius"] * 0.5)

        le = head_center + _vec( p["head_radius"] * 0.3, p["head_radius"] * 1.0, p["head_radius"] * 0.8)
        re = head_center + _vec(-p["head_radius"] * 0.3, p["head_radius"] * 1.0, p["head_radius"] * 0.8)
        le_i = head_center + _vec( p["head_radius"] * 0.1, p["head_radius"] * 1.0, p["head_radius"] * 0.8)
        re_i = head_center + _vec(-p["head_radius"] * 0.1, p["head_radius"] * 1.0, p["head_radius"] * 0.8)
        le_o = head_center + _vec( p["head_radius"] * 0.5, p["head_radius"] * 1.0, p["head_radius"] * 0.7)
        re_o = head_center + _vec(-p["head_radius"] * 0.5, p["head_radius"] * 1.0, p["head_radius"] * 0.7)
        lear = head_center + _vec( p["head_radius"] * 0.9, p["head_radius"] * 0.4, 0.0)
        rear = head_center + _vec(-p["head_radius"] * 0.9, p["head_radius"] * 0.4, 0.0)
        ml = head_center + _vec( p["head_radius"] * 0.25, p["head_radius"] * 0.5, p["head_radius"] * 0.9)
        mr = head_center + _vec(-p["head_radius"] * 0.25, p["head_radius"] * 0.5, p["head_radius"] * 0.9)

        half_sw = p["shoulder_width"] / 2.0
        LShoulder = _vec( half_sw, spine_top[1] - 0.02, 0.0)
        RShoulder = _vec(-half_sw, spine_top[1] - 0.02, 0.0)

        # 左臂 (辅手)
        LElbow = LShoulder + _vec(0.05, -p["upper_arm"] * 0.5, p["upper_arm"] * 0.3)
        LWrist = LElbow + _vec(0.02, -p["lower_arm"] * 0.4, p["lower_arm"] * 0.3)
        LPinky = LWrist + _vec( 0.03, -0.01, 0.02)
        LIndex = LWrist + _vec( 0.005, -0.01, 0.04)
        LThumb = LWrist + _vec(-0.03,  0.005, 0.01)

        # 右臂 (投篮手)
        shoot_dir = _vec(
            math.sin(sd) * 0.1,
            math.cos(sd),
            math.sin(sd) * 0.6,
        )
        RElbow = RShoulder + shoot_dir * p["upper_arm"]
        forearm_angle = ed - math.pi
        forearm_dir = _vec(
            shoot_dir[0] + math.sin(forearm_angle) * 0.15,
            shoot_dir[1] - math.cos(forearm_angle) * 0.5,
            shoot_dir[2] * 0.7,
        )
        RWrist = RElbow + (forearm_dir / max(np.linalg.norm(forearm_dir), 0.001)) * p["lower_arm"]
        RPinky = RWrist + _vec(-0.03, -0.01, 0.02)
        RIndex = RWrist + _vec(-0.005, -0.01, 0.04)
        RThumb = RWrist + _vec( 0.03,  0.005, 0.01)

        half_hw = p["hip_width"] / 2.0
        LHip = _vec( half_hw, pelvis_y, -0.02)
        RHip = _vec(-half_hw, pelvis_y, -0.02)

        thigh_angle = math.pi - hd
        thigh_dir = _vec(
            math.sin(torso_lean + thigh_angle * 0.45),
            -math.cos(thigh_angle * 0.75),
            0.02,
        )
        LKnee = LHip + thigh_dir * p["upper_leg"]
        RKnee = RHip + thigh_dir * p["upper_leg"]

        shank_angle = math.pi - kd
        shank_dir = _vec(
            thigh_dir[0] + math.sin(shank_angle - thigh_angle * 0.4) * 0.25,
            thigh_dir[1] - math.cos(shank_angle) * 0.75,
            0.0,
        )
        LAnkle = LKnee + shank_dir * p["lower_leg"]
        RAnkle = RKnee + shank_dir * p["lower_leg"]

        LHeel = LAnkle + _vec(0.0, -p["foot_length"] * 0.25, -p["foot_length"] * 0.4)
        RHeel = RAnkle + _vec(0.0, -p["foot_length"] * 0.25, -p["foot_length"] * 0.4)
        LFootIdx = LAnkle + _vec(0.0, -p["foot_length"] * 0.05,  p["foot_length"] * 0.6)
        RFootIdx = RAnkle + _vec(0.0, -p["foot_length"] * 0.05,  p["foot_length"] * 0.6)

        pts = [
            nose, le_i, le, le_o, re_i, re, re_o, lear, rear, ml, mr,
            LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist,
            LPinky, RPinky, LIndex, RIndex, LThumb, RThumb,
            LHip, RHip, LKnee, RKnee, LAnkle, RAnkle,
            LHeel, RHeel, LFootIdx, RFootIdx,
        ]

        frame_data = {"frame": f}
        for i, name in enumerate(LANDMARK_NAMES):
            pt = pts[i]
            frame_data[f"{name}_x"] = round(float(pt[0]), 6)
            frame_data[f"{name}_y"] = round(float(pt[1]), 6)
            frame_data[f"{name}_z"] = round(float(pt[2]), 6)
        sequence.append(frame_data)

    return sequence


def _convert_fk_to_world_space(sequence: list[dict]) -> list[dict]:
    """将正向运动学坐标转为 MediaPipe world 坐标系 (髋部为原点, Y 向下, 米).

    使用首帧统一缩放，避免逐帧缩放导致动作失真。
    """
    if not sequence:
        return sequence

    fr0 = sequence[0]
    hip_x = (fr0["left_hip_x"] + fr0["right_hip_x"]) / 2
    hip_y = (fr0["left_hip_y"] + fr0["right_hip_y"]) / 2
    hip_z = (fr0["left_hip_z"] + fr0["right_hip_z"]) / 2
    ys0 = [fr0[f"{n}_y"] for n in LANDMARK_NAMES]
    span = max(ys0) - min(ys0)
    scale = 0.85 / span if span > 1e-6 else 0.35

    converted = []
    for frame in sequence:
        new_frame = {"frame": frame["frame"]}
        for name in LANDMARK_NAMES:
            new_frame[f"{name}_x"] = round((frame[f"{name}_x"] - hip_x) * scale, 6)
            new_frame[f"{name}_y"] = round(-(frame[f"{name}_y"] - hip_y) * scale, 6)
            new_frame[f"{name}_z"] = round((frame[f"{name}_z"] - hip_z) * scale, 6)
        converted.append(new_frame)
    return converted


# 已是 MediaPipe world 坐标的关键帧模板，无需 FK 转换
WORLD_SPACE_MOVEMENTS = {"squat", "pushup", "deadlift", "plank"}


def generate_template(movement: str = "squat", output_dir: str = "templates") -> str:
    """生成指定动作的标准模板 JSON 文件.

    Args:
        movement: 动作名称 (squat / pushup)
        output_dir: 输出目录

    Returns:
        输出文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    builders = {
        "squat": _build_squat_landmarks,
        "pushup": _build_pushup_landmarks,
        "deadlift": _build_deadlift_landmarks,
        "pullup": _build_pullup_landmarks,
        "plank": _build_plank_landmarks,
        "shooting": _build_shooting_landmarks,
    }

    if movement in builders:
        sequence = builders[movement](num_frames=90)
    else:
        print(f"[generate_template] 警告: '{movement}' 未实现，使用深蹲模板")
        sequence = _build_squat_landmarks(num_frames=90)

    if movement not in WORLD_SPACE_MOVEMENTS:
        sequence = _convert_fk_to_world_space(sequence)

    output_path = os.path.join(output_dir, f"template_{movement}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sequence, f, ensure_ascii=False, indent=2)

    print(f"[generate_template] '{movement}' 模板已生成:")
    print(f"  总帧数:       {len(sequence)}")
    print(f"  每帧关键点数: {len(LANDMARK_NAMES)}")
    print(f"  每帧坐标字段: {len(LANDMARK_NAMES) * 3} (x, y, z)")
    print(f"  输出文件:     {output_path}")

    if sequence:
        s0 = sequence[0]
        print(f"  样例帧0: nose=({s0['nose_x']:.4f}, {s0['nose_y']:.4f}, {s0['nose_z']:.4f})")
        s_mid = sequence[len(sequence) // 2]
        print(f"  样例帧{len(sequence)//2}: nose=({s_mid['nose_x']:.4f}, {s_mid['nose_y']:.4f}, {s_mid['nose_z']:.4f})")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="生成标准动作模板 JSON (33关键点)")
    parser.add_argument("--movement", "-m", default="squat",
                        choices=["squat", "pushup", "deadlift", "pullup", "plank", "shooting"],
                        help="动作名称 (默认: squat)")
    parser.add_argument("--output-dir", "-o", default="templates",
                        help="输出目录 (默认: templates/)")
    args = parser.parse_args()

    generate_template(args.movement, args.output_dir)


if __name__ == "__main__":
    main()
