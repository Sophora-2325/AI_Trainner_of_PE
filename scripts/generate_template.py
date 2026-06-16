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


def _build_squat_landmarks(num_frames: int = 90) -> list[dict]:
    """使用正向运动学构建标准深蹲的 33 关键点序列.

    核心关节角度:
      - 膝角: 180°(直立) → 95°(底部) → 180°
      - 髋角: 180° → 60° → 180°
      - 踝角: 90° → 70° → 90°
    """
    t = np.linspace(0, 1, num_frames)
    p = BODY_PROPS

    # 关节角度轨迹 (度)
    knee_deg = _smooth_valley(t, 180.0, 95.0)
    hip_deg = _smooth_valley(t, 180.0, 60.0)
    ankle_deg = _smooth_valley(t, 90.0, 70.0)
    arm_fwd_deg = _smooth_valley(t, 0.0, 55.0)

    sequence = []

    for f in range(num_frames):
        ka = math.radians(knee_deg[f])
        ha = math.radians(hip_deg[f])
        aa = math.radians(ankle_deg[f])
        af = math.radians(arm_fwd_deg[f])

        # ─── 骨盆 — 随下蹲降低 ──────────────────────
        squat_fraction = 1.0 - (hip_deg[f] / 180.0)
        pelvis_y = 1.0 - squat_fraction * 0.42
        pelvis = _vec(0.0, pelvis_y, 0.0)

        # ─── 躯干倾斜 ──────────────────────────────
        torso_lean = math.radians((180.0 - hip_deg[f]) * 0.28)
        spine_top = pelvis + _vec(
            math.sin(torso_lean) * p["torso_length"],
            math.cos(torso_lean) * p["torso_length"],
            0.0,
        )

        # ─── 头颈 ──────────────────────────────────
        neck = spine_top + _vec(0.0, p["neck_length"], 0.0)
        head_center = neck + _vec(0.0, p["head_radius"] * 1.5, 0.0)
        nose = head_center + _vec(0.0, p["head_radius"], p["head_radius"] * 0.5)

        # 面部点 (在头部球面附近分布)
        le = head_center + _vec( p["head_radius"] * 0.3,  p["head_radius"] * 1.0, p["head_radius"] * 0.8)
        re = head_center + _vec(-p["head_radius"] * 0.3,  p["head_radius"] * 1.0, p["head_radius"] * 0.8)
        le_i = head_center + _vec( p["head_radius"] * 0.1,  p["head_radius"] * 1.0, p["head_radius"] * 0.8)
        re_i = head_center + _vec(-p["head_radius"] * 0.1,  p["head_radius"] * 1.0, p["head_radius"] * 0.8)
        le_o = head_center + _vec( p["head_radius"] * 0.5,  p["head_radius"] * 1.0, p["head_radius"] * 0.7)
        re_o = head_center + _vec(-p["head_radius"] * 0.5,  p["head_radius"] * 1.0, p["head_radius"] * 0.7)
        lear = head_center + _vec( p["head_radius"] * 0.9,  p["head_radius"] * 0.4, 0.0)
        rear = head_center + _vec(-p["head_radius"] * 0.9,  p["head_radius"] * 0.4, 0.0)
        ml = head_center + _vec( p["head_radius"] * 0.25, p["head_radius"] * 0.5, p["head_radius"] * 0.9)
        mr = head_center + _vec(-p["head_radius"] * 0.25, p["head_radius"] * 0.5, p["head_radius"] * 0.9)

        # ─── 肩部 ──────────────────────────────────
        half_sw = p["shoulder_width"] / 2.0
        LShoulder = _vec( half_sw, spine_top[1] - 0.02, 0.0)
        RShoulder = _vec(-half_sw, spine_top[1] - 0.02, 0.0)

        # ─── 手臂 (前伸保持平衡) ───────────────────
        arm_dir = _vec(math.sin(af), -math.cos(af) * 0.25, math.cos(af))
        LElbow = LShoulder + arm_dir * p["upper_arm"]
        RElbow = RShoulder + arm_dir * p["upper_arm"]
        LWrist = LElbow + arm_dir * p["lower_arm"]
        RWrist = RElbow + arm_dir * p["lower_arm"]

        LPinky = LWrist + _vec( 0.03, -0.01, 0.03)
        RPinky = RWrist + _vec(-0.03, -0.01, 0.03)
        LIndex = LWrist + _vec( 0.005, -0.01, 0.05)
        RIndex = RWrist + _vec(-0.005, -0.01, 0.05)
        LThumb = LWrist + _vec(-0.03,  0.005, 0.02)
        RThumb = RWrist + _vec( 0.03,  0.005, 0.02)

        # ─── 髋部 ──────────────────────────────────
        half_hw = p["hip_width"] / 2.0
        LHip = _vec( half_hw, pelvis_y, -0.02)
        RHip = _vec(-half_hw, pelvis_y, -0.02)

        # ─── 大腿 ──────────────────────────────────
        thigh_angle_vert = math.pi - ha
        thigh_dir = _vec(
            math.sin(torso_lean + thigh_angle_vert * 0.45),
            -math.cos(thigh_angle_vert * 0.75),
            0.02,
        )
        LKnee = LHip + thigh_dir * p["upper_leg"]
        RKnee = RHip + thigh_dir * p["upper_leg"]

        # ─── 小腿 ──────────────────────────────────
        shank_angle = math.pi - ka
        shank_dir = _vec(
            thigh_dir[0] + math.sin(shank_angle - thigh_angle_vert * 0.4) * 0.25,
            thigh_dir[1] - math.cos(shank_angle) * 0.75,
            0.0,
        )
        LAnkle = LKnee + shank_dir * p["lower_leg"]
        RAnkle = RKnee + shank_dir * p["lower_leg"]

        # ─── 足部 ──────────────────────────────────
        LHeel = LAnkle + _vec(0.0, -p["foot_length"] * 0.25, -p["foot_length"] * 0.4)
        RHeel = RAnkle + _vec(0.0, -p["foot_length"] * 0.25, -p["foot_length"] * 0.4)
        LFootIdx = LAnkle + _vec(0.0, -p["foot_length"] * 0.05,  p["foot_length"] * 0.6)
        RFootIdx = RAnkle + _vec(0.0, -p["foot_length"] * 0.05,  p["foot_length"] * 0.6)

        # ─── 按 MediaPipe 索引顺序组装 ──────────────
        pts = [
            nose,                                                              # 0
            le_i, le, le_o,                                                    # 1-3
            re_i, re, re_o,                                                    # 4-6
            lear, rear,                                                        # 7-8
            ml, mr,                                                            # 9-10
            LShoulder, RShoulder,                                              # 11-12
            LElbow, RElbow,                                                    # 13-14
            LWrist, RWrist,                                                    # 15-16
            LPinky, RPinky,                                                    # 17-18
            LIndex, RIndex,                                                    # 19-20
            LThumb, RThumb,                                                    # 21-22
            LHip, RHip,                                                        # 23-24
            LKnee, RKnee,                                                      # 25-26
            LAnkle, RAnkle,                                                    # 27-28
            LHeel, RHeel,                                                      # 29-30
            LFootIdx, RFootIdx,                                                # 31-32
        ]

        frame_data = {"frame": f}
        for i, name in enumerate(LANDMARK_NAMES):
            pt = pts[i]
            frame_data[f"{name}_x"] = round(float(pt[0]), 6)
            frame_data[f"{name}_y"] = round(float(pt[1]), 6)
            frame_data[f"{name}_z"] = round(float(pt[2]), 6)

        sequence.append(frame_data)

    return sequence


def _build_pushup_landmarks(num_frames: int = 90) -> list[dict]:
    """正向运动学构建标准俯卧撑的 33 关键点.

    核心关节角度:
      - 肘角: 180°(直臂) → 90°(底部) → 180°
      - 肩角: 约 45° 外展
      - 身体呈直线
    """
    t = np.linspace(0, 1, num_frames)
    p = BODY_PROPS

    elbow_deg = _smooth_valley(t, 180.0, 90.0)
    # 身体高度随俯卧撑变化 (相对于地面)
    body_height = _smooth_valley(t, 0.35, 0.12)

    sequence = []
    plank_y = 0.12  # 地面以上身体高度

    for f in range(num_frames):
        ed = math.radians(elbow_deg[f])
        bh = body_height[f]
        shoulder_y = bh + 0.15  # 肩在髋之上

        # 骨盆 — 平板姿势，身体呈直线
        pelvis = _vec(0.0, bh, 0.0)

        # 躯干 (水平，略抬头)
        spine_top = _vec(0.0, shoulder_y, 0.0)
        neck = spine_top + _vec(0.0, p["neck_length"] * 0.7, p["neck_length"] * 0.5)
        head_center = neck + _vec(0.0, p["head_radius"] * 0.8, p["head_radius"] * 0.8)
        nose = head_center + _vec(0.0, p["head_radius"] * 0.7, p["head_radius"] * 0.5)
        # 面部点省略完整计算，使用简化偏移
        le = head_center + _vec( p["head_radius"] * 0.3, p["head_radius"] * 0.5, p["head_radius"] * 0.6)
        re = head_center + _vec(-p["head_radius"] * 0.3, p["head_radius"] * 0.5, p["head_radius"] * 0.6)
        le_i = head_center + _vec( p["head_radius"] * 0.1, p["head_radius"] * 0.5, p["head_radius"] * 0.6)
        re_i = head_center + _vec(-p["head_radius"] * 0.1, p["head_radius"] * 0.5, p["head_radius"] * 0.6)
        le_o = head_center + _vec( p["head_radius"] * 0.5, p["head_radius"] * 0.5, p["head_radius"] * 0.5)
        re_o = head_center + _vec(-p["head_radius"] * 0.5, p["head_radius"] * 0.5, p["head_radius"] * 0.5)
        lear = head_center + _vec( p["head_radius"] * 0.8, p["head_radius"] * 0.2, 0.0)
        rear = head_center + _vec(-p["head_radius"] * 0.8, p["head_radius"] * 0.2, 0.0)
        ml = head_center + _vec( p["head_radius"] * 0.2, p["head_radius"] * 0.3, p["head_radius"] * 0.7)
        mr = head_center + _vec(-p["head_radius"] * 0.2, p["head_radius"] * 0.3, p["head_radius"] * 0.7)

        # 肩
        half_sw = p["shoulder_width"] / 2.0
        LShoulder = _vec( half_sw, shoulder_y, 0.0)
        RShoulder = _vec(-half_sw, shoulder_y, 0.0)

        # 手臂 — 手撑地
        upper_arm_angle = math.pi - ed
        LElbow = LShoulder + _vec(0.0, -p["upper_arm"] * math.cos(upper_arm_angle * 0.5),
                                  -p["upper_arm"] * math.sin(upper_arm_angle * 0.5))
        RElbow = RShoulder + _vec(0.0, -p["upper_arm"] * math.cos(upper_arm_angle * 0.5),
                                  -p["upper_arm"] * math.sin(upper_arm_angle * 0.5))
        # 前臂指向地面
        forearm_dir = _vec(0.0, -1.0, 0.0)
        LWrist = LElbow + forearm_dir * p["lower_arm"]
        RWrist = RElbow + forearm_dir * p["lower_arm"]
        LPinky = LWrist + _vec( 0.03, 0.02, -0.02)
        RPinky = RWrist + _vec(-0.03, 0.02, -0.02)
        LIndex = LWrist + _vec( 0.0, 0.0, -0.04)
        RIndex = RWrist + _vec( 0.0, 0.0, -0.04)
        LThumb = LWrist + _vec(-0.03, 0.0, -0.02)
        RThumb = RWrist + _vec( 0.03, 0.0, -0.02)

        # 髋
        half_hw = p["hip_width"] / 2.0
        LHip = _vec( half_hw, bh + 0.02, -0.02)
        RHip = _vec(-half_hw, bh + 0.02, -0.02)

        # 腿 (伸直)
        leg_len = p["upper_leg"] + p["lower_leg"]
        LKnee = LHip + _vec(0.0, -p["upper_leg"], 0.0)
        RKnee = RHip + _vec(0.0, -p["upper_leg"], 0.0)
        LAnkle = LKnee + _vec(0.0, -p["lower_leg"], 0.0)
        RAnkle = RKnee + _vec(0.0, -p["lower_leg"], 0.0)
        LHeel = LAnkle + _vec(0.0, 0.0, -p["foot_length"] * 0.3)
        RHeel = RAnkle + _vec(0.0, 0.0, -p["foot_length"] * 0.3)
        LFootIdx = LAnkle + _vec(0.0, -p["foot_length"] * 0.05, p["foot_length"] * 0.3)
        RFootIdx = RAnkle + _vec(0.0, -p["foot_length"] * 0.05, p["foot_length"] * 0.3)

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


def _build_deadlift_landmarks(num_frames: int = 90) -> list[dict]:
    """正向运动学构建标准硬拉的 33 关键点.

    核心关节角度:
      - 髋角: 60°(俯身) → 180°(直立锁定)
      - 膝角: 130°(起始) → 180°(锁定) → 130°(下放)
    """
    t = np.linspace(0, 1, num_frames)
    p = BODY_PROPS

    hip_deg = _smooth_valley(t, 60.0, 60.0, descent_pct=0.01, bottom_pct=0.55)
    for i in range(len(t)):
        if t[i] > 0.55:
            frac = (t[i] - 0.55) / 0.45
            hip_deg[i] = 60.0 + 120.0 * (1.0 - (1.0 - frac) ** 2)
    hip_deg = np.array(hip_deg)

    knee_deg = np.full_like(t, 130.0)
    for i in range(len(t)):
        if t[i] > 0.55:
            frac = (t[i] - 0.55) / 0.25
            if frac < 1.0:
                knee_deg[i] = 130.0 + 50.0 * frac
            else:
                knee_deg[i] = 180.0

    sequence = []
    for f in range(num_frames):
        ha = math.radians(hip_deg[f])
        ka = math.radians(knee_deg[f])

        hip_y = 0.95
        pelvis = _vec(0.0, hip_y, 0.0)

        # 躯干前倾: 髋角60°时前倾约54°, 锁定后直立
        torso_lean = math.radians((180.0 - hip_deg[f]) * 0.45)
        spine_top = pelvis + _vec(
            math.sin(torso_lean) * p["torso_length"],
            math.cos(torso_lean) * p["torso_length"],
            0.0,
        )

        neck = spine_top + _vec(
            math.sin(torso_lean) * p["neck_length"] * 0.3,
            math.cos(torso_lean) * p["neck_length"],
            0.0,
        )
        head_center = neck + _vec(
            math.sin(torso_lean) * p["head_radius"] * 0.8,
            math.cos(torso_lean) * p["head_radius"] * 1.5,
            0.0,
        )
        nose = head_center + _vec(
            math.sin(torso_lean) * p["head_radius"] * 0.3,
            math.cos(torso_lean) * p["head_radius"],
            p["head_radius"] * 0.4,
        )

        le = head_center + _vec( p["head_radius"] * 0.3, p["head_radius"] * 1.0, p["head_radius"] * 0.7)
        re = head_center + _vec(-p["head_radius"] * 0.3, p["head_radius"] * 1.0, p["head_radius"] * 0.7)
        le_i = head_center + _vec( p["head_radius"] * 0.1, p["head_radius"] * 1.0, p["head_radius"] * 0.7)
        re_i = head_center + _vec(-p["head_radius"] * 0.1, p["head_radius"] * 1.0, p["head_radius"] * 0.7)
        le_o = head_center + _vec( p["head_radius"] * 0.5, p["head_radius"] * 1.0, p["head_radius"] * 0.6)
        re_o = head_center + _vec(-p["head_radius"] * 0.5, p["head_radius"] * 1.0, p["head_radius"] * 0.6)
        lear = head_center + _vec( p["head_radius"] * 0.9, p["head_radius"] * 0.4, 0.0)
        rear = head_center + _vec(-p["head_radius"] * 0.9, p["head_radius"] * 0.4, 0.0)
        ml = head_center + _vec( p["head_radius"] * 0.25, p["head_radius"] * 0.5, p["head_radius"] * 0.8)
        mr = head_center + _vec(-p["head_radius"] * 0.25, p["head_radius"] * 0.5, p["head_radius"] * 0.8)

        half_sw = p["shoulder_width"] / 2.0
        LShoulder = _vec( half_sw, spine_top[1] - 0.02, 0.0)
        RShoulder = _vec(-half_sw, spine_top[1] - 0.02, 0.0)

        LElbow = LShoulder + _vec(0.0, -p["upper_arm"], 0.05)
        RElbow = RShoulder + _vec(0.0, -p["upper_arm"], 0.05)
        LWrist = LElbow + _vec(0.0, -p["lower_arm"], 0.02)
        RWrist = RElbow + _vec(0.0, -p["lower_arm"], 0.02)
        LPinky = LWrist + _vec( 0.03, -0.01, 0.02)
        RPinky = RWrist + _vec(-0.03, -0.01, 0.02)
        LIndex = LWrist + _vec( 0.005, -0.01, 0.04)
        RIndex = RWrist + _vec(-0.005, -0.01, 0.04)
        LThumb = LWrist + _vec(-0.03,  0.005, 0.01)
        RThumb = RWrist + _vec( 0.03,  0.005, 0.01)

        half_hw = p["hip_width"] / 2.0
        LHip = _vec( half_hw, hip_y, -0.02)
        RHip = _vec(-half_hw, hip_y, -0.02)

        thigh_angle = math.pi - ha - math.pi / 2.0
        thigh_dir = _vec(math.sin(torso_lean + thigh_angle) * 0.3, -0.8, 0.02)
        LKnee = LHip + thigh_dir * p["upper_leg"]
        RKnee = RHip + thigh_dir * p["upper_leg"]

        shank_angle = math.pi - ka
        shank_dir = _vec(-math.sin(shank_angle - math.pi / 2.0) * 0.25, -0.85, 0.0)
        LAnkle = LKnee + shank_dir * p["lower_leg"]
        RAnkle = RKnee + shank_dir * p["lower_leg"]

        LHeel = LAnkle + _vec(0.0, -p["foot_length"] * 0.25, -p["foot_length"] * 0.3)
        RHeel = RAnkle + _vec(0.0, -p["foot_length"] * 0.25, -p["foot_length"] * 0.3)
        LFootIdx = LAnkle + _vec(0.0, -p["foot_length"] * 0.05,  p["foot_length"] * 0.5)
        RFootIdx = RAnkle + _vec(0.0, -p["foot_length"] * 0.05,  p["foot_length"] * 0.5)

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
    """正向运动学构建标准平板支撑的 33 关键点.

    核心关节角度:
      - 身体保持直线 (肩-髋-膝-踝对齐)
      - 肘部90°, 支撑于肩正下方
    """
    t = np.linspace(0, 1, num_frames)
    p = BODY_PROPS

    sequence = []
    plank_height = 0.12
    shoulder_y = plank_height + 0.14

    for f in range(num_frames):
        pelvis = _vec(0.0, plank_height, 0.0)
        spine_top = _vec(0.0, shoulder_y, 0.0)
        neck = spine_top + _vec(0.0, p["neck_length"] * 0.7, p["neck_length"] * 0.5)
        head_center = neck + _vec(0.0, p["head_radius"] * 0.8, p["head_radius"] * 0.8)
        nose = head_center + _vec(0.0, p["head_radius"] * 0.7, p["head_radius"] * 0.5)

        le = head_center + _vec( p["head_radius"] * 0.3, p["head_radius"] * 0.5, p["head_radius"] * 0.6)
        re = head_center + _vec(-p["head_radius"] * 0.3, p["head_radius"] * 0.5, p["head_radius"] * 0.6)
        le_i = head_center + _vec( p["head_radius"] * 0.1, p["head_radius"] * 0.5, p["head_radius"] * 0.6)
        re_i = head_center + _vec(-p["head_radius"] * 0.1, p["head_radius"] * 0.5, p["head_radius"] * 0.6)
        le_o = head_center + _vec( p["head_radius"] * 0.5, p["head_radius"] * 0.5, p["head_radius"] * 0.5)
        re_o = head_center + _vec(-p["head_radius"] * 0.5, p["head_radius"] * 0.5, p["head_radius"] * 0.5)
        lear = head_center + _vec( p["head_radius"] * 0.8, p["head_radius"] * 0.2, 0.0)
        rear = head_center + _vec(-p["head_radius"] * 0.8, p["head_radius"] * 0.2, 0.0)
        ml = head_center + _vec( p["head_radius"] * 0.2, p["head_radius"] * 0.3, p["head_radius"] * 0.7)
        mr = head_center + _vec(-p["head_radius"] * 0.2, p["head_radius"] * 0.3, p["head_radius"] * 0.7)

        half_sw = p["shoulder_width"] / 2.0
        LShoulder = _vec( half_sw, shoulder_y, 0.0)
        RShoulder = _vec(-half_sw, shoulder_y, 0.0)

        LElbow = LShoulder + _vec(0.0, -p["upper_arm"] * 0.6, -p["upper_arm"] * 0.4)
        RElbow = RShoulder + _vec(0.0, -p["upper_arm"] * 0.6, -p["upper_arm"] * 0.4)
        LWrist = LElbow + _vec(0.0, -p["lower_arm"] * 0.3, p["lower_arm"] * 0.5)
        RWrist = RElbow + _vec(0.0, -p["lower_arm"] * 0.3, p["lower_arm"] * 0.5)
        LPinky = LWrist + _vec( 0.03, 0.02, -0.02)
        RPinky = RWrist + _vec(-0.03, 0.02, -0.02)
        LIndex = LWrist + _vec( 0.0, 0.0, -0.04)
        RIndex = RWrist + _vec( 0.0, 0.0, -0.04)
        LThumb = LWrist + _vec(-0.03, 0.0, -0.02)
        RThumb = RWrist + _vec( 0.03, 0.0, -0.02)

        half_hw = p["hip_width"] / 2.0
        LHip = _vec( half_hw, plank_height + 0.02, -0.02)
        RHip = _vec(-half_hw, plank_height + 0.02, -0.02)

        LKnee = LHip + _vec(0.0, -p["upper_leg"], 0.0)
        RKnee = RHip + _vec(0.0, -p["upper_leg"], 0.0)
        LAnkle = LKnee + _vec(0.0, -p["lower_leg"], 0.0)
        RAnkle = RKnee + _vec(0.0, -p["lower_leg"], 0.0)
        LHeel = LAnkle + _vec(0.0, 0.0, -p["foot_length"] * 0.3)
        RHeel = RAnkle + _vec(0.0, 0.0, -p["foot_length"] * 0.3)
        LFootIdx = LAnkle + _vec(0.0, -p["foot_length"] * 0.05, p["foot_length"] * 0.3)
        RFootIdx = RAnkle + _vec(0.0, -p["foot_length"] * 0.05, p["foot_length"] * 0.3)

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
