"""模板序列后处理 — 重采样、平滑、对齐、单周期截取."""

import copy
from typing import List

import numpy as np

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


def resample_sequence(sequence: list[dict], target_frames: int = 90) -> list[dict]:
    """线性插值到固定帧数 (一个完整动作周期)."""
    n = len(sequence)
    if n < 2:
        return sequence
    if n == target_frames:
        return sequence

    keys = [k for k in sequence[0] if k != "frame"]
    out = []
    for i in range(target_frames):
        t = i * (n - 1) / max(target_frames - 1, 1)
        i0 = int(t)
        i1 = min(i0 + 1, n - 1)
        alpha = t - i0
        frame = {"frame": i}
        for k in keys:
            v0 = float(sequence[i0][k])
            v1 = float(sequence[i1][k])
            frame[k] = round(v0 + (v1 - v0) * alpha, 6)
        out.append(frame)
    return out


def smooth_sequence(sequence: list[dict], window: int = 5) -> list[dict]:
    """滑动平均平滑抖动."""
    if len(sequence) < window:
        return sequence
    keys = [k for k in sequence[0] if k != "frame"]
    half = window // 2
    out = []
    for i in range(len(sequence)):
        frame = {"frame": i}
        for k in keys:
            vals = [
                float(sequence[j][k])
                for j in range(max(0, i - half), min(len(sequence), i + half + 1))
            ]
            frame[k] = round(sum(vals) / len(vals), 6)
        out.append(frame)
    return out


def hip_centered(sequence: list[dict]) -> list[dict]:
    """每帧以左右髋中点为原点 (消除摄像头平移漂移)."""
    out = []
    for i, fr in enumerate(sequence):
        cx = (fr["left_hip_x"] + fr["right_hip_x"]) / 2
        cy = (fr["left_hip_y"] + fr["right_hip_y"]) / 2
        cz = (fr["left_hip_z"] + fr["right_hip_z"]) / 2
        nf = {"frame": i}
        for name in LANDMARK_NAMES:
            nf[f"{name}_x"] = round(fr[f"{name}_x"] - cx, 6)
            nf[f"{name}_y"] = round(fr[f"{name}_y"] - cy, 6)
            nf[f"{name}_z"] = round(fr[f"{name}_z"] - cz, 6)
        out.append(nf)
    return out


def _knee_depth(frame: dict) -> float:
    """膝相对髋的垂直深度 (Y 向下，值越大表示站得越直)."""
    hip_y = (frame["left_hip_y"] + frame["right_hip_y"]) / 2
    knee_y = (frame["left_knee_y"] + frame["right_knee_y"]) / 2
    return knee_y - hip_y


def extract_single_cycle(sequence: list[dict], stand_ratio: float = 0.72) -> list[dict]:
    """从长视频序列中截取一个完整动作周期 (站直 → 底部 → 站直)."""
    if len(sequence) < 30:
        return sequence

    seq = hip_centered(sequence)
    depth = np.array([_knee_depth(fr) for fr in seq], dtype=np.float64)
    bottom = int(np.argmin(depth))

    stand_depth = float(np.median(np.concatenate([
        depth[: max(bottom // 4, 1)],
        depth[min(bottom + (len(depth) - bottom) // 4, len(depth) - 1):],
    ])))
    thresh = depth[bottom] + stand_ratio * (stand_depth - depth[bottom])

    start = bottom
    while start > 0 and depth[start] < thresh:
        start -= 1

    end = bottom
    while end < len(depth) - 1 and depth[end] < thresh:
        end += 1

    cycle = seq[start: end + 1]
    return cycle if len(cycle) >= 10 else seq


def symmetrize_sequence(sequence: list[dict]) -> list[dict]:
    """左右对称化，消除拍摄角度带来的偏差."""
    pairs = []
    for name in LANDMARK_NAMES:
        if name.startswith("left_"):
            right = "right_" + name[5:]
            if right in LANDMARK_NAMES:
                pairs.append((name, right))

    out = []
    for i, fr in enumerate(sequence):
        nf = {"frame": i}
        for lname, rname in pairs:
            lx, ly, lz = fr[f"{lname}_x"], fr[f"{lname}_y"], fr[f"{lname}_z"]
            rx, ry, rz = fr[f"{rname}_x"], fr[f"{rname}_y"], fr[f"{rname}_z"]
            ax = (abs(lx) + abs(rx)) / 2
            ay = (ly + ry) / 2
            az = (abs(lz) + abs(rz)) / 2
            nf[f"{lname}_x"] = round(ax, 6)
            nf[f"{lname}_y"] = round(ly, 6)
            nf[f"{lname}_z"] = round(az, 6)
            nf[f"{rname}_x"] = round(-ax, 6)
            nf[f"{rname}_y"] = round(ry, 6)
            nf[f"{rname}_z"] = round(-az, 6)

        for name in LANDMARK_NAMES:
            if name.startswith("left_") or name.startswith("right_"):
                continue
            for axis in ("x", "y", "z"):
                nf[f"{name}_{axis}"] = fr[f"{name}_{axis}"]
        out.append(nf)
    return out


def refine_for_template(sequence: list[dict], target_frames: int = 90) -> list[dict]:
    """视频提取结果 → 标准模板 (单周期 + 去漂移 + 对称 + 平滑 + 重采样)."""
    seq = extract_single_cycle(sequence)
    seq = hip_centered(seq)
    seq = symmetrize_sequence(seq)
    seq = smooth_sequence(seq, window=5)
    seq = resample_sequence(seq, target_frames)
    return seq
