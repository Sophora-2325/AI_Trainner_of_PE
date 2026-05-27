"""关键点预处理：滤波、归一化、坐标变换."""

import numpy as np
from collections import deque
from typing import Optional


# MediaPipe Pose 关键点索引
MP_LANDMARK_NAMES = {
    0:  "nose",
    1:  "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4:  "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7:  "left_ear", 8: "right_ear",
    9:  "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    17: "left_pinky", 18: "right_pinky",
    19: "left_index", 20: "right_index",
    21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
    29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index",
}


class LandmarkSmoother:
    """指数滑动平均 (EMA) 关键点平滑滤波器."""

    def __init__(self, window_size: int = 5, alpha: float = 0.3):
        self.alpha = alpha
        self.history: deque = deque(maxlen=window_size)
        self._initialized = False

    def update(self, landmarks: np.ndarray) -> np.ndarray:
        """对关键点序列做 EMA 滤波.

        Args:
            landmarks: (33, 4) 当前帧关键点

        Returns:
            平滑后的关键点
        """
        self.history.append(landmarks)
        if not self._initialized:
            if len(self.history) >= 3:
                self._initialized = True
            return landmarks
        return self.alpha * landmarks + (1 - self.alpha) * self._last_smoothed

    @property
    def _last_smoothed(self):
        if len(self.history) < 2:
            return self.history[-1] if self.history else np.zeros((33, 4))
        return self.history[-2]

    def reset(self):
        self.history.clear()
        self._initialized = False


def smooth_landmarks(
    landmarks: np.ndarray,
    history: Optional[deque] = None,
    window: int = 5,
) -> np.ndarray:
    """简单移动平均平滑."""
    if history is None:
        history = deque(maxlen=window)
    history.append(landmarks)
    if len(history) < 2:
        return landmarks
    return np.mean(history, axis=0)


def normalize_pose(landmarks: np.ndarray) -> np.ndarray:
    """以左右髋部中点为原点归一化关键点.

    Args:
        landmarks: (33, 4) 原始关键点

    Returns:
        归一化后的关键点，以髋部中点为 (0,0,0)
    """
    hip_center = (landmarks[23, :3] + landmarks[24, :3]) / 2.0
    normalized = landmarks.copy()
    normalized[:, :3] -= hip_center
    return normalized


def compute_body_scale(landmarks: np.ndarray) -> float:
    """估算身体尺度（用于后续缩放），使用肩到髋的距离."""
    shoulder_mid = (landmarks[11, :3] + landmarks[12, :3]) / 2.0
    hip_mid = (landmarks[23, :3] + landmarks[24, :3]) / 2.0
    return np.linalg.norm(shoulder_mid - hip_mid)


def mediapipe_to_opensim_markers(landmarks_3d: np.ndarray) -> dict:
    """将 MediaPipe 33个3D关键点映射为 OpenSim 虚拟 marker 集.

    Args:
        landmarks_3d: (33, 4) MediaPipe全身关键点

    Returns:
        dict: {marker_name: (x, y, z)} OpenSim marker坐标
    """
    # 提取坐标部分
    pts = landmarks_3d[:, :3]

    # MediaPipe → OpenSim marker 映射
    mapping = {
        # 头部
        "head":         pts[0],                       # nose
        "head_extra_r": pts[8],                       # right_ear
        "head_extra_l": pts[7],                       # left_ear

        # 躯干
        "acromion_r":   pts[12],                      # right_shoulder
        "acromion_l":   pts[11],                      # left_shoulder
        "mid_torso":    (pts[11] + pts[12]) / 2.0,    # shoulder midpoint → C7

        # 骨盆
        "hip_r":        pts[24],                      # right_hip
        "hip_l":        pts[23],                      # left_hip
        "sacrum":       (pts[23] + pts[24]) / 2.0,    # hip midpoint → sacrum

        # 上肢
        "elbow_r":      pts[14],                      # right_elbow
        "elbow_l":      pts[13],                      # left_elbow
        "wrist_r":      pts[16],                      # right_wrist
        "wrist_l":      pts[15],                      # left_wrist

        # 下肢
        "knee_r":       pts[26],                      # right_knee
        "knee_l":       pts[25],                      # left_knee
        "ankle_r":      pts[28],                      # right_ankle
        "ankle_l":      pts[27],                      # left_ankle

        # 足部
        "heel_r":       pts[30],                      # right_heel
        "heel_l":       pts[29],                      # left_heel
        "toe_r":        pts[32],                      # right_foot_index
        "toe_l":        pts[31],                      # left_foot_index
    }

    return mapping
