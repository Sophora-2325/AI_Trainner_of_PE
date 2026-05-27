"""MediaPipe Pose 姿态估计封装."""

import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional

mp_pose = mp.solutions.pose


@dataclass
class PoseResult:
    """姿态检测结果."""
    landmarks_3d: np.ndarray    # (33, 4) — x, y, z, visibility
    landmarks_2d: np.ndarray    # (33, 3) — x, y, visibility (图像坐标)
    world_landmarks: np.ndarray # (33, 4) — 世界坐标系
    timestamp: float
    detected: bool


class PoseEstimator:
    """MediaPipe Pose 封装，提供实时全身姿态估计."""

    def __init__(
        self,
        model_complexity: int = 2,
        static_image_mode: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smooth_landmarks: bool = True,
    ):
        self.pose = mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame: np.ndarray, timestamp: Optional[float] = None) -> PoseResult:
        """检测单帧图像中的人体姿态.

        Args:
            frame: BGR图像 (H, W, 3)
            timestamp: 时间戳

        Returns:
            PoseResult 包含3D/2D关键点和世界坐标
        """
        rgb = frame  # 期望输入已是RGB格式
        results = self.pose.process(rgb)

        if results.pose_landmarks is None:
            return PoseResult(
                landmarks_3d=np.zeros((33, 4)),
                landmarks_2d=np.zeros((33, 3)),
                world_landmarks=np.zeros((33, 4)),
                timestamp=timestamp or 0.0,
                detected=False,
            )

        h, w = frame.shape[:2]

        # 提取图像坐标关键点
        lm_2d = np.array([
            [lm.x * w, lm.y * h, lm.visibility]
            for lm in results.pose_landmarks.landmark
        ], dtype=np.float32)

        # 提取归一化3D关键点 (相对于髋部中心)
        lm_3d = np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in results.pose_landmarks.landmark
        ], dtype=np.float32)

        # 提取世界坐标 (米)
        if results.pose_world_landmarks:
            world_lm = np.array([
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in results.pose_world_landmarks.landmark
            ], dtype=np.float32)
        else:
            world_lm = np.zeros((33, 4), dtype=np.float32)

        return PoseResult(
            landmarks_3d=lm_3d,
            landmarks_2d=lm_2d,
            world_landmarks=world_lm,
            timestamp=timestamp or 0.0,
            detected=True,
        )

    def close(self):
        self.pose.close()
