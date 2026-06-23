"""MediaPipe Pose 姿态估计 — Tasks API (兼容 mediapipe >= 0.10.30)."""

import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from src.pose.model_manager import get_pose_model_path


@dataclass
class PoseResult:
    """姿态检测结果."""
    landmarks_3d: np.ndarray    # (33, 4) — x, y, z, visibility
    landmarks_2d: np.ndarray    # (33, 3) — x, y, visibility (图像坐标)
    world_landmarks: np.ndarray # (33, 4) — 世界坐标系
    timestamp: float
    detected: bool


class PoseEstimator:
    """MediaPipe Pose Landmarker 封装."""

    def __init__(
        self,
        model_complexity: int = 2,
        static_image_mode: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smooth_landmarks: bool = True,
    ):
        del static_image_mode, smooth_landmarks  # Tasks API 由模型文件决定
        variant = {0: "lite", 1: "full", 2: "heavy"}.get(model_complexity, "lite")
        model_path = get_pose_model_path(variant)

        options = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray, timestamp: Optional[float] = None) -> PoseResult:
        """检测单帧 RGB 图像中的人体姿态."""
        h, w = frame.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
        results = self._landmarker.detect(mp_image)

        if not results.pose_landmarks:
            return PoseResult(
                landmarks_3d=np.zeros((33, 4), dtype=np.float32),
                landmarks_2d=np.zeros((33, 3), dtype=np.float32),
                world_landmarks=np.zeros((33, 4), dtype=np.float32),
                timestamp=timestamp or 0.0,
                detected=False,
            )

        pose = results.pose_landmarks[0]
        world = results.pose_world_landmarks[0] if results.pose_world_landmarks else pose

        lm_2d = np.array([[lm.x * w, lm.y * h, lm.visibility] for lm in pose], dtype=np.float32)
        lm_3d = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose], dtype=np.float32)
        world_lm = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in world], dtype=np.float32)

        return PoseResult(
            landmarks_3d=lm_3d,
            landmarks_2d=lm_2d,
            world_landmarks=world_lm,
            timestamp=timestamp or 0.0,
            detected=True,
        )

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
