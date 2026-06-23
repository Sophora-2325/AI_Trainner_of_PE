"""MediaPipe Pose Landmarker 模型下载与管理 (Tasks API, >=0.10.30)."""

import os
import urllib.request

from src.utils.paths import resource_path, data_path

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def get_pose_model_path(variant: str = "lite") -> str:
    """返回本地 .task 模型路径，不存在则自动下载."""
    variant = variant if variant in MODEL_URLS else "lite"
    filename = f"pose_landmarker_{variant}.task"
    for base in (data_path("models"), resource_path("models")):
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, filename)
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            return path

    path = data_path("models", filename)
    url = MODEL_URLS[variant]
    print(f"[PoseModel] 正在下载 {variant} 模型…")
    urllib.request.urlretrieve(url, path)
    print(f"[PoseModel] 已保存: {path}")
    return path
