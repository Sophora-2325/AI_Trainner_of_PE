"""视频 → 骨骼序列批量提取器.

将健身动作视频数据集批量转换为 MediaPipe 骨骼关键点序列。
"""

import os
import glob
import numpy as np
import cv2
from typing import Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.pose.estimator import PoseEstimator


@dataclass
class SkeletonSequence:
    """单条骨骼序列."""
    filepath: str
    landmarks: np.ndarray        # (T, 33, 4) 关键点序列
    fps: float
    duration: float
    label: str = ""              # 动作标签
    error_labels: list = None    # 错误类型标签 (多标签)


class VideoProcessor:
    """批量处理视频文件，提取骨骼关键点序列."""

    def __init__(
        self,
        model_complexity: int = 2,
        target_fps: float = 30.0,
        max_duration: float = 15.0,
    ):
        """
        Args:
            model_complexity: MediaPipe Pose 复杂度
            target_fps: 目标提取帧率
            max_duration: 单视频最大处理时长(秒)
        """
        self.estimator = PoseEstimator(model_complexity=model_complexity)
        self.target_fps = target_fps
        self.max_duration = max_duration

    def process_video(self, video_path: str) -> Optional[SkeletonSequence]:
        """处理单个视频文件.

        Args:
            video_path: 视频文件路径

        Returns:
            SkeletonSequence 或 None（处理失败时）
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[VideoProcessor] 无法打开视频: {video_path}")
            return None

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0:
            original_fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = int(self.max_duration * original_fps)
        frames_to_process = min(total_frames, max_frames)

        # 计算抽帧间隔（匹配目标帧率）
        skip_interval = max(1, int(original_fps / self.target_fps))

        landmarks_list = []
        timestamps = []

        for i in range(frames_to_process):
            ret, frame = cap.read()
            if not ret:
                break

            if i % skip_interval != 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.estimator.detect(frame_rgb)

            if result.detected:
                landmarks_list.append(result.world_landmarks)
            else:
                # 保持序列长度，填充NaN
                landmarks_list.append(np.full((33, 4), np.nan, dtype=np.float32))

            timestamps.append(i / original_fps)

        cap.release()

        if len(landmarks_list) < 10:
            print(f"[VideoProcessor] 有效帧不足: {video_path}")
            return None

        landmarks = np.stack(landmarks_list)  # (T, 33, 4)
        actual_fps = len(landmarks) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else self.target_fps

        return SkeletonSequence(
            filepath=video_path,
            landmarks=landmarks,
            fps=actual_fps,
            duration=timestamps[-1] - timestamps[0] if timestamps else 0,
        )

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        label: str = "",
        max_workers: int = 4,
    ) -> list[str]:
        """批量处理目录下的所有视频.

        Args:
            input_dir: 输入视频目录
            output_dir: 输出 .npy 文件目录
            label: 动作类型标签
            max_workers: 并行处理数

        Returns:
            已处理的文件路径列表
        """
        os.makedirs(output_dir, exist_ok=True)

        video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(input_dir, ext)))
            video_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

        if not video_files:
            print(f"[VideoProcessor] 目录中未找到视频: {input_dir}")
            return []

        output_paths = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process_video, vf): vf
                for vf in video_files
            }

            for future in as_completed(futures):
                vf = futures[future]
                try:
                    seq = future.result()
                    if seq is not None:
                        seq.label = label
                        basename = os.path.splitext(os.path.basename(vf))[0]
                        out_path = os.path.join(output_dir, f"{basename}.npy")
                        self._save_sequence(seq, out_path)
                        output_paths.append(out_path)
                        print(f"[VideoProcessor] 已处理: {basename} ({seq.landmarks.shape[0]}帧)")
                except Exception as e:
                    print(f"[VideoProcessor] 处理失败 {vf}: {e}")

        print(f"[VideoProcessor] 完成: {len(output_paths)}/{len(video_files)} 个视频")
        return output_paths

    def close(self):
        self.estimator.close()

    @staticmethod
    def _save_sequence(seq: SkeletonSequence, path: str):
        """保存骨骼序列为 .npy 文件."""
        data = {
            "landmarks": seq.landmarks,
            "fps": seq.fps,
            "duration": seq.duration,
            "label": seq.label,
            "error_labels": seq.error_labels or [],
            "source_file": seq.filepath,
        }
        np.save(path, data, allow_pickle=True)

    @staticmethod
    def load_sequence(path: str) -> Optional[SkeletonSequence]:
        """加载 .npy 骨骼序列."""
        try:
            data = np.load(path, allow_pickle=True).item()
            return SkeletonSequence(
                filepath=path,
                landmarks=data["landmarks"],
                fps=data.get("fps", 30),
                duration=data.get("duration", 0),
                label=data.get("label", ""),
                error_labels=data.get("error_labels", []),
            )
        except Exception as e:
            print(f"[VideoProcessor] 加载失败 {path}: {e}")
            return None


def extract_skeleton_sequence(video_path: str, target_fps: float = 30.0) -> Optional[np.ndarray]:
    """便捷函数: 提取单个视频的骨骼序列.

    Args:
        video_path: 视频路径
        target_fps: 目标帧率

    Returns:
        (T, 33, 4) 关键点数组 或 None
    """
    processor = VideoProcessor(target_fps=target_fps)
    seq = processor.process_video(video_path)
    processor.close()
    return seq.landmarks if seq else None
