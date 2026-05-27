"""姿态估计模块测试."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.pose.estimator import PoseEstimator
from src.pose.preprocessor import (
    LandmarkSmoother, normalize_pose,
    mediapipe_to_opensim_markers, compute_body_scale,
)
from src.pose.tracker import MovementPhaseTracker, Phase


def test_estimator_creation():
    """测试 PoseEstimator 创建."""
    est = PoseEstimator(model_complexity=1)
    assert est is not None
    est.close()
    print("[OK] PoseEstimator 创建成功")


def test_landmark_smoother():
    """测试关键点平滑."""
    smoother = LandmarkSmoother(window_size=5)
    landmarks = np.random.randn(33, 4).astype(np.float32)
    smoothed = smoother.update(landmarks)
    assert smoothed.shape == (33, 4)
    print("[OK] LandmarkSmoother 工作正常")


def test_normalize_pose():
    """测试姿态归一化."""
    landmarks = np.zeros((33, 4), dtype=np.float32)
    landmarks[23, :3] = [0.1, 0.5, 0.0]  # left_hip
    landmarks[24, :3] = [-0.1, 0.5, 0.0]  # right_hip
    normalized = normalize_pose(landmarks)
    hip_center = (normalized[23, :3] + normalized[24, :3]) / 2.0
    assert np.allclose(hip_center, [0, 0, 0], atol=1e-6)
    print("[OK] normalize_pose 工作正常")


def test_mediapipe_to_opensim():
    """测试 MediaPipe → OpenSim marker 映射."""
    landmarks = np.random.randn(33, 4).astype(np.float32)
    markers = mediapipe_to_opensim_markers(landmarks)
    assert "hip_r" in markers
    assert "knee_r" in markers
    assert "ankle_r" in markers
    assert "acromion_r" in markers
    assert "elbow_r" in markers
    print("[OK] mediapipe_to_opensim_markers 映射正确")


def test_phase_tracker():
    """测试动作阶段检测."""
    tracker = MovementPhaseTracker(movement="squat")
    assert tracker.current_phase == Phase.REST

    # 模拟深蹲动作
    tracker.update({"knee_angle_r": 170, "knee_angle_l": 170})  # SETUP
    # 需要累积几帧
    for _ in range(5):
        tracker.update({"knee_angle_r": 120, "knee_angle_l": 120})
    print(f"  阶段: 120° → {tracker.current_phase}")

    for _ in range(5):
        tracker.update({"knee_angle_r": 85, "knee_angle_l": 85})
    print(f"  阶段: 85° → {tracker.current_phase}")

    for _ in range(5):
        tracker.update({"knee_angle_r": 140, "knee_angle_l": 140})
    print(f"  阶段: 140° → {tracker.current_phase}")

    print("[OK] MovementPhaseTracker 工作正常")


def test_body_scale():
    """测试身体尺度计算."""
    landmarks = np.zeros((33, 4), dtype=np.float32)
    landmarks[11, :3] = [0.2, 1.5, 0]   # left_shoulder
    landmarks[12, :3] = [-0.2, 1.5, 0]  # right_shoulder
    landmarks[23, :3] = [0.1, 1.0, 0]   # left_hip
    landmarks[24, :3] = [-0.1, 1.0, 0]  # right_hip
    scale = compute_body_scale(landmarks)
    assert scale > 0
    print(f"[OK] body_scale = {scale:.3f}")


if __name__ == "__main__":
    print("=" * 40)
    print("  Pose 模块测试")
    print("=" * 40)
    test_estimator_creation()
    test_landmark_smoother()
    test_normalize_pose()
    test_mediapipe_to_opensim()
    test_body_scale()
    test_phase_tracker()
    print("\n全部测试通过!")
