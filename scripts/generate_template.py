"""从标准动作视频生成 .npy 模板文件.

用法:
  python scripts/generate_template.py --video <视频路径> --movement squat
"""
import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pose.estimator import PoseEstimator
from src.bridge.socket_server import GeometricIKSolver


def generate_template(video_path: str, movement: str, output_dir: str = "movement_data"):
    import cv2

    estimator = PoseEstimator(model_complexity=2)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 目标帧率 30fps，跳帧提取
    target_fps = 30.0
    skip = max(1, int(fps / target_fps))

    joint_angles_list = []
    frame_count = 0

    print(f"处理视频: {video_path}")
    print(f"  原始帧率: {fps:.1f}, 总帧数: {total_frames}, 抽帧间隔: {skip}")

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if i % skip != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = estimator.detect(frame_rgb)

        if result.detected:
            angles = GeometricIKSolver.solve(result.world_landmarks)
            joint_angles_list.append(angles)
            frame_count += 1

        if frame_count % 30 == 0 and frame_count > 0:
            print(f"  已处理 {frame_count} 帧...")

    cap.release()
    estimator.close()

    if frame_count < 10:
        print(f"有效帧不足: {frame_count}")
        return False

    # 提取深蹲相关关节
    joint_names = [
        "hip_flexion_r", "hip_flexion_l",
        "knee_angle_r", "knee_angle_l",
        "ankle_angle_r", "ankle_angle_l",
        "lumbar_extension",
        "hip_abduction_r", "hip_abduction_l",
    ]

    angles_matrix = np.zeros((frame_count, len(joint_names)))
    for i, angles_dict in enumerate(joint_angles_list):
        for j, name in enumerate(joint_names):
            angles_matrix[i, j] = angles_dict.get(name, 0.0)

    # 自动划分阶段（基于膝角变化）
    knee_avg = (angles_matrix[:, 2] + angles_matrix[:, 3]) / 2.0  # 平均膝角
    min_knee_idx = np.argmin(knee_avg)
    max_start = np.argmax(knee_avg[:min_knee_idx]) if min_knee_idx > 0 else 0
    max_end = np.argmax(knee_avg[min_knee_idx:]) + min_knee_idx

    from src.pose.tracker import Phase
    phases = [
        (0, max_start, Phase.SETUP),
        (max_start, min_knee_idx, Phase.DESCENT),
        (min_knee_idx, min(min_knee_idx + 5, frame_count - 1), Phase.BOTTOM),
        (min_knee_idx + 5, max_end, Phase.ASCENT),
        (max_end, frame_count - 1, Phase.LOCKOUT),
    ]

    # 保存模板
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{movement}_reference.npy")
    data = {
        "joint_angles": angles_matrix,
        "joint_names": joint_names,
        "phases": phases,
        "fps": target_fps,
    }
    np.save(output_path, data, allow_pickle=True)

    print(f"\n模板已保存: {output_path}")
    print(f"  帧数: {frame_count}, 关节: {joint_names}")
    print(f"  各阶段帧: {phases}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从视频生成动作模板")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--movement", default="squat", help="动作名称")
    parser.add_argument("--output-dir", default="movement_data", help="输出目录")
    args = parser.parse_args()

    generate_template(args.video, args.movement, args.output_dir)
