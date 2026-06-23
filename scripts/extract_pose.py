"""姿态提取脚本 — 从视频中提取 MediaPipe 33 关键点并保存为 JSON.
第3周：姿态提取与存储

运行方式:
  python scripts/extract_pose.py --video test_squat.mp4
  python scripts/extract_pose.py --video my_squat.mp4 --output my_pose.json --skip 10

输出 JSON 格式:
  [{"frame": 0, "nose_x": 0.5, "nose_y": 0.3, "nose_z": -0.1, ...}, ...]
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

# MediaPipe Pose 33个关键点名称 (按索引顺序)
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


def extract_pose(
    video_path: str,
    output_path: str = None,
    skip_frames: int = 10,
    coord_space: str = "image",
) -> list[dict]:
    """从视频中提取姿态关键点序列.

    Args:
        video_path: 输入视频路径
        output_path: 输出 JSON 路径
        skip_frames: 下采样间隔
        coord_space: image=屏幕归一化(0~1, 供 LLM 评分) / world=3D世界坐标(供 3D孪生模板)

    Returns:
        关键点序列列表
    """
    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{base}_pose.json"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[extract_pose] 视频信息: {total_frames} 帧, {fps:.1f} fps")
    print(f"[extract_pose] 下采样间隔: 每 {skip_frames} 帧保存一次")
    print(f"[extract_pose] 坐标系: {coord_space} ({'3D孪生模板' if coord_space == 'world' else 'LLM评分'})")

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.pose.estimator import PoseEstimator

    estimator = PoseEstimator(model_complexity=1)
    pose_sequence = []
    frame_idx = 0
    saved_count = 0

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = estimator.detect(image_rgb)

            if result.detected:
                landmarks = result.world_landmarks if coord_space == "world" else result.landmarks_3d
                frame_data = {"frame": frame_idx}
                for i, name in enumerate(LANDMARK_NAMES):
                    frame_data[f"{name}_x"] = round(float(landmarks[i, 0]), 6)
                    frame_data[f"{name}_y"] = round(float(landmarks[i, 1]), 6)
                    frame_data[f"{name}_z"] = round(float(landmarks[i, 2]), 6)
                pose_sequence.append(frame_data)
                saved_count += 1

            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"  [extract_pose] 已处理 {frame_idx}/{total_frames} 帧, "
                      f"已保存 {saved_count} 帧关键点")
    finally:
        estimator.close()
        cap.release()

    # 保存 JSON
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pose_sequence, f, ensure_ascii=False, indent=2)

    # ─── 验证输出 ──────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"[验证] 提取完成:")
    print(f"  视频总帧数:     {total_frames}")
    print(f"  下采样后帧数:   {len(pose_sequence)}")
    print(f"  每帧关键点数:   {len(LANDMARK_NAMES)}")
    print(f"  每帧坐标字段数: {len(LANDMARK_NAMES) * 3} (x, y, z)")
    print(f"  输出文件:       {output_path}")

    # 打印前2帧样例
    if pose_sequence:
        frame0 = pose_sequence[0]
        sample_keys = [k for k in frame0.keys() if k != "frame"][:6]
        print(f"  样例帧0: frame={frame0['frame']}")
        for k in sample_keys:
            print(f"    {k}: {frame0[k]}")

    print(f"{'='*50}")
    return pose_sequence


def main():
    parser = argparse.ArgumentParser(description="从视频提取 MediaPipe Pose 关键点")
    parser.add_argument("--video", "-v", required=True, help="输入视频路径")
    parser.add_argument("--output", "-o", default=None,
                        help="输出 JSON 路径 (默认: {视频名}_pose.json)")
    parser.add_argument("--skip", "-s", type=int, default=10,
                        help="下采样间隔, 每N帧保存一次 (默认: 10)")
    parser.add_argument("--space", choices=["image", "world"], default="image",
                        help="image=屏幕归一化(LLM); world=3D世界坐标(3D孪生模板, 默认image)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"错误: 视频文件不存在: {args.video}")
        sys.exit(1)

    extract_pose(args.video, args.output, args.skip, args.space)


if __name__ == "__main__":
    main()
