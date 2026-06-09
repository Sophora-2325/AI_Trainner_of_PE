"""MediaPipe Pose 姿态追踪示例.
第1-2周：环境搭建验证

运行方式:
  python pose_tracking_demo.py             # 使用摄像头
  python pose_tracking_demo.py --video test_squat.mp4  # 使用视频文件

此脚本验证:
  1. MediaPipe 正确安装且可调用
  2. 能检测33个全身关键点
  3. 实时绘制骨骼连线
"""

import cv2
import mediapipe as mp
import argparse

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def main():
    parser = argparse.ArgumentParser(description="MediaPipe Pose 姿态追踪示例")
    parser.add_argument("--video", type=str, default=None, help="输入视频路径")
    parser.add_argument("--model", type=int, default=2,
                        choices=[0, 1, 2], help="模型复杂度: 0=Lite, 1=Full, 2=Heavy")
    args = parser.parse_args()

    if args.video:
        cap = cv2.VideoCapture(args.video)
        print(f"[PoseTracker] 输入视频: {args.video}")
    else:
        cap = cv2.VideoCapture(0)
        print("[PoseTracker] 摄像头模式, 按 ESC 退出")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=args.model,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        frame_idx = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print(f"[PoseTracker] 视频结束, 共处理 {frame_idx} 帧")
                break

            frame_idx += 1
            image = cv2.cvtColor(cv2.flip(image, 1) if args.video is None else image,
                                 cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )
                # 显示关键点数量
                count = len(results.pose_landmarks.landmark)
                cv2.putText(image, f"Keypoints: {count}/33", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(image, "No pose detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("MediaPipe Pose Tracking", image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
