import cv2
import mediapipe as mp

# 初始化 MediaPipe 姿态模型和绘图工具
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 创建姿态检测器对象
pose = mp_pose.Pose(
    min_detection_confidence=0.5,   # 检测置信度阈值（首次检测到人）
    min_tracking_confidence=0.5     # 追踪置信度阈值（后续帧追踪）
)

# 打开摄像头（0 代表默认摄像头，如果有多个可尝试 1, 2 ...）
cap = cv2.VideoCapture(0)

# 创建可自由缩放的窗口
cv2.namedWindow('MediaPipe Pose - 摄像头姿态检测', cv2.WINDOW_NORMAL)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("无法读取摄像头画面")
        break

    # 水平翻转图像，获得镜像效果（更符合自拍视角）
    image = cv2.flip(image, 1)

    # MediaPipe 需要 RGB 格式输入，OpenCV 默认是 BGR，需要转换
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 运行姿态检测
    results = pose.process(image_rgb)

    # 如果检测到人体姿态关键点，就在图像上绘制骨架
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,                         # 绘制目标图像
            results.pose_landmarks,        # 检测到的 33 个关键点
            mp_pose.POSE_CONNECTIONS,      # 关键点之间的连线定义
            # 关键点样式：橙色圆点
            mp_drawing.DrawingSpec(color=(0, 165, 255), thickness=2, circle_radius=3),
            # 连线样式：绿色线条
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )

    # 显示处理后的画面
    cv2.imshow('MediaPipe Pose - 摄像头姿态检测', image)

    # 按下 ESC 键（ASCII 27）退出循环
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 释放摄像头资源并关闭所有窗口
cap.release()
cv2.destroyAllWindows()