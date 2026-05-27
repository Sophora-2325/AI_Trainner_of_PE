import cv2
import mediapipe as mp

# 导入MediaPipe手部关键点检测模块
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 打开摄像头
cap = cv2.VideoCapture(0)

# 配置手部检测参数
with mp_hands.Hands(
    static_image_mode=False,  # 适用于视频流，设置为False会启用追踪以提升性能
    max_num_hands=2,          # 最大检测手数量
    min_detection_confidence=0.5,  # 检测置信度阈值
    min_tracking_confidence=0.5) as hands:  # 追踪置信度阈值
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # 为了提高性能，将图像标记为不可写，通过引用传递
        image.flags.writeable = False
        # 转换BGR图像为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 处理图像并检测手部关键点
        results = hands.process(image)
        
        # 在图像上绘制手部注释
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制关键点和连接线
                mp_drawing.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS)
        
        # 水平翻转图像以获得自拍视图显示
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        # 按ESC键退出
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()