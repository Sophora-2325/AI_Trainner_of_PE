"""摄像头打开工具 — Windows 兼容."""

import cv2


def open_camera(device_id: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
    """尝试打开摄像头，Windows 优先使用 DirectShow."""
    candidates = [device_id]
    for i in range(3):
        if i not in candidates:
            candidates.append(i)

    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    backends.append(cv2.CAP_ANY)

    for dev in candidates:
        for backend in backends:
            cap = cv2.VideoCapture(dev, backend) if backend != cv2.CAP_ANY else cv2.VideoCapture(dev)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                print(f"[Camera] 已打开 device_id={dev} backend={backend}")
                return cap, dev
            cap.release()

    return None, None
