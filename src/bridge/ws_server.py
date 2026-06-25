"""WebSocket 服务器 — 向 sports_twin.html 实时推送 36 个关键点坐标.
第5周：3D孪生对比

协议:
  - 服务端: ws://localhost:8765
  - 消息格式: JSON {"type":"pose","frame":N,"points":[x1,y1,z1, ... x12,y12,z12]}
  - 36 个数值 = 12 个主要关节 × 3 坐标 (x, y, z)

关节顺序:
  0: left_shoulder,  1: right_shoulder
  2: left_elbow,     3: right_elbow
  4: left_wrist,     5: right_wrist
  5: left_hip,       7: right_hip  (注意: 不，index 6,7)
  6: left_hip,       7: right_hip
  8: left_knee,      9: right_knee
  10: left_ankle,    11: right_ankle
"""

import asyncio
import json
import threading
import time
from typing import Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)

# 12 个主要关节在 MediaPipe 33 点中的索引
MAJOR_JOINTS = [
    (11, "left_shoulder"),
    (12, "right_shoulder"),
    (13, "left_elbow"),
    (14, "right_elbow"),
    (15, "left_wrist"),
    (16, "right_wrist"),
    (23, "left_hip"),
    (24, "right_hip"),
    (25, "left_knee"),
    (26, "right_knee"),
    (27, "left_ankle"),
    (28, "right_ankle"),
]

# 骨骼连接定义 (用于前端绘制圆柱体)
BONE_CONNECTIONS = [
    # 躯干
    (0, 1),    # left_shoulder → right_shoulder
    (0, 6),    # left_shoulder → left_hip
    (1, 7),    # right_shoulder → right_hip
    (6, 7),    # left_hip → right_hip
    # 脊柱 (估算: 肩中点 → 髋中点)
    # 左臂
    (0, 2),    # left_shoulder → left_elbow
    (2, 4),    # left_elbow → left_wrist
    # 右臂
    (1, 3),    # right_shoulder → right_elbow
    (3, 5),    # right_elbow → right_wrist
    # 左腿
    (6, 8),    # left_hip → left_knee
    (8, 10),   # left_knee → left_ankle
    # 右腿
    (7, 9),    # right_hip → right_knee
    (9, 11),   # right_knee → right_ankle
]


class TwinWebSocketServer:
    """WebSocket 服务器，向 3D 孪生对比页面推送关键点数据.

    Usage:
        server = TwinWebSocketServer()
        server.on_movement_change = lambda m: print(f"切换动作: {m}")
        server.start()
        server.send_pose(landmarks)   # 每帧调用
        server.stop()
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self._server = None
        self._clients: set = set()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_count = 0
        self.on_movement_change = None  # callback(movement_name)
        self._last_template_message: Optional[str] = None

    def start(self):
        """在后台线程启动 WebSocket 服务器."""
        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        time.sleep(0.5)  # 等待服务器启动
        print(f"[TwinWS] WebSocket 服务器已启动 ws://{self.host}:{self.port}")

    def stop(self):
        """停止服务器."""
        self._running = False
        if self._server:
            try:
                self._server.close()
            except Exception:
                pass

    def _broadcast(self, payload: dict):
        """向所有 WebSocket 客户端广播 JSON 消息."""
        if not self._clients or self._server is None:
            return
        message = json.dumps(payload, ensure_ascii=False)
        dead = set()
        for client in self._clients:
            try:
                asyncio.run_coroutine_threadsafe(client.send(message), self._loop)
            except Exception:
                dead.add(client)
        self._clients -= dead

    def send_pose(self, landmarks, frame_idx: int = None):
        """发送一帧关键点数据到所有连接的客户端."""
        if not self._clients or self._server is None:
            return

        if frame_idx is None:
            self._frame_count += 1
            frame_idx = self._frame_count
        else:
            self._frame_count = frame_idx

        points = []
        for mp_idx, _name in MAJOR_JOINTS:
            pt = landmarks[mp_idx, :3]
            points.extend([round(float(pt[0]), 4),
                           round(float(pt[1]), 4),
                           round(float(pt[2]), 4)])

        self._broadcast({
            "type": "pose",
            "frame": frame_idx,
            "points": points,
            "joints": [n for _, n in MAJOR_JOINTS],
            "bones": BONE_CONNECTIONS,
        })

    def send_frame(self, jpeg_bytes: bytes):
        """发送 JPEG 画面到 Web 控制台 (base64)."""
        import base64
        self._broadcast({
            "type": "frame",
            "data": base64.b64encode(jpeg_bytes).decode("ascii"),
        })

    def send_stats(self, stats: dict):
        """发送评分/阶段等 HUD 数据到 Web 控制台."""
        self._broadcast({"type": "stats", **stats})

    def send_template(self, template_seq: list[dict]):
        """发送模板关键点序列 (用于右侧模型回放).

        Args:
            template_seq: 模板关键点序列 [{frame, joint_x, joint_y, joint_z, ...}, ...]
        """
        if not self._clients or self._server is None:
            return

        # 转换为 36 值数组序列
        frames = []
        for fdata in template_seq:
            points = []
            for _mp_idx, name in MAJOR_JOINTS:
                points.extend([
                    fdata.get(f"{name}_x", 0.0),
                    fdata.get(f"{name}_y", 0.0),
                    fdata.get(f"{name}_z", 0.0),
                ])
            frames.append(points)

        message = json.dumps({
            "type": "template",
            "frames": frames,
            "joints": [n for _, n in MAJOR_JOINTS],
            "bones": BONE_CONNECTIONS,
        })
        self._last_template_message = message

        for client in list(self._clients):
            try:
                asyncio.run_coroutine_threadsafe(
                    client.send(message), self._loop
                )
            except Exception:
                self._clients.discard(client)

    def send_template_from_json_file(self, json_path: str):
        """从 template_*.json 加载并发送模板."""
        import json as _json
        with open(json_path, "r", encoding="utf-8") as f:
            self.send_template(_json.load(f))

    @property
    def connected(self) -> bool:
        return len(self._clients) > 0

    def _run_server(self):
        """在新线程中运行 asyncio 事件循环."""
        asyncio.set_event_loop(asyncio.new_event_loop())
        self._loop = asyncio.get_event_loop()

        async def handler(websocket):
            self._clients.add(websocket)
            print(f"[TwinWS] 客户端已连接 (共 {len(self._clients)} 个)")
            if self._last_template_message:
                try:
                    await websocket.send(self._last_template_message)
                except Exception:
                    pass
            try:
                async for msg in websocket:
                    try:
                        data = json.loads(msg)
                        if data.get("type") == "set_movement" and self.on_movement_change:
                            self.on_movement_change(data["movement"])
                    except json.JSONDecodeError:
                        pass
            except Exception:
                pass
            finally:
                self._clients.discard(websocket)
                print(f"[TwinWS] 客户端已断开 (剩余 {len(self._clients)} 个)")

        async def serve():
            import websockets
            self._server = await websockets.serve(
                handler, self.host, self.port
            )
            while self._running:
                await asyncio.sleep(0.5)
            self._server.close()

        try:
            self._loop.run_until_complete(serve())
        except Exception as e:
            logger.error(f"[TwinWS] 服务器错误: {e}")


# ─── 便捷工具 ────────────────────────────────────────────────

def landmarks_to_36(landmarks) -> list[float]:
    """将 MediaPipe landmarks (33, 4) 转为 36 个坐标值."""
    points = []
    for mp_idx, _name in MAJOR_JOINTS:
        pt = landmarks[mp_idx, :3]
        points.extend([float(pt[0]), float(pt[1]), float(pt[2])])
    return points
