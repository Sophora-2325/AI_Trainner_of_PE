"""Windows端 socket 客户端 — 连接 WSL2 中的 OpenSim IK 服务."""

import json
import socket
import struct
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IKRequest:
    """逆运动学求解请求."""
    type: str = "ik_solve"
    timestamp: float = 0.0
    movement: str = ""
    landmarks: list = field(default_factory=list)  # [[x,y,z,v], ...]


@dataclass
class IKResponse:
    """逆运动学求解响应."""
    type: str = "ik_result"
    timestamp: float = 0.0
    success: bool = False
    joint_angles: dict = field(default_factory=dict)
    ik_error: float = 0.0
    warnings: list = field(default_factory=list)


class OpenSimClient:
    """连接WSL2中OpenSim IK服务的异步客户端.

    使用方式:
        client = OpenSimClient()
        client.connect()
        response = client.solve_ik(landmarks)
        client.disconnect()
    """

    def __init__(
        self,
        host: str = "localhost",
        request_port: int = 5000,
        result_port: int = 5001,
        timeout: float = 0.05,
    ):
        self.host = host
        self.request_port = request_port
        self.result_port = result_port
        self.timeout = timeout
        self._request_sock: Optional[socket.socket] = None
        self._result_sock: Optional[socket.socket] = None
        self._connected = False

    def connect(self) -> bool:
        """建立到WSL2 OpenSim服务的连接."""
        try:
            # 发送请求的 socket
            self._request_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._request_sock.settimeout(self.timeout)
            self._request_sock.connect((self.host, self.request_port))

            # 接收结果的 socket
            self._result_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._result_sock.settimeout(self.timeout)
            self._result_sock.connect((self.host, self.result_port))

            self._connected = True
            return True
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            print(f"[OpenSimClient] 连接失败: {e}")
            self._connected = False
            return False

    def solve_ik(
        self,
        landmarks: np.ndarray,
        movement: str = "",
    ) -> IKResponse:
        """发送关键点到 WSL2，获取解算后的关节角度.

        Args:
            landmarks: (33, 4) MediaPipe关键点
            movement: 动作名称

        Returns:
            IKResponse 包含关节角度字典
        """
        if not self._connected:
            return IKResponse(success=False, warnings=["未连接到OpenSim服务"])

        # 构建请求
        request = IKRequest(
            timestamp=time.time(),
            movement=movement,
            landmarks=landmarks.tolist(),
        )

        try:
            self._send_json(self._request_sock, request.__dict__)
            raw = self._recv_json(self._result_sock)
            response = IKResponse(**raw)
            return response
        except (socket.timeout, ConnectionError, json.JSONDecodeError) as e:
            return IKResponse(success=False, warnings=[str(e)])

    def check_health(self) -> bool:
        """检查 OpenSim 服务是否正常."""
        if not self._connected:
            return False
        try:
            self._send_json(self._request_sock, {"type": "ping"})
            resp = self._recv_json(self._result_sock)
            return resp.get("type") == "pong"
        except Exception:
            return False

    def disconnect(self):
        """断开连接."""
        self._connected = False
        for sock in (self._request_sock, self._result_sock):
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
        self._request_sock = None
        self._result_sock = None

    @staticmethod
    def _send_json(sock: socket.socket, data: dict):
        """发送 JSON 消息：4字节长度 + JSON数据."""
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        sock.sendall(struct.pack(">I", len(body)) + body)

    @staticmethod
    def _recv_json(sock: socket.socket) -> dict:
        """接收 JSON 消息."""
        raw_len = sock.recv(4)
        if len(raw_len) < 4:
            raise ConnectionError("连接已断开")
        msg_len = struct.unpack(">I", raw_len)[0]
        chunks = []
        received = 0
        while received < msg_len:
            chunk = sock.recv(min(msg_len - received, 4096))
            if not chunk:
                raise ConnectionError("连接已断开")
            chunks.append(chunk)
            received += len(chunk)
        return json.loads(b"".join(chunks).decode("utf-8"))
