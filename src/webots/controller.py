"""Webots 骨骼控制器 — 接收关节角度数据并驱动3D骨骼模型.

此模块可以在两种模式下运行:
1. Webots内部控制器模式 (在Webots仿真中运行)
2. 外部控制模式 (从Python主程序通过socket发送关节角度)
"""

import json
import socket
import struct
import threading
from typing import Optional


class SkeletonController:
    """Webots骨骼模型的外部控制器.

    从Python主程序向Webots发送关节目标角度，
    Webots端的控制器接收并执行。

    使用方式:
        ctrl = SkeletonController()
        ctrl.connect()
        ctrl.set_joint_angles({"knee_angle_r": 90.0, ...})
        ctrl.play_reference_motion("squat")
        ctrl.disconnect()
    """

    # 关节名称 → Webots motor 名称映射
    JOINT_MOTOR_MAP = {
        "spine_angle":      "spine_motor",
        "hip_flexion_r":    "hip_flexion_r_motor",
        "hip_flexion_l":    "hip_flexion_l_motor",
        "hip_abduction_r":  "hip_abd_r_motor",
        "hip_abduction_l":  "hip_abd_l_motor",
        "hip_rotation_r":   "hip_rot_r_motor",
        "hip_rotation_l":   "hip_rot_l_motor",
        "knee_angle_r":     "knee_r_motor",
        "knee_angle_l":     "knee_l_motor",
        "ankle_angle_r":    "ankle_r_motor",
        "ankle_angle_l":    "ankle_l_motor",
        "elbow_angle_r":    "elbow_r_motor",
        "elbow_angle_l":    "elbow_l_motor",
        "shoulder_flex_r":  "shoulder_flex_r_motor",
        "shoulder_flex_l":  "shoulder_flex_l_motor",
        "shoulder_abd_r":   "shoulder_abd_r_motor",
        "shoulder_abd_l":   "shoulder_abd_l_motor",
        "neck_angle":       "neck_motor",
    }

    def __init__(self, host: str = "localhost", port: int = 10001):
        self.host = host
        self.port = port
        self._sock: Optional[socket.socket] = None
        self._connected = False
        self._thread: Optional[threading.Thread] = None

    def connect(self) -> bool:
        """连接到Webots仿真."""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(2.0)
            self._sock.connect((self.host, self.port))
            self._connected = True
            print(f"[SkeletonCtrl] 已连接到 Webots ({self.host}:{self.port})")
            return True
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            print(f"[SkeletonCtrl] 连接 Webots 失败: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """断开连接."""
        self._connected = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def set_joint_angles(self, joint_angles: dict):
        """设置所有关节的目标角度.

        Args:
            joint_angles: {joint_name: angle_radians}
        """
        if not self._connected:
            return

        # 转换关节名称到 motor 名称，并将角度转为度
        motor_commands = {}
        for joint_name, angle_rad in joint_angles.items():
            motor_name = self.JOINT_MOTOR_MAP.get(joint_name, joint_name)
            motor_commands[motor_name] = angle_rad  # Webots使用弧度

        try:
            self._send_json({
                "type": "set_angles",
                "targets": motor_commands,
            })
        except Exception:
            self._connected = False

    def play_reference_motion(self, motion_data: dict):
        """发送标准动作的关键帧数据.

        Args:
            motion_data: {frame_idx: {joint: angle, ...}}
        """
        if not self._connected:
            return
        try:
            self._send_json({
                "type": "play_motion",
                "motion": motion_data,
            })
        except Exception:
            self._connected = False

    def reset_to_tpose(self):
        """重置骨骼到T-Pose."""
        t_pose = {
            "spine_angle": 0, "neck_angle": 0,
            "hip_flexion_r": 0, "hip_flexion_l": 0,
            "hip_abduction_r": 0, "hip_abduction_l": 0,
            "hip_rotation_r": 0, "hip_rotation_l": 0,
            "knee_angle_r": 0, "knee_angle_l": 0,
            "ankle_angle_r": 0, "ankle_angle_l": 0,
            "elbow_angle_r": 0, "elbow_angle_l": 0,
            "shoulder_flex_r": 0, "shoulder_flex_l": 0,
            "shoulder_abd_r": 0, "shoulder_abd_l": 0,
        }
        self.set_joint_angles(t_pose)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _send_json(self, data: dict):
        """发送 JSON 消息."""
        if self._sock is None:
            return
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._sock.sendall(struct.pack(">I", len(body)) + body)


# ═══════════════════════════════════════════════════════════════
# Webots 内部控制器 (在Webots仿真进程内运行)
# ═══════════════════════════════════════════════════════════════

class WebotsInternalController:
    """在Webots仿真内部运行的骨骼控制器.

    作为TCP服务器接收外部主程序的关节角度命令，
    在Webots仿真步中驱动 motor 运动。

    使用方式（Webots控制器Python脚本）:
        from src.webots.controller import WebotsInternalController
        ctrl = WebotsInternalController(robot)
        ctrl.run()
    """

    def __init__(self, robot, port: int = 10001):
        """
        Args:
            robot: Webots Robot 实例
            port: 监听端口
        """
        self.robot = robot
        self.port = port
        self.time_step = int(robot.getBasicTimeStep())

        # 初始化所有 motor
        self.motors = {}
        self._init_motors()

        # 目标角度缓存
        self._targets: dict = {}
        self._lock = threading.Lock()

        # TCP 服务器
        self._server: Optional[socket.socket] = None
        self._client: Optional[socket.socket] = None
        self._running = False

    def _init_motors(self):
        """枚举 Robot 节点的所有 motor 设备."""
        for i in range(self.robot.getNumberOfDevices()):
            device = self.robot.getDeviceByIndex(i)
            if device.getNodeType() == self.robot.NODE_TYPES.get("RotationalMotor", 0):
                name = device.getName()
                self.motors[name] = device
                # 设置初始位置为0
                device.setPosition(0.0)

        print(f"[WebotsCtrl] 已初始化 {len(self.motors)} 个电机: {list(self.motors.keys())}")

    def run(self):
        """主循环 — 在 Webots step 中运行."""
        self._running = True
        self._start_server()

        while self.robot.step(self.time_step) != -1 and self._running:
            with self._lock:
                # 将所有 motor 设置到目标位置
                for name, motor in self.motors.items():
                    if name in self._targets:
                        motor.setPosition(self._targets[name])

    def _start_server(self):
        """启动TCP服务器监听外部命令."""
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("0.0.0.0", self.port))
        self._server.listen(1)
        self._server.settimeout(0.1)

        def accept_client():
            while self._running:
                try:
                    conn, addr = self._server.accept()
                    self._client = conn
                    print(f"[WebotsCtrl] 客户端已连接: {addr}")
                    self._handle_client(conn)
                except socket.timeout:
                    continue
                except Exception:
                    break

        t = threading.Thread(target=accept_client, daemon=True)
        t.start()

    def _handle_client(self, conn: socket.socket):
        """处理客户端命令."""
        while self._running:
            try:
                raw_len = conn.recv(4)
                if len(raw_len) < 4:
                    break

                msg_len = struct.unpack(">I", raw_len)[0]
                chunks = []
                received = 0
                while received < msg_len:
                    chunk = conn.recv(min(msg_len - received, 4096))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    received += len(chunk)

                cmd = json.loads(b"".join(chunks).decode("utf-8"))

                if cmd["type"] == "set_angles":
                    with self._lock:
                        self._targets.update(cmd["targets"])
                elif cmd["type"] == "reset":
                    with self._lock:
                        self._targets.clear()

            except (ConnectionError, json.JSONDecodeError):
                break
            except Exception as e:
                print(f"[WebotsCtrl] 错误: {e}")

        self._client = None
