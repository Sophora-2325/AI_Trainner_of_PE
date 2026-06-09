"""WSL2端 OpenSim IK 服务 — 接收关键点，返回关节角度.

在WSL2中运行: python socket_server.py
"""

import json
import socket
import struct
import threading
import time
import numpy as np
from typing import Optional


class GeometricIKSolver:
    """基于几何关系的快速IK求解器（OpenSim不可用时的回退方案）.

    通过简单的三角几何计算关键关节角度，不依赖OpenSim。
    精度低于OpenSim IK，但延迟极低（<1ms）。
    """

    @staticmethod
    def solve(landmarks: np.ndarray) -> dict:
        """从33个MediaPipe关键点计算主要关节角度."""
        pts = landmarks[:, :3]  # (33, 3)

        angles = {}

        # 膝关节角度
        angles["knee_angle_r"] = _calc_joint_angle(pts, 24, 26, 28)
        angles["knee_angle_l"] = _calc_joint_angle(pts, 23, 25, 27)

        # 髋关节角度（躯干-大腿）
        # 使用肩-髋-膝三点
        hip_r = _calc_joint_angle(pts, 12, 24, 26)
        hip_l = _calc_joint_angle(pts, 11, 23, 25)
        angles["hip_flexion_r"] = 180.0 - hip_r
        angles["hip_flexion_l"] = 180.0 - hip_l

        # 踝关节角度
        angles["ankle_angle_r"] = _calc_joint_angle(pts, 26, 28, 32)
        angles["ankle_angle_l"] = _calc_joint_angle(pts, 25, 27, 31)

        # 肘关节角度
        angles["elbow_angle_r"] = _calc_joint_angle(pts, 12, 14, 16)
        angles["elbow_angle_l"] = _calc_joint_angle(pts, 11, 13, 15)

        # 肩关节角度
        angles["shoulder_angle_r"] = _calc_joint_angle(pts, 14, 12, 24)
        angles["shoulder_angle_l"] = _calc_joint_angle(pts, 13, 11, 23)

        # 腰椎伸展角度（躯干前倾）
        shoulder_mid = (pts[11] + pts[12]) / 2.0
        hip_mid = (pts[23] + pts[24]) / 2.0
        vertical = np.array([0, -1, 0])
        torso_vec = shoulder_mid - hip_mid
        torso_vec_norm = np.linalg.norm(torso_vec)
        if torso_vec_norm > 1e-9:
            torso_vec = torso_vec / torso_vec_norm
            lumbar_angle = np.degrees(np.arccos(np.clip(
                np.dot(torso_vec, vertical), -1, 1
            )))
            angles["lumbar_extension"] = 180.0 - lumbar_angle

        # 颈部偏航角度（头部左右转动）
        mid_shoulder = (pts[11] + pts[12]) / 2.0
        nose_rel = pts[0] - mid_shoulder
        neck_yaw = np.degrees(np.arctan2(nose_rel[0], -nose_rel[2] + 1e-9))
        angles["neck_yaw"] = float(neck_yaw)

        # 头部俯仰角度（头部上下倾斜）
        mid_ear = (pts[7] + pts[8]) / 2.0
        facing_vec = pts[0] - mid_ear
        horiz_mag = np.linalg.norm([facing_vec[0], facing_vec[2]])
        head_pitch = np.degrees(np.arctan2(facing_vec[1], horiz_mag + 1e-9))
        angles["head_pitch"] = float(head_pitch)

        # 肩外展角度（手臂侧向抬起）
        angles["shoulder_abduction_r"] = _calc_shoulder_abduction(pts, 12, 14)
        angles["shoulder_abduction_l"] = _calc_shoulder_abduction(pts, 11, 13)

        # 膝外翻角度（膝-踝连线与垂直面的偏离）
        angles["knee_valgus_angle_r"] = _calc_valgus(pts, 24, 26, 28)
        angles["knee_valgus_angle_l"] = _calc_valgus(pts, 23, 25, 27)

        # 髋外展角度
        angles["hip_abduction_r"] = _calc_abduction(pts, 24, 26, 12)
        angles["hip_abduction_l"] = _calc_abduction(pts, 23, 25, 11)

        # 躯干旋转角度（肩部连线相对于骨盆连线的水平旋转）
        shoulder_mid = (pts[11] + pts[12]) / 2.0
        hip_mid = (pts[23] + pts[24]) / 2.0
        shoulder_vec = pts[12] - pts[11]  # right shoulder → left shoulder
        hip_vec = pts[24] - pts[23]        # right hip → left hip
        shoulder_yaw = np.degrees(np.arctan2(shoulder_vec[0], shoulder_vec[2] + 1e-9))
        hip_yaw = np.degrees(np.arctan2(hip_vec[0], hip_vec[2] + 1e-9))
        angles["torso_rotation"] = shoulder_yaw - hip_yaw

        # 对称性指标
        angles["knee_symmetry"] = abs(angles["knee_angle_r"] - angles["knee_angle_l"])
        angles["hip_symmetry"] = abs(angles["hip_flexion_r"] - angles["hip_flexion_l"])

        # 估算的力矩（相对值，非牛顿米）
        angles["knee_torque_r"] = _estimate_torque(pts, 24, 26, 28, angles.get("knee_angle_r", 180))

        return angles


class OpenSimIKSolver:
    """基于 OpenSim Python API 的精确 IK 求解器.

    完整流程:
      1. 加载 .osim 模型
      2. 将 MediaPipe landmarks 映射为 OpenSim MarkerSet
      3. 配置 IKTool (权重、精度)
      4. 运行 IK 求解 → 输出关节角度

    当 OpenSim API 不可用时，回退到 GeometricIKSolver。
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None
        self._initialized = False
        self._marker_weights: dict[str, float] = {}
        self._ik_tool = None

    def initialize(self) -> bool:
        """加载 OpenSim 模型并初始化 IK 工具."""
        try:
            import opensim
            self._model = opensim.Model(self.model_path)
            self._model.initSystem()

            # 获取模型中的 marker 名称，设置权重
            marker_set = self._model.getMarkerSet()
            for i in range(marker_set.getSize()):
                marker = marker_set.get(i)
                name = marker.getName()
                # 躯干和骨盆标记点权重更高
                if any(k in name.lower() for k in ["torso", "pelvis", "sacrum", "sternum"]):
                    self._marker_weights[name] = 10.0
                else:
                    self._marker_weights[name] = 1.0

            self._initialized = True
            print(f"[IKServer] OpenSim 模型已加载: {len(self._marker_weights)} 个标记点")
            return True
        except ImportError:
            print("[IKServer] OpenSim Python API 不可用，将使用几何方法回退")
            return False
        except Exception as e:
            print(f"[IKServer] OpenSim 模型加载失败: {e}")
            return False

    def solve(self, landmarks: np.ndarray) -> dict:
        """使用 OpenSim IK 求解关节角度.

        Args:
            landmarks: (33, 4) MediaPipe world landmarks

        Returns:
            {joint_name: angle_deg} 关节角度字典
        """
        if not self._initialized:
            return GeometricIKSolver.solve(landmarks)

        try:
            import opensim

            # 1. 将 landmarks 映射为 OpenSim marker 位置
            marker_positions = self._landmarks_to_markers(landmarks)

            # 2. 构建 IK 工具
            ik_tool = opensim.InverseKinematicsTool()
            ik_tool.setModel(self._model)

            # 创建临时 marker 数据文件 (.trc)
            import tempfile
            trc_path = tempfile.mktemp(suffix=".trc")
            self._write_trc(trc_path, marker_positions)
            ik_tool.setMarkerDataFileName(trc_path)

            # 设置输出
            mot_path = tempfile.mktemp(suffix=".mot")
            ik_tool.setOutputMotionFileName(mot_path)

            # 设置时间范围（单帧）
            ik_tool.setStartTime(0.0)
            ik_tool.setEndTime(0.01)

            # 3. 运行 IK
            ik_tool.run()

            # 4. 读取结果
            result = self._read_mot_result(mot_path)

            # 清理临时文件
            try:
                os.unlink(trc_path)
                os.unlink(mot_path)
            except Exception:
                pass

            return result if result else GeometricIKSolver.solve(landmarks)

        except Exception as e:
            print(f"[IKServer] OpenSim IK 求解失败: {e}")
            return GeometricIKSolver.solve(landmarks)

    def _landmarks_to_markers(self, landmarks: np.ndarray) -> dict:
        """将 MediaPipe 33 关键点映射到 OpenSim marker 位置.

        使用简化的映射表（MediaPipe ID → OpenSim marker name）。
        完整的映射需要根据具体 .osim 模型的 marker set 调整。
        """
        pts = landmarks[:, :3]

        # MediaPipe landmark → OpenSim marker 近似映射
        # 实际使用时需要校准：将世界坐标缩放到模型坐标系
        mapping = {
            0:  "nose",
            7:  "L_ear",      8:  "R_ear",
            11: "L_shoulder", 12: "R_shoulder",
            13: "L_elbow",    14: "R_elbow",
            15: "L_wrist",    16: "R_wrist",
            23: "L_hip",      24: "R_hip",
            25: "L_knee",     26: "R_knee",
            27: "L_ankle",    28: "R_ankle",
            29: "L_heel",     30: "R_heel",
            31: "L_toe",      32: "R_toe",
        }

        # 缩放因子：MediaPipe 世界坐标 → OpenSim 模型坐标（米）
        # MediaPipe 世界坐标以髋部中心为原点，单位大致为米
        scale = 1.0

        marker_pos = {}
        for mp_id, marker_name in mapping.items():
            if marker_name in self._marker_weights:
                marker_pos[marker_name] = pts[mp_id] * scale

        return marker_pos

    @staticmethod
    def _write_trc(filepath: str, marker_positions: dict):
        """写入 TRC 格式的 marker 轨迹文件（单帧）."""
        marker_names = list(marker_positions.keys())
        n_markers = len(marker_names)

        with open(filepath, "w") as f:
            # Header
            f.write("PathFileType\t4\t(X/Y/Z)\t临时.trc\n")
            f.write(f"DataRate\tCameraRate\tNumFrames\tNumMarkers\t"
                    f"Units\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
            f.write(f"100.0\t100.0\t1\t{n_markers}\tm\t100.0\t1\t1\n")

            # Column headers
            f.write("Frame#\tTime\t")
            for name in marker_names:
                f.write(f"{name}_X\t{name}_Y\t{name}_Z\t")
            f.write("\n")

            # Data row (frame 1, time 0)
            f.write("1\t0.0000\t")
            for name in marker_names:
                pos = marker_positions[name]
                f.write(f"{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\t")
            f.write("\n")

    @staticmethod
    def _read_mot_result(filepath: str) -> dict:
        """读取 OpenSim IK 输出的 .mot 文件，提取关节角度."""
        import re
        angles = {}

        try:
            with open(filepath) as f:
                lines = f.readlines()

            # 查找数据行
            in_data = False
            headers = []
            for line in lines:
                line = line.strip()
                if line.lower().startswith("time"):
                    headers = line.split()
                    in_data = True
                    continue
                if in_data and line:
                    values = line.split()
                    for i, name in enumerate(headers):
                        if name.lower() != "time" and i < len(values):
                            # 将 OpenSim 弧度转为度
                            try:
                                angles[name] = float(np.degrees(float(values[i])))
                            except ValueError:
                                pass
                    break  # 只读第一帧

        except Exception as e:
            print(f"[IKServer] 读取 .mot 失败: {e}")

        return angles


class OpenSimServer:
    """TCP 服务器，接收 Windows 端的 IK 请求.

    协议:
      请求: 4字节长度 + JSON {type, timestamp, movement, landmarks}
      响应: 4字节长度 + JSON {type, timestamp, success, joint_angles, ik_error, warnings}
    """

    def __init__(
        self,
        model_path: str = "",
        request_port: int = 5000,
        result_port: int = 5001,
    ):
        self.model_path = model_path
        self.request_port = request_port
        self.result_port = result_port
        self.ik_solver = OpenSimIKSolver(model_path) if model_path else None
        self.geometric_solver = GeometricIKSolver()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # 存储最近一次求解结果
        self._latest_result: dict = {}
        self._result_lock = threading.Lock()

    def start(self):
        """启动服务."""
        if self.model_path:
            ok = self.ik_solver.initialize()
            if not ok:
                print("[IKServer] 使用几何IK回退方案")

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[IKServer] 服务已启动 — 请求端口:{self.request_port} 结果端口:{self.result_port}")

    def stop(self):
        self._running = False

    def _run(self):
        """主循环：接收请求并处理."""
        # 启动请求监听
        req_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        req_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        req_sock.bind(("0.0.0.0", self.request_port))
        req_sock.listen(1)
        req_sock.settimeout(0.5)

        # 启动结果监听
        res_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        res_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        res_sock.bind(("0.0.0.0", self.result_port))
        res_sock.listen(1)
        res_sock.settimeout(0.5)

        print("[IKServer] 等待连接...")

        req_conn = None
        res_conn = None

        while self._running:
            try:
                if req_conn is None:
                    req_conn, _ = req_sock.accept()
                    print("[IKServer] 请求连接已建立")
                if res_conn is None:
                    res_conn, _ = res_sock.accept()
                    print("[IKServer] 结果连接已建立")

                if req_conn and res_conn:
                    self._handle_client(req_conn, res_conn)
                    req_conn = None
                    res_conn = None
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[IKServer] 错误: {e}")
                if req_conn:
                    req_conn.close()
                    req_conn = None
                if res_conn:
                    res_conn.close()
                    res_conn = None

        req_sock.close()
        res_sock.close()

    def _handle_client(self, req_conn: socket.socket, res_conn: socket.socket):
        """处理单个客户端连接."""
        while self._running:
            try:
                raw_len = req_conn.recv(4)
                if len(raw_len) < 4:
                    break

                msg_len = struct.unpack(">I", raw_len)[0]
                chunks = []
                received = 0
                while received < msg_len:
                    chunk = req_conn.recv(min(msg_len - received, 4096))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    received += len(chunk)

                request = json.loads(b"".join(chunks).decode("utf-8"))

                if request.get("type") == "ping":
                    resp = {"type": "pong", "timestamp": time.time()}
                    self._send_json(res_conn, resp)
                    continue

                # 执行 IK 求解
                landmarks = np.array(request["landmarks"], dtype=np.float32)

                if self.ik_solver and self.ik_solver._initialized:
                    joint_angles = self.ik_solver.solve(landmarks)
                else:
                    joint_angles = self.geometric_solver.solve(landmarks)

                response = {
                    "type": "ik_result",
                    "timestamp": time.time(),
                    "success": True,
                    "joint_angles": joint_angles,
                    "ik_error": 0.0,
                    "warnings": [],
                }

                self._send_json(res_conn, response)

            except (ConnectionError, json.JSONDecodeError):
                break
            except Exception as e:
                print(f"[IKServer] 处理请求时出错: {e}")

    @staticmethod
    def _send_json(sock: socket.socket, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        sock.sendall(struct.pack(">I", len(body)) + body)


# ─── 几何计算辅助函数 ─────────────────────────────────────────

def _calc_joint_angle(pts: np.ndarray, a: int, b: int, c: int) -> float:
    """三点法计算关节角度 (a-b-c，以b为顶点)."""
    ba = pts[a] - pts[b]
    bc = pts[c] - pts[b]
    dot = np.dot(ba, bc)
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm < 1e-9:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0))))


def _calc_valgus(pts: np.ndarray, hip: int, knee: int, ankle: int) -> float:
    """计算膝外翻角度."""
    hip_p = pts[hip, :2]
    knee_p = pts[knee, :2]
    ankle_p = pts[ankle, :2]

    thigh = knee_p - hip_p
    shank = ankle_p - knee_p

    norm_thigh = np.linalg.norm(thigh)
    norm_shank = np.linalg.norm(shank)

    if norm_thigh < 1e-9 or norm_shank < 1e-9:
        return 0.0

    thigh_unit = thigh / norm_thigh
    shank_unit = shank / norm_shank

    # 大腿和小腿在冠状面的投影偏差
    cross = np.cross(
        np.array([thigh_unit[0], thigh_unit[1], 0]),
        np.array([shank_unit[0], shank_unit[1], 0]),
    )
    return float(np.degrees(np.arcsin(np.clip(cross[2], -1.0, 1.0))))


def _calc_abduction(pts: np.ndarray, hip: int, knee: int, shoulder: int) -> float:
    """计算髋外展角度."""
    hip_p = pts[hip]
    knee_p = pts[knee]
    shoulder_p = pts[shoulder]

    thigh = knee_p - hip_p
    vertical = np.array([0, -1, 0])

    norm_thigh = np.linalg.norm(thigh)
    if norm_thigh < 1e-9:
        return 0.0
    thigh_unit = thigh / norm_thigh

    # 大腿与垂直线的夹角在冠状面的分量
    angle = np.degrees(np.arccos(np.clip(np.dot(thigh_unit, vertical), -1, 1)))
    return float(angle)


def _calc_shoulder_abduction(pts: np.ndarray, shoulder: int, elbow: int) -> float:
    """计算肩外展角度（手臂在冠状面内离开身体垂线的角度）."""
    shoulder_p = pts[shoulder]
    elbow_p = pts[elbow]
    arm_vec = elbow_p - shoulder_p
    body_vertical = np.array([0, -1, 0])

    norm_arm = np.linalg.norm(arm_vec)
    if norm_arm < 1e-9:
        return 0.0
    arm_unit = arm_vec / norm_arm

    cos_angle = np.clip(np.dot(arm_unit, body_vertical), -1, 1)
    return float(np.degrees(np.arccos(cos_angle)))


def _estimate_torque(
    pts: np.ndarray, hip: int, knee: int, ankle: int, knee_angle: float
) -> float:
    """估算膝关节力矩（相对值，非物理单位）."""
    thigh_vec = pts[knee] - pts[hip]
    leg_vec = pts[ankle] - pts[knee]
    thigh_len = np.linalg.norm(thigh_vec)
    leg_len = np.linalg.norm(leg_vec)

    if thigh_len < 1e-9 or leg_len < 1e-9:
        return 0.0

    # 力矩随膝角偏离180°而增大
    deviation = abs(180.0 - knee_angle)
    # 简化的力矩模型：力矩 ∝ sin(偏离角) × 腿长
    return float(np.sin(np.radians(deviation)) * leg_len * 10.0)


# ─── 启动入口 ─────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    model_path = os.environ.get(
        "OPENSIM_MODEL_PATH",
        "/mnt/d/Project_of_Teacher_Feng/WorkPlace/opensim_models/Rajagopal2015.osim",
    )
    server = OpenSimServer(model_path=model_path)
    server.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[IKServer] 正在关闭...")
        server.stop()
