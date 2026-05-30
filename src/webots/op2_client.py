"""OP2 real-time TCP client — sends human joint angles to Webots OP2 controller.

Connects to the op2_realtime_controller running inside Webots (port 10020)
and streams joint angles computed by GeometricIKSolver / OpenSim IK.

Protocol: 4-byte big-endian length prefix + JSON payload.
"""

import json
import socket
import struct
import time
from typing import Optional


class OP2RealtimeClient:
    """TCP client that sends human joint angles to the Webots OP2 robot.

    Usage:
        client = OP2RealtimeClient()
        client.connect()
        client.send_joint_angles({"knee_angle_r": 150.0, ...})
        client.disconnect()
    """

    def __init__(self, host: str = "localhost", port: int = 10020, timeout: float = 2.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._connected = False
        self._frame_skip = 0
        self._skip_interval = 2  # send every 2nd frame (~15 Hz)
        self._send_count = 0

    def connect(self) -> bool:
        """Connect to the OP2 realtime controller in Webots."""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(self.timeout)
            self._sock.connect((self.host, self.port))
            self._connected = True
            print(f"[OP2Client] Connected to Webots OP2 ({self.host}:{self.port})")
            return True
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            print(f"[OP2Client] Connection failed: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect from Webots."""
        self._connected = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def send_joint_angles(self, joint_angles: dict):
        """Send human joint angles to the OP2 robot.

        The C++ controller handles the human→OP2 retargeting mapping.
        Throttled to send every N frames for performance.

        Args:
            joint_angles: dict from GeometricIKSolver.solve(),
                          e.g. {"knee_angle_r": 150.0, "hip_flexion_r": 140.0, ...}
        """
        if not self._connected:
            return

        self._frame_skip += 1
        if self._frame_skip % self._skip_interval != 0:
            return

        try:
            self._send_count += 1
            self._send_json({
                "type": "set_angles",
                "targets": joint_angles,
                "timestamp": time.time(),
            })
            if self._send_count % 90 == 1:
                print(f"[OP2Client] 已发送 {self._send_count} 帧 | "
                      f"knee_r={joint_angles.get('knee_angle_r', '?'):.0f} "
                      f"hip_r={joint_angles.get('hip_flexion_r', '?'):.0f}")
        except (ConnectionError, OSError, socket.timeout):
            print("[OP2Client] Send failed, disconnected")
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _send_json(self, data: dict):
        """Send JSON message with 4-byte big-endian length prefix."""
        if self._sock is None:
            return
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._sock.sendall(struct.pack(">I", len(body)) + body)
