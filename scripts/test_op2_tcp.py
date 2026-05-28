#!/usr/bin/env python3
"""Test script for OP2 real-time TCP controller.

Sends synthetic squat-like joint angles to the Webots OP2 robot
to verify the TCP connection and motor mapping.

Usage:
  python scripts/test_op2_tcp.py            # default: localhost:10020
  python scripts/test_op2_tcp.py --host 127.0.0.1 --port 10020 --movement squat
"""

import argparse
import json
import math
import socket
import struct
import time
import sys


class TestOP2Sender:
    def __init__(self, host: str = "localhost", port: int = 10020):
        self.host = host
        self.port = port
        self._sock = None

    def connect(self) -> bool:
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(3.0)
            self._sock.connect((self.host, self.port))
            print(f"Connected to OP2 controller at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def disconnect(self):
        if self._sock:
            self._sock.close()
            self._sock = None

    def send(self, targets: dict):
        if self._sock is None:
            return
        msg = json.dumps({"type": "set_angles", "targets": targets})
        body = msg.encode("utf-8")
        self._sock.sendall(struct.pack(">I", len(body)) + body)

    def send_standing_pose(self):
        """Send standing pose (all angles at neutral)."""
        standing = {
            "knee_angle_r": 180.0, "knee_angle_l": 180.0,
            "hip_flexion_r": 180.0, "hip_flexion_l": 180.0,
            "ankle_angle_r": 90.0, "ankle_angle_l": 90.0,
            "elbow_angle_r": 180.0, "elbow_angle_l": 180.0,
            "shoulder_angle_r": 0.0, "shoulder_angle_l": 0.0,
            "hip_abduction_r": 0.0, "hip_abduction_l": 0.0,
            "lumbar_extension": 0.0,
        }
        self.send(standing)
        print("Sent: standing pose")

    def send_squat_frame(self, phase: float):
        """Generate a synthetic squat frame at the given phase [0, 1].

        phase=0 → standing upright
        phase=0.5 → bottom of squat
        phase=1.0 → back to standing
        """
        if phase < 0.15:
            t = 0
        elif phase < 0.5:
            t = (phase - 0.15) / 0.35
            t = t * t  # ease-in
        elif phase < 0.57:
            t = 1.0
        else:
            t = (phase - 0.57) / 0.43
            t = 1.0 - (1.0 - t) * (1.0 - t)  # ease-out

        knee_angle = 180.0 - t * 85.0
        hip_angle  = 180.0 - t * 120.0
        ankle_angle = 90.0 - t * 20.0
        lumbar = -t * 5.0

        targets = {
            "knee_angle_r": round(knee_angle, 2),
            "knee_angle_l": round(knee_angle, 2),
            "hip_flexion_r": round(hip_angle, 2),
            "hip_flexion_l": round(hip_angle, 2),
            "ankle_angle_r": round(ankle_angle, 2),
            "ankle_angle_l": round(ankle_angle, 2),
            "elbow_angle_r": 180.0,
            "elbow_angle_l": 180.0,
            "shoulder_angle_r": 0.0,
            "shoulder_angle_l": 0.0,
            "hip_abduction_r": 0.0,
            "hip_abduction_l": 0.0,
            "lumbar_extension": round(lumbar, 2),
        }
        self.send(targets)


def main():
    parser = argparse.ArgumentParser(description="Test OP2 TCP controller")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=10020)
    parser.add_argument("--movement", default="squat",
                        choices=["squat", "standing", "arm_wave"])
    parser.add_argument("--cycles", type=int, default=5,
                        help="Number of squat cycles")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second for playback")
    args = parser.parse_args()

    sender = TestOP2Sender(args.host, args.port)
    if not sender.connect():
        print("\nMake sure:")
        print("  1. Webots is running with AI_Trainer_Robot.wbt")
        print("  2. The robot uses 'op2_realtime_controller'")
        sys.exit(1)

    try:
        if args.movement == "standing":
            print("Sending standing pose...")
            sender.send_standing_pose()
            time.sleep(2)
        elif args.movement == "squat":
            frame_duration = 1.0 / args.fps
            print(f"Playing {args.cycles} squat cycles at {args.fps} FPS...")
            print("Press Ctrl+C to stop")

            for cycle in range(args.cycles):
                for frame in range(args.fps * 2):
                    phase = (frame / (args.fps * 2)) % 1.0
                    sender.send_squat_frame(phase)
                    time.sleep(frame_duration)

            print("Returning to standing...")
            for i in range(30):
                t = i / 30.0
                sender.send_squat_frame(1.0 - t * (1.0 - 0.0))
                time.sleep(0.05)
        elif args.movement == "arm_wave":
            print("Waving arms...")
            for cycle in range(3):
                for frame in range(60):
                    t = frame / 60.0 * 2 * math.pi
                    targets = {
                        "knee_angle_r": 180.0, "knee_angle_l": 180.0,
                        "hip_flexion_r": 180.0, "hip_flexion_l": 180.0,
                        "ankle_angle_r": 90.0, "ankle_angle_l": 90.0,
                        "elbow_angle_r": 180.0 - 30 * math.sin(t),
                        "elbow_angle_l": 180.0 - 30 * math.sin(t + math.pi),
                        "shoulder_angle_r": 40 * math.sin(t),
                        "shoulder_angle_l": 40 * math.sin(t + math.pi),
                        "hip_abduction_r": 0.0, "hip_abduction_l": 0.0,
                        "lumbar_extension": 0.0,
                    }
                    sender.send(targets)
                    time.sleep(0.033)

    except KeyboardInterrupt:
        print("\nStopped.")
        sender.send_standing_pose()
    finally:
        sender.disconnect()


if __name__ == "__main__":
    main()
