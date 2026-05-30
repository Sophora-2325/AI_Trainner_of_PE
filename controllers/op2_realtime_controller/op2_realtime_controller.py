"""OP2 Real-Time Mirror Controller.

Receives human joint angles via TCP and drives OP2 motors in real time.

Startup: all motors at 0 (upright neutral). Robot stands still.
When TCP client connects and sends joint angles, robot mirrors the human.

Key controls:
  Space — Pause / Resume mirroring
  S     — Return to neutral standing pose
  ESC   — Quit
"""

import json
import os
import socket
import struct
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from controller import Robot, Keyboard


MOTOR_NAMES = [
    "ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL",
    "ArmLowerR", "ArmLowerL", "PelvYR", "PelvYL",
    "PelvR", "PelvL", "LegUpperR", "LegUpperL",
    "LegLowerR", "LegLowerL", "AnkleR", "AnkleL",
    "FootR", "FootL", "Neck", "Head",
]

TCP_PORT = 10020
STARTUP_GRACE_STEPS = 100
FALL_COUNT_THRESHOLD = 50
FALL_AY_MIN = 3.0


class OP2RealtimeController:
    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())

        self.motors = {}
        self.position_sensors = {}
        self.motor_min = {}
        self.motor_max = {}

        for name in MOTOR_NAMES:
            motor = self.robot.getDevice(name)
            self.motors[name] = motor
            sensor = self.robot.getDevice(name + "S")
            sensor.enable(self.time_step)
            self.position_sensors[name] = sensor
            self.motor_min[name] = motor.getMinPosition()
            self.motor_max[name] = motor.getMaxPosition()
            # Set reasonable velocity for all motors
            motor.setVelocity(1.0)

        self.accelerometer = self.robot.getDevice("Accelerometer")
        self.accelerometer.enable(self.time_step)

        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.time_step)

        self.paused = False
        self.mirroring_active = False
        self.fall_count = 0
        self.step_count = 0

        # TCP shared state
        self.tcp_lock = threading.Lock()
        self.motor_targets = {}   # latest OP2 motor positions from TCP
        self.has_new_data = False

        # TCP server
        self.tcp_running = False
        self.server_sock = None
        self.client_sock = None
        self.tcp_thread = None

        # Transition state
        self.transition_active = False
        self.transition_step = 0
        self.transition_total = 0
        self.transition_start = {}
        self.transition_end = {}

    # ─── TCP Server ────────────────────────────────────────────

    def start_tcp_server(self):
        self.tcp_running = True
        self.tcp_thread = threading.Thread(target=self._tcp_loop, daemon=True)
        self.tcp_thread.start()

    def _tcp_loop(self):
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(("0.0.0.0", TCP_PORT))
        self.server_sock.listen(1)
        self.server_sock.settimeout(0.5)
        print(f"[OP2] TCP listening on port {TCP_PORT}")

        while self.tcp_running:
            if self.client_sock is None:
                try:
                    self.client_sock, addr = self.server_sock.accept()
                    self.client_sock.settimeout(None)  # blocking — wait for data
                    print(f"[OP2] TCP client connected: {addr}")
                    self.mirroring_active = True
                except socket.timeout:
                    continue
                except Exception:
                    continue

            if self.client_sock:
                if not self._receive_message():
                    self.client_sock.close()
                    self.client_sock = None
                    self.mirroring_active = False
                    print("[OP2] TCP client disconnected")

    def _receive_message(self):
        try:
            raw_len = self.client_sock.recv(4)
            if len(raw_len) < 4:
                return False
            msg_len = struct.unpack(">I", raw_len)[0]
            if msg_len > 65536:
                return False
            chunks = []
            received = 0
            while received < msg_len:
                chunk = self.client_sock.recv(min(msg_len - received, 4096))
                if not chunk:
                    return False
                chunks.append(chunk)
                received += len(chunk)
            data = json.loads(b"".join(chunks).decode("utf-8"))
            self._apply_human_angles(data.get("targets", {}))
            return True
        except (ConnectionError, json.JSONDecodeError, socket.timeout):
            return False

    def _apply_human_angles(self, human_angles: dict):
        if not human_angles:
            return
        try:
            from lib.retargeting_mapper import HumanAngleToOP2Mapper
            mapper = HumanAngleToOP2Mapper()
            motor_positions = mapper.map_frame(human_angles)
        except ImportError as e:
            print(f"[OP2] Retarget mapper import failed: {e}")
            return
        except Exception as e:
            print(f"[OP2] Map frame error: {e}")
            return

        with self.tcp_lock:
            self.motor_targets = motor_positions
            self.has_new_data = True

    # ─── Motor helpers ─────────────────────────────────────────

    def _clamp(self, name, value):
        lo = self.motor_min[name]
        hi = self.motor_max[name]
        return max(lo, min(hi, value))

    def _set_motors_direct(self, positions):
        for name, motor in self.motors.items():
            if name in positions:
                motor.setPosition(self._clamp(name, positions[name]))

    def _set_motors_smooth(self, targets):
        for name, motor in self.motors.items():
            if name not in targets:
                continue
            current = self.position_sensors[name].getValue()
            target = targets[name]
            smoothed = current + (target - current) * 0.2
            motor.setPosition(self._clamp(name, smoothed))

    # ─── Transition ────────────────────────────────────────────

    def _start_transition(self, target, steps=30):
        self.transition_active = True
        self.transition_step = 0
        self.transition_total = steps
        self.transition_end = dict(target)
        self.transition_start = {}
        for name in MOTOR_NAMES:
            self.transition_start[name] = self.position_sensors[name].getValue()
        self.mirroring_active = False

    def _step_transition(self):
        self.transition_step += 1
        if self.transition_step >= self.transition_total:
            self._set_motors_direct(self.transition_end)
            self.transition_active = False
            self.mirroring_active = (self.client_sock is not None)
            return

        t = self.transition_step / self.transition_total
        t = t * t * (3.0 - 2.0 * t)
        positions = {}
        for name in MOTOR_NAMES:
            s = self.transition_start[name]
            e = self.transition_end.get(name, s)
            positions[name] = s + (e - s) * t
        self._set_motors_direct(positions)

    def _neutral_pose(self):
        """All motors at 0 = upright standing."""
        return {name: 0.0 for name in MOTOR_NAMES}

    # ─── Fall detection ────────────────────────────────────────

    def _check_fallen(self):
        acc = self.accelerometer.getValues()
        ay = acc[1]
        if abs(ay) < FALL_AY_MIN:
            self.fall_count += 1
        else:
            self.fall_count = max(0, self.fall_count - 1)
        return self.fall_count > FALL_COUNT_THRESHOLD

    def _recover(self):
        self.paused = False
        self.fall_count = 0
        print("[OP2] Fall detected — returning to neutral")
        self._start_transition(self._neutral_pose(), steps=60)

    # ─── Keyboard ──────────────────────────────────────────────

    def _handle_keyboard(self):
        key = self.keyboard.getKey()
        while key >= 0:
            if key == ord(" "):
                self.paused = not self.paused
                print(f"[OP2] {'PAUSED' if self.paused else 'RESUMED'}")
            elif key in (ord("S"), ord("s")):
                self.paused = False
                print("[OP2] Returning to neutral standing pose")
                self._start_transition(self._neutral_pose(), steps=40)
            elif key == 27:
                self.tcp_running = False
            key = self.keyboard.getKey()

    # ─── Main Loop ─────────────────────────────────────────────

    def run(self):
        print(f"[OP2] Realtime controller starting (TCP port {TCP_PORT})")
        print("[OP2] Controls: Space=Pause  S=Stand  ESC=Quit")

        # Step 1: Read world file pose and lock motors to hold it
        for _ in range(10):
            self.robot.step(self.time_step)
        hold_pose = {}
        for name in MOTOR_NAMES:
            hold_pose[name] = self.position_sensors[name].getValue()
        self._set_motors_direct(hold_pose)
        print("[OP2] Locked to world file pose, waiting for TCP...")

        # Step 2: Start TCP and enter main loop
        self.start_tcp_server()

        while self.robot.step(self.time_step) != -1:
            self.step_count += 1

            # Fall detection (after grace period)
            if self.step_count > STARTUP_GRACE_STEPS and self._check_fallen():
                self._recover()

            self._handle_keyboard()

            if self.transition_active:
                self._step_transition()
            elif not self.paused:
                with self.tcp_lock:
                    active = self.mirroring_active and self.has_new_data
                    targets = dict(self.motor_targets) if active else None
                    self.has_new_data = False

                if targets:
                    self._set_motors_smooth(targets)
                    if self.step_count % 90 == 1:
                        print(f"[OP2] frame {self.step_count} | "
                              f"LegLowerR={targets.get('LegLowerR', 0):.3f} "
                              f"LegUpperR={targets.get('LegUpperR', 0):.3f}")

        print("[OP2] Shutting down.")
        if self.server_sock:
            self.server_sock.close()


if __name__ == "__main__":
    OP2RealtimeController().run()
