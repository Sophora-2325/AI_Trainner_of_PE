"""ROBOTIS OP3 Motion Imitation Controller.

Loads a pre-computed motion JSON file and plays it back on the robot
frame-by-frame in the Webots simulation.

Keyboard controls:
  Space — Play / Pause
  R     — Reset to frame 0 (smooth)
  S     — Stop and return to calibrated standing pose (smooth)
  1     — Slow down (0.5x)
  2     — Speed up (2.0x)
  ESC   — Quit

Fall detection: checks if |ay| (vertical axis) drops below 3.0 m/s^2
for 50 consecutive steps, indicating the robot is horizontal.
"""

import json
import math
import os

from controller import Robot, Keyboard


MOTOR_NAMES = [
    "ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL",
    "ArmLowerR", "ArmLowerL", "PelvYR", "PelvYL",
    "PelvR", "PelvL", "LegUpperR", "LegUpperL",
    "LegLowerR", "LegLowerL", "AnkleR", "AnkleL",
    "FootR", "FootL", "Neck", "Head",
]

DEFAULT_MOTION_FILE = "../../motions/squat_imitation.json"

# Velocity limits for safety (rad/s) — limits how fast motors track targets
VELOCITY_LIMITS = {
    "LegUpperR": 3.0, "LegUpperL": 3.0,
    "LegLowerR": 3.0, "LegLowerL": 3.0,
    "AnkleR": 2.0, "AnkleL": 2.0,
    "PelvR": 1.0, "PelvL": 1.0,
}


class OP3ImitationController:
    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())

        # --- Devices (using getDevice to avoid deprecation warnings) ---
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

            # Set velocity limits
            if name in VELOCITY_LIMITS:
                motor.setVelocity(VELOCITY_LIMITS[name])

        self.accelerometer = self.robot.getDevice("Accelerometer")
        self.accelerometer.enable(self.time_step)

        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.time_step)

        # --- Motion state ---
        self.motion_data = None
        self.motor_sequences = {}
        self.num_frames = 0
        self.current_frame = 0
        self.playing = False
        self.play_speed = 1.0
        self.frame_accumulator = 0.0

        # --- Fall detection ---
        self.fall_count = 0
        self.step_count = 0
        self.STARTUP_GRACE_STEPS = 100   # ~3.2s before fall detection activates

        # --- Calibrated standing pose (set after startup) ---
        self.standing_pose = {name: 0.0 for name in MOTOR_NAMES}

        # --- Smooth transition state ---
        self._transition_active = False
        self._transition_start = {}       # sensor positions at transition start
        self._transition_target = {}      # target positions
        self._transition_step = 0
        self._transition_total = 30       # steps for smooth transition

        print(f"[OP3 Controller] Initialized with {len(MOTOR_NAMES)} motors, "
              f"time_step={self.time_step}ms")
        print("[OP3 Controller] Motor limits:")
        for name in ["LegLowerR", "LegUpperR", "AnkleR"]:
            print(f"  {name}: [{self.motor_min[name]:.3f}, {self.motor_max[name]:.3f}] rad")

    # ---- Motion Loading ----

    def load_motion(self, path: str) -> bool:
        try:
            with open(path, "r") as f:
                self.motion_data = json.load(f)
        except FileNotFoundError:
            print(f"[OP3 Controller] Motion file not found: {path}")
            return False
        except json.JSONDecodeError as e:
            print(f"[OP3 Controller] Invalid JSON: {e}")
            return False

        raw = self.motion_data.get("motor_positions", {})
        if not raw:
            print("[OP3 Controller] Motion file has no motor_positions")
            return False

        self.motor_sequences = {}
        for name in MOTOR_NAMES:
            seq = raw.get(name, [])
            self.motor_sequences[name] = seq if seq else [0.0]

        self.num_frames = len(next(iter(self.motor_sequences.values())))
        self.current_frame = 0
        self.frame_accumulator = 0.0

        name = self.motion_data.get("name", "unknown")
        fps = self.motion_data.get("frame_rate", 30)
        duration = self.num_frames / fps if fps > 0 else 0
        print(f"[OP3 Controller] Loaded '{name}': {self.num_frames} frames, "
              f"{fps} FPS, {duration:.1f}s")
        return True

    # ---- Main Loop ----

    def run(self):
        motion_path = os.environ.get("OP3_MOTION_FILE", DEFAULT_MOTION_FILE)
        if os.path.exists(motion_path):
            self.load_motion(motion_path)
        else:
            print(f"[OP3 Controller] No motion file at {motion_path}")

        # --- Startup calibration phase ---
        # Set a mild crouch pose for stability during calibration
        init_pose = {}
        for name in MOTOR_NAMES:
            if name.startswith("LegLower"):
                init_pose[name] = 0.3 if name.endswith("R") else -0.3
            elif name.startswith("LegUpper"):
                init_pose[name] = 0.1 if name.endswith("R") else -0.1
            elif name.startswith("Ankle"):
                init_pose[name] = -0.1 if name.endswith("R") else 0.1
            elif name.startswith("Pelv") and not name.startswith("PelvY"):
                init_pose[name] = 0.0
            elif name.startswith("Foot"):
                init_pose[name] = 0.0
            else:
                init_pose[name] = 0.0
        self._set_all_motors(init_pose)

        print("[OP3 Controller] Calibrating standing pose...")
        for _ in range(150):
            self._step()

        # Read actual sensor positions as calibrated standing pose
        for name in MOTOR_NAMES:
            self.standing_pose[name] = self.position_sensors[name].getValue()
        print("[OP3 Controller] Calibrated standing pose:")
        for name in ["LegLowerR", "LegLowerL", "LegUpperR", "LegUpperL", "AnkleR", "AnkleL"]:
            print(f"  {name}: {self.standing_pose[name]:.3f} rad")

        # Set motors to calibrated standing pose
        self._set_all_motors(self.standing_pose)

        print("[OP3 Controller] Ready. Controls: Space=Play, R=Reset, S=Stop, 1/2=Speed, ESC=Quit")

        # --- Main simulation loop ---
        while self.robot.step(self.time_step) != -1:
            self.step_count += 1
            dt = self.time_step / 1000.0

            # Fall detection (after startup grace period)
            if self.step_count > self.STARTUP_GRACE_STEPS:
                if self._check_fallen():
                    self._recover_from_fall()

            # Keyboard
            self._handle_keyboard()

            # Smooth transition
            if self._transition_active:
                self._step_transition()
            elif self.playing and self.motor_sequences:
                self._advance_playback(dt)
            else:
                self._hold_current_frame()

        print("[OP3 Controller] Shutting down.")

    # ---- Smooth Transition ----

    def _start_transition(self, target: dict[str, float], steps: int = 30):
        """Begin a smooth transition from current sensor positions to target."""
        self._transition_active = True
        self._transition_step = 0
        self._transition_total = steps
        self._transition_target = dict(target)
        self._transition_start = {}
        for name in MOTOR_NAMES:
            self._transition_start[name] = self.position_sensors[name].getValue()

    def _step_transition(self):
        """Advance the smooth transition by one step."""
        self._transition_step += 1
        if self._transition_step >= self._transition_total:
            self._transition_active = False
            self._set_all_motors(self._transition_target)
            return

        t = self._transition_step / self._transition_total
        # Ease-in-out
        t = t * t * (3.0 - 2.0 * t)

        positions = {}
        for name in MOTOR_NAMES:
            start = self._transition_start[name]
            end = self._transition_target.get(name, start)
            positions[name] = start + (end - start) * t

        self._set_all_motors(positions)

    # ---- Playback ----

    def _advance_playback(self, dt: float):
        if self.num_frames == 0:
            return

        fps = self.motion_data.get("frame_rate", 30) if self.motion_data else 30
        frame_duration = 1.0 / fps
        self.frame_accumulator += dt * self.play_speed

        frames_to_advance = int(self.frame_accumulator / frame_duration)
        self.frame_accumulator %= frame_duration

        if frames_to_advance > 0:
            loop = self.motion_data.get("loop", True) if self.motion_data else True
            self.current_frame += frames_to_advance
            if self.current_frame >= self.num_frames:
                if loop:
                    self.current_frame %= self.num_frames
                else:
                    self.current_frame = self.num_frames - 1
                    self.playing = False
                    print("[OP3 Controller] Motion complete. Press Space to replay.")

        self._set_frame(self.current_frame)

    def _set_frame(self, frame_idx: int):
        frame_idx = max(0, min(frame_idx, self.num_frames - 1 if self.num_frames > 0 else 0))
        positions = {}
        for name in MOTOR_NAMES:
            seq = self.motor_sequences.get(name, [0.0])
            pos = seq[frame_idx] if frame_idx < len(seq) else 0.0
            lo = self.motor_min.get(name, -math.pi)
            hi = self.motor_max.get(name, math.pi)
            positions[name] = max(lo, min(hi, pos))
        self._set_all_motors(positions)

    def _hold_current_frame(self):
        if self.motor_sequences:
            self._set_frame(self.current_frame)

    def _set_all_motors(self, positions: dict[str, float]):
        for name, motor in self.motors.items():
            if name in positions:
                motor.setPosition(positions[name])

    # ---- Fall Detection ----

    def _check_fallen(self) -> bool:
        """Check if robot has fallen by monitoring vertical axis.

        When standing: |ay| ~ 9.81 m/s^2 (gravity along body Y)
        When fallen horizontal: |ay| ~ 0 (gravity perpendicular to body Y)
        """
        acc = self.accelerometer.getValues()
        ay = acc[1]  # vertical axis in robot frame (m/s^2)

        if abs(ay) < 3.0:  # robot tilted >70° from vertical
            self.fall_count += 1
        else:
            self.fall_count = max(0, self.fall_count - 1)  # decay

        return self.fall_count > 50

    def _recover_from_fall(self):
        self.playing = False
        self._transition_active = False
        self.fall_count = 0
        print("[OP3 Controller] Fall detected! Returning to standing pose.")
        self._start_transition(self.standing_pose, steps=60)
        self.current_frame = 0
        self.frame_accumulator = 0.0

    # ---- Keyboard ----

    def _handle_keyboard(self):
        key = self.keyboard.getKey()
        while key >= 0:
            if key == ord(" "):
                self.playing = not self.playing
                self._transition_active = False
                state = "PLAYING" if self.playing else "PAUSED"
                print(f"[OP3 Controller] {state} at frame {self.current_frame}")
            elif key == ord("R") or key == ord("r"):
                self.playing = False
                self.current_frame = 0
                self.frame_accumulator = 0.0
                target = {}
                for name in MOTOR_NAMES:
                    seq = self.motor_sequences.get(name, [0.0])
                    target[name] = seq[0] if seq else 0.0
                self._start_transition(target, steps=30)
                print("[OP3 Controller] Resetting to frame 0 (smooth)")
            elif key == ord("S") or key == ord("s"):
                self.playing = False
                self.current_frame = 0
                self.frame_accumulator = 0.0
                self._start_transition(self.standing_pose, steps=40)
                print("[OP3 Controller] Stopping to standing pose (smooth)")
            elif key == ord("1"):
                self.play_speed = max(0.1, self.play_speed * 0.5)
                print(f"[OP3 Controller] Speed: {self.play_speed:.1f}x")
            elif key == ord("2"):
                self.play_speed = min(4.0, self.play_speed * 2.0)
                print(f"[OP3 Controller] Speed: {self.play_speed:.1f}x")
            elif key == Keyboard.LEFT:
                self.playing = False
                self._transition_active = False
                self.current_frame = max(0, self.current_frame - 1)
                self._set_frame(self.current_frame)
            elif key == Keyboard.RIGHT:
                self.playing = False
                self._transition_active = False
                self.current_frame = min(self.num_frames - 1, self.current_frame + 1)
                self._set_frame(self.current_frame)
            key = self.keyboard.getKey()

    def _step(self):
        self.robot.step(self.time_step)


if __name__ == "__main__":
    controller = OP3ImitationController()
    controller.run()
