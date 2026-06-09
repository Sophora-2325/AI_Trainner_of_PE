"""IMU sensor fusion — complementary filter + Kalman filter for joint angle smoothing.

Provides:
  - IMUSample: structured IMU reading (accel_xyz + gyro_xyz)
  - ComplementaryFilter: accelerometer/gyroscope fusion for orientation estimation
  - JointAngleKalmanFilter: per-joint Kalman filter for angle smoothing
  - VirtualIMU: derives pseudo-IMU readings from MediaPipe landmark sequences

These are the theoretical foundations described in the research proposal (Method 1):
  complementary filter: low-freq accel + high-freq gyro weighted fusion
  Kalman filter: state prediction + measurement update for optimal estimation
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ─── Data structures ────────────────────────────────────────────

@dataclass
class IMUSample:
    """Single IMU reading from a wearable sensor node."""
    timestamp: float
    accel_x: float       # m/s^2
    accel_y: float
    accel_z: float
    gyro_x: float        # rad/s
    gyro_y: float
    gyro_z: float

    @property
    def accel(self) -> np.ndarray:
        return np.array([self.accel_x, self.accel_y, self.accel_z])

    @property
    def gyro(self) -> np.ndarray:
        return np.array([self.gyro_x, self.gyro_y, self.gyro_z])


# ─── Complementary Filter ───────────────────────────────────────

class ComplementaryFilter:
    """Fuse accelerometer (low-freq, stable) with gyroscope (high-freq, responsive).

    Pitch/Roll from accelerometer: orientation relative to gravity (no drift, noisy).
    Integration of gyro angular velocity: orientation change (smooth, drifts over time).
    The filter blends them with coefficient alpha.

    angle = alpha * (angle + gyro * dt) + (1 - alpha) * accel_angle
    """

    def __init__(self, alpha: float = 0.98):
        """
        Args:
            alpha: gyro trust coefficient (0~1).
                   Higher = more responsive to motion, drift-prone.
                   Lower = more stable, less responsive.
                   0.98 is typical for human motion tracking.
        """
        self.alpha = alpha
        self._pitch = 0.0
        self._roll = 0.0
        self._initialized = False

    def update(self, sample: IMUSample, dt: float) -> tuple[float, float]:
        """Process one IMU sample, return (pitch_deg, roll_deg)."""
        # Accelerometer-based angles
        ax, ay, az = sample.accel_x, sample.accel_y, sample.accel_z
        accel_norm = math.sqrt(ax * ax + ay * ay + az * az)
        if accel_norm < 1e-9:
            return self._pitch, self._roll

        accel_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
        accel_roll = math.degrees(math.atan2(ay, az))

        if not self._initialized:
            self._pitch = accel_pitch
            self._roll = accel_roll
            self._initialized = True
            return self._pitch, self._roll

        # Gyro integration
        gyro_pitch = self._pitch + math.degrees(sample.gyro_y) * dt
        gyro_roll = self._roll + math.degrees(sample.gyro_x) * dt

        # Complementary fusion
        self._pitch = self.alpha * gyro_pitch + (1.0 - self.alpha) * accel_pitch
        self._roll = self.alpha * gyro_roll + (1.0 - self.alpha) * accel_roll

        return self._pitch, self._roll

    def reset(self):
        self._initialized = False


# ─── Joint Angle Kalman Filter ─────────────────────────────────

class JointAngleKalmanFilter:
    """1D Kalman filter for a single joint angle trajectory.

    State vector: [angle, angular_velocity]^T
    Observation:   [angle]  (from GeometricIKSolver or OpenSim IK)

    Constant-velocity process model:
        angle_{t+1} = angle_t + vel_t * dt + noise
        vel_{t+1}   = vel_t + noise

    This produces a smooth, delay-compensated angle estimate suitable
    for driving robot motors without jitter.
    """

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        process_noise: float = 0.01,
        measurement_noise: float = 1.0,
    ):
        self.dt = dt

        # State: [angle, angular_velocity]
        self.x = np.zeros((2, 1))

        # State transition matrix
        self.F = np.array([[1.0, dt],
                           [0.0, 1.0]])

        # Observation matrix (only angle observed)
        self.H = np.array([[1.0, 0.0]])

        # Process noise covariance
        q_angle = process_noise
        q_vel = process_noise * 0.1
        self.Q = np.array([
            [q_angle * dt**3 / 3, q_angle * dt**2 / 2],
            [q_angle * dt**2 / 2, q_angle * dt],
        ]) + np.array([
            [q_vel * dt**3 / 3, q_vel * dt**2 / 2],
            [q_vel * dt**2 / 2, q_vel * dt],
        ])

        # Measurement noise covariance
        self.R = np.array([[measurement_noise]])

        # State covariance
        self.P = np.eye(2) * 1000.0

        self._initialized = False

    def update(self, measurement: float) -> float:
        """Process one angle measurement, return filtered angle estimate."""
        z = np.array([[measurement]])

        if not self._initialized:
            self.x[0, 0] = measurement
            self.x[1, 0] = 0.0
            self._initialized = True
            return measurement

        # ── Predict ──
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # ── Update ──
        y = z - self.H @ x_pred                      # innovation
        S = self.H @ P_pred @ self.H.T + self.R      # innovation covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S)     # Kalman gain

        self.x = x_pred + K @ y
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        return float(self.x[0, 0])

    def reset(self):
        self._initialized = False


class MultiJointKalmanFilter:
    """Kalman filter bank: one JointAngleKalmanFilter per joint name.

    Usage:
        kf = MultiJointKalmanFilter()
        filtered = kf.update({"knee_angle_r": 150.0, "hip_flexion_r": 140.0})
    """

    def __init__(self, dt: float = 1.0 / 30.0, process_noise: float = 0.01):
        self.dt = dt
        self.process_noise = process_noise
        self._filters: dict[str, JointAngleKalmanFilter] = {}

    def update(self, joint_angles: dict[str, float]) -> dict[str, float]:
        """Filter a frame of joint angles, return smoothed dict."""
        result = {}
        for name, angle in joint_angles.items():
            if name not in self._filters:
                self._filters[name] = JointAngleKalmanFilter(
                    dt=self.dt, process_noise=self.process_noise
                )
            result[name] = self._filters[name].update(angle)
        return result

    def reset(self):
        for kf in self._filters.values():
            kf.reset()
        self._filters.clear()


# ─── Virtual IMU ────────────────────────────────────────────────

class VirtualIMU:
    """Derive pseudo-IMU readings from MediaPipe landmark sequences.

    Treats a sequence of world landmarks as sensor data:
      - "accelerometer": second derivative of landmark position
      - "gyroscope": angular velocity of limb segments

    This allows testing IMU-based algorithms (complementary filter, 1D-CNN+LSTM)
    without physical IMU hardware. When real IMUs become available, swap
    VirtualIMU → real IMU data source.
    """

    def __init__(self, frame_rate: float = 30.0):
        self.frame_rate = frame_rate
        self.dt = 1.0 / frame_rate
        self._prev_velocities: Optional[np.ndarray] = None
        self._prev_positions: Optional[np.ndarray] = None

    def landmarks_to_imu(
        self, landmarks: np.ndarray, timestamp: float
    ) -> list[IMUSample]:
        """Convert a frame of (33, 3) world landmarks to virtual IMU readings.

        Each limb segment produces one IMUSample. Simplified to a single
        pelvis-mounted IMU for the research proposal IMU path.

        Returns:
            List with one IMUSample (pelvis proxy IMU).
        """
        pts = landmarks[:, :3]

        # Use mid-hip as pelvis IMU location
        hip_center = (pts[23] + pts[24]) / 2.0

        if self._prev_positions is None:
            self._prev_positions = hip_center.copy()
            self._prev_velocities = np.zeros(3)
            return [IMUSample(timestamp, 0, -9.81, 0, 0, 0, 0)]

        # Velocity = first derivative of position
        velocity = (hip_center - self._prev_positions) / self.dt

        # Acceleration = second derivative (gravity removed for pure motion accel)
        acceleration = (velocity - self._prev_velocities) / self.dt

        # Angular velocity from shoulder-hip segment direction change
        shoulder_center = (pts[11] + pts[12]) / 2.0
        torso_vec = shoulder_center - hip_center
        torso_dir = torso_vec / (np.linalg.norm(torso_vec) + 1e-9)

        gyro = np.zeros(3)  # simplified — full IMU would attach to each segment
        if hasattr(self, '_prev_torso_dir'):
            cross = np.cross(self._prev_torso_dir, torso_dir)
            gyro = cross / self.dt
        self._prev_torso_dir = torso_dir

        self._prev_positions = hip_center.copy()
        self._prev_velocities = velocity.copy()

        return [IMUSample(
            timestamp=timestamp,
            accel_x=float(acceleration[0]),
            accel_y=float(acceleration[1]),
            accel_z=float(acceleration[2]),
            gyro_x=float(gyro[0]),
            gyro_y=float(gyro[1]),
            gyro_z=float(gyro[2]),
        )]

    def reset(self):
        self._prev_velocities = None
        self._prev_positions = None
