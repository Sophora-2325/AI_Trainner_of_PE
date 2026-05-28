"""End-to-end pipeline: video/template → OP3 motor keyframes → JSON.

Orchestrates the full processing chain:
  1. Extract human poses (from video or template)
  2. Compute biomechanical joint angles
  3. Retarget to OP3 motor positions
  4. Apply balance corrections
  5. Save as MotionKeyframes JSON
"""

import os
import sys
import json
from typing import Optional

import numpy as np

from .retargeting_mapper import HumanAngleToOP3Mapper
from .balance_controller import BalanceController
from .motion_keyframes import MotionKeyframes


class PipelineRunner:
    """Run the full video/template → OP3 motion pipeline."""

    def __init__(
        self,
        motor_limits: Optional[dict[str, tuple[float, float]]] = None,
        balance_enabled: bool = True,
        time_scale: float = 1.0,
    ):
        """
        Args:
            motor_limits: Override default motor limits
            balance_enabled: Whether to apply balance corrections
            time_scale: Playback speed factor (>1 = slower, <1 = faster)
        """
        self.mapper = HumanAngleToOP3Mapper(motor_limits)
        self.balance = BalanceController() if balance_enabled else None
        self.time_scale = time_scale

    # ---- From human angle sequence (e.g. from .npy template) ----

    def run_from_angles(
        self,
        human_angle_sequence: list[dict[str, float]],
        name: str = "motion",
        frame_rate: int = 30,
        output_path: Optional[str] = None,
    ) -> MotionKeyframes:
        """Convert a sequence of human angle dicts to OP3 keyframes.

        Args:
            human_angle_sequence: List of {angle_name: degrees} from GeometricIKSolver
            name: Motion name for output
            frame_rate: FPS of the input data
            output_path: If given, save JSON to this path

        Returns:
            MotionKeyframes ready for playback
        """
        # Step 1: Retarget to OP3 motor positions
        motor_seqs = self.mapper.map_sequence(human_angle_sequence)

        # Step 2: Balance corrections
        if self.balance is not None:
            motor_seqs = self.balance.process_keyframes(motor_seqs, frame_rate)

        # Step 3: Time scaling
        keyframes = MotionKeyframes.from_motor_sequences(
            name=name,
            sequences=motor_seqs,
            frame_rate=frame_rate,
            loop=True,
        )

        if self.time_scale != 1.0:
            keyframes = keyframes.time_scale(self.time_scale)

        # Step 4: Save
        if output_path is not None:
            keyframes.save(output_path)
            print(f"[Pipeline] Saved {keyframes.num_frames} frames to {output_path}")

        return keyframes

    # ---- From .npy template file ----

    def run_from_template(
        self,
        template_path: str,
        name: Optional[str] = None,
        output_dir: str = "motions",
    ) -> MotionKeyframes:
        """Convert a .npy movement template to OP3 keyframes.

        The .npy file should contain a dict with:
          - "joint_angles": (T, N) array of human joint angles in degrees
          - "joint_names": list of N joint names
          - "fps": frame rate (default 30)

        If loading fails (e.g. missing dependencies in pickled data),
        falls back to generating synthetic squat data.

        Args:
            template_path: Path to .npy template file
            name: Motion name (defaults to filename stem)
            output_dir: Directory for output JSON

        Returns:
            MotionKeyframes
        """
        if name is None:
            name = os.path.splitext(os.path.basename(template_path))[0].replace("_reference", "")

        data = self._load_template_data(template_path, name)

        joint_angles: np.ndarray = data["joint_angles"]
        joint_names: list[str] = data.get("joint_names", [])
        fps: int = data.get("fps", 30)

        num_frames = joint_angles.shape[0]
        human_sequence = []
        for frame_idx in range(num_frames):
            frame_dict = {}
            for j, jname in enumerate(joint_names):
                frame_dict[jname] = float(joint_angles[frame_idx, j])
            human_sequence.append(frame_dict)

        output_path = os.path.join(output_dir, f"{name}_imitation.json")
        return self.run_from_angles(
            human_sequence,
            name=name,
            frame_rate=fps,
            output_path=output_path,
        )

    @staticmethod
    def _load_template_data(filepath: str, name: str) -> dict:
        """Load template data from .npy, with fallback to synthetic generation."""
        # Try direct numpy load first
        try:
            teacher_feng_path = r"D:\Project_of_Teacher_Feng\WorkPlace"
            if teacher_feng_path not in sys.path:
                sys.path.insert(0, teacher_feng_path)
            return np.load(filepath, allow_pickle=True).item()
        except Exception as e:
            print(f"[Pipeline] Cannot load {filepath} directly: {e}")
            print(f"[Pipeline] Generating synthetic '{name}' template instead.")

        # Fallback: generate synthetic trajectory
        return PipelineRunner._generate_synthetic_data(name)

    @staticmethod
    def _generate_synthetic_data(movement: str) -> dict:
        """Generate synthetic joint angle data for common exercises."""
        total_frames = 90
        fps = 30

        if movement == "squat":
            joint_names = [
                "hip_flexion_r", "hip_flexion_l",
                "knee_angle_r", "knee_angle_l",
                "ankle_angle_r", "ankle_angle_l",
                "lumbar_extension",
                "hip_abduction_r", "hip_abduction_l",
            ]
            angles = np.zeros((total_frames, len(joint_names)))
            t = np.linspace(0, 1, total_frames)

            for i, name in enumerate(joint_names):
                if "knee" in name:
                    angles[:, i] = _smooth_valley(t, 180.0, 95.0)
                elif "hip" in name and "abduction" not in name:
                    angles[:, i] = _smooth_valley(t, 180.0, 60.0)
                elif "ankle" in name:
                    angles[:, i] = _smooth_valley(t, 90.0, 70.0)
                elif "lumbar" in name:
                    angles[:, i] = _smooth_valley(t, 0.0, -5.0)
                elif "abduction" in name:
                    angles[:, i] = np.full(total_frames, 0.0)
        elif movement == "pushup":
            joint_names = [
                "elbow_angle_r", "elbow_angle_l",
                "shoulder_angle_r", "shoulder_angle_l",
                "lumbar_extension",
            ]
            angles = np.zeros((total_frames, len(joint_names)))
            t = np.linspace(0, 1, total_frames)
            for i, name in enumerate(joint_names):
                if "elbow" in name:
                    angles[:, i] = _smooth_valley(t, 180.0, 90.0)
                elif "shoulder" in name:
                    angles[:, i] = _smooth_valley(t, 0.0, -40.0)
                elif "lumbar" in name:
                    angles[:, i] = np.full(total_frames, 0.0)
        else:
            # Generic squat-like trajectory
            joint_names = [
                "hip_flexion_r", "hip_flexion_l",
                "knee_angle_r", "knee_angle_l",
            ]
            angles = np.zeros((total_frames, len(joint_names)))
            t = np.linspace(0, 1, total_frames)
            for i, name in enumerate(joint_names):
                if "knee" in name:
                    angles[:, i] = _smooth_valley(t, 180.0, 95.0)
                elif "hip" in name:
                    angles[:, i] = _smooth_valley(t, 180.0, 60.0)

        return {
            "joint_angles": angles,
            "joint_names": joint_names,
            "fps": fps,
        }

    # ---- From video file ----

    def run_from_video(
        self,
        video_path: str,
        name: Optional[str] = None,
        output_dir: str = "motions",
        target_fps: int = 30,
    ) -> Optional[MotionKeyframes]:
        """Process a training video and produce OP3 keyframes.

        Uses MediaPipe Pose + GeometricIKSolver from Teacher Feng's project.
        The project must be on sys.path.

        Args:
            video_path: Path to training video (.mp4, .avi, etc.)
            name: Motion name (defaults to video filename stem)
            output_dir: Directory for output JSON
            target_fps: Target processing frame rate

        Returns:
            MotionKeyframes or None if video can't be processed
        """
        try:
            import cv2
            sys.path.insert(0, r"D:\Project_of_Teacher_Feng\WorkPlace")
            from src.pose.estimator import PoseEstimator
            from src.bridge.socket_server import GeometricIKSolver
        except ImportError as e:
            print(f"[Pipeline] Cannot import video processing modules: {e}")
            print("[Pipeline] Install: pip install mediapipe opencv-python")
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Pipeline] Cannot open video: {video_path}")
            return None

        if name is None:
            name = os.path.splitext(os.path.basename(video_path))[0]

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if src_fps <= 0:
            src_fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        skip = max(1, int(src_fps / target_fps))
        estimator = PoseEstimator(model_complexity=2)

        human_sequence = []
        frame_count = 0

        print(f"[Pipeline] Processing video: {video_path}")
        print(f"  Source FPS: {src_fps:.1f}, Total frames: {total_frames}, Skip: {skip}")

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i % skip != 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = estimator.detect(frame_rgb)

            if result.detected:
                angles = GeometricIKSolver.solve(result.world_landmarks)
                human_sequence.append(angles)
                frame_count += 1

            if frame_count % 60 == 0 and frame_count > 0:
                print(f"  Processed {frame_count} frames...")

        cap.release()
        estimator.close()

        if frame_count < 10:
            print(f"[Pipeline] Too few valid frames: {frame_count}")
            return None

        print(f"[Pipeline] Extracted {frame_count} valid frames")

        output_path = os.path.join(output_dir, f"{name}_imitation.json")
        return self.run_from_angles(
            human_sequence,
            name=name,
            frame_rate=target_fps,
            output_path=output_path,
        )


def _smooth_valley(
    t: np.ndarray,
    start_val: float,
    bottom_val: float,
    descent_pct: float = 0.15,
    bottom_pct: float = 0.50,
) -> np.ndarray:
    """Generate a smooth valley curve for one rep cycle."""
    y = np.full_like(t, start_val)
    for i in range(len(t)):
        ti = t[i]
        if descent_pct < ti <= bottom_pct:
            p = ((ti - descent_pct) / (bottom_pct - descent_pct)) ** 2
            y[i] = start_val + (bottom_val - start_val) * p
        elif bottom_pct < ti <= bottom_pct + 0.07:
            y[i] = bottom_val
        elif ti > bottom_pct + 0.07:
            p = (ti - bottom_pct - 0.07) / (1.0 - bottom_pct - 0.07)
            p = 1.0 - (1.0 - p) ** 2
            y[i] = bottom_val + (start_val - bottom_val) * p
    return y
