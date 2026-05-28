#!/usr/bin/env python3
"""Generate OP2 motion JSON from a training video.

Uses MediaPipe Pose + GeometricIKSolver from Teacher Feng's project
to extract human poses, then retargets to OP2 motor positions.

Usage:
  python scripts/generate_motion_from_video.py --video path/to/squat.mp4
  python scripts/generate_motion_from_video.py --video test_squat.mp4 --name squat --time-scale 0.5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.pipeline_runner import PipelineRunner


def main():
    parser = argparse.ArgumentParser(
        description="Convert training video to OP2 motion JSON"
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to training video (.mp4, .avi, etc.)"
    )
    parser.add_argument(
        "--name", default=None,
        help="Motion name (defaults to video filename)"
    )
    parser.add_argument(
        "--output-dir", default="motions",
        help="Output directory for JSON files (default: motions/)"
    )
    parser.add_argument(
        "--target-fps", type=int, default=30,
        help="Processing frame rate (default: 30)"
    )
    parser.add_argument(
        "--time-scale", type=float, default=1.0,
        help="Playback speed factor (default: 1.0)"
    )
    parser.add_argument(
        "--no-balance", action="store_true",
        help="Disable balance corrections"
    )
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    runner = PipelineRunner(
        balance_enabled=not args.no_balance,
        time_scale=args.time_scale,
    )

    keyframes = runner.run_from_video(
        video_path=args.video,
        name=args.name,
        output_dir=args.output_dir,
        target_fps=args.target_fps,
    )

    if keyframes is None:
        print("Error: Failed to process video. Check dependencies (mediapipe, opencv-python).")
        sys.exit(1)

    print(f"\nDone! Motion saved with {keyframes.num_frames} frames, "
          f"{keyframes.duration:.1f}s duration")


if __name__ == "__main__":
    main()
