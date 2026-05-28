#!/usr/bin/env python3
"""Plot OP2 motor position curves from a motion JSON file.

Usage:
  python scripts/visualize_keyframes.py motions/squat_imitation.json
  python scripts/visualize_keyframes.py motions/squat_imitation.json --motors LegLowerR LegUpperR
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Try importing matplotlib; don't fail if not available
try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def main():
    parser = argparse.ArgumentParser(
        description="Visualize OP2 motion keyframe curves"
    )
    parser.add_argument("motion_file", help="Path to motion JSON file")
    parser.add_argument(
        "--motors", nargs="*", default=None,
        help="Motor names to plot (default: all leg motors)"
    )
    args = parser.parse_args()

    if not HAS_MPL:
        print("Error: matplotlib is required. Install with: pip install matplotlib")
        sys.exit(1)

    if not os.path.exists(args.motion_file):
        print(f"Error: File not found: {args.motion_file}")
        sys.exit(1)

    with open(args.motion_file, "r") as f:
        data = json.load(f)

    motor_positions = data.get("motor_positions", {})
    if not motor_positions:
        print("No motor data found in file")
        sys.exit(1)

    # Default: show leg motors
    if args.motors is None:
        args.motors = [
            "LegUpperR", "LegUpperL",
            "LegLowerR", "LegLowerL",
            "AnkleR", "AnkleL",
        ]

    frame_rate = data.get("frame_rate", 30)
    num_frames = len(next(iter(motor_positions.values())))

    time_axis = np.arange(num_frames) / frame_rate

    # Group: right side, left side, balance
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Motion: {data.get('name', 'unknown')}  ({num_frames} frames @ {frame_rate} FPS)")

    # Plot 1: Right leg
    ax = axes[0, 0]
    ax.set_title("Right Leg")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (rad)")
    for name in ["LegUpperR", "LegLowerR", "AnkleR", "FootR", "PelvR"]:
        if name in motor_positions and name in args.motors:
            ax.plot(time_axis, np.rad2deg(motor_positions[name]), label=name)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Left leg
    ax = axes[0, 1]
    ax.set_title("Left Leg")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (rad)")
    for name in ["LegUpperL", "LegLowerL", "AnkleL", "FootL", "PelvL"]:
        if name in motor_positions and name in args.motors:
            ax.plot(time_axis, np.rad2deg(motor_positions[name]), label=name)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Arms
    ax = axes[1, 0]
    ax.set_title("Arms")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (rad)")
    for name in ["ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL",
                 "ArmLowerR", "ArmLowerL"]:
        if name in motor_positions:
            ax.plot(time_axis, np.rad2deg(motor_positions[name]), label=name)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Head + remaining
    ax = axes[1, 1]
    ax.set_title("Head & Other")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (rad)")
    for name in ["Neck", "Head", "PelvYR", "PelvYL"]:
        if name in motor_positions:
            ax.plot(time_axis, np.rad2deg(motor_positions[name]), label=name)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
