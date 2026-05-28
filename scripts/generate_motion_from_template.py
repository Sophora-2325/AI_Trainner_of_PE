#!/usr/bin/env python3
"""Generate OP3 motion JSON from a .npy movement template.

Usage:
  python scripts/generate_motion_from_template.py --template movement_data/squat_reference.npy
  python scripts/generate_motion_from_template.py --template squat_reference.npy --name squat --time-scale 0.5
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.pipeline_runner import PipelineRunner


def main():
    parser = argparse.ArgumentParser(
        description="Convert .npy movement template to OP3 motion JSON"
    )
    parser.add_argument(
        "--template", required=True,
        help="Path to .npy template file (from MovementLibrary)"
    )
    parser.add_argument(
        "--name", default=None,
        help="Motion name (defaults to template filename)"
    )
    parser.add_argument(
        "--output-dir", default="motions",
        help="Output directory for JSON files (default: motions/)"
    )
    parser.add_argument(
        "--time-scale", type=float, default=1.0,
        help="Playback speed factor: 0.5=half speed, 2.0=double speed (default: 1.0)"
    )
    parser.add_argument(
        "--no-balance", action="store_true",
        help="Disable balance corrections"
    )
    args = parser.parse_args()

    if not os.path.exists(args.template):
        print(f"Error: Template file not found: {args.template}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    runner = PipelineRunner(
        balance_enabled=not args.no_balance,
        time_scale=args.time_scale,
    )

    keyframes = runner.run_from_template(
        template_path=args.template,
        name=args.name,
        output_dir=args.output_dir,
    )

    print(f"\nDone! Motion saved with {keyframes.num_frames} frames, "
          f"{keyframes.duration:.1f}s duration")
    print(f"Motor count: {len(keyframes.motor_names)}")
    print(f"Sample frame 0 knee positions: "
          f"LegLowerR={keyframes.motor_positions['LegLowerR'][0]:.3f} rad, "
          f"LegLowerL={keyframes.motor_positions['LegLowerL'][0]:.3f} rad")


if __name__ == "__main__":
    main()
