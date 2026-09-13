#!/usr/bin/env python3
"""Rerender GrapePick checkpoints without command/velocity arrows.

The arrows are simulator debug geometry and cannot be cleanly removed from
pixels they already cover. This script reruns the deterministic checkpoint
recorder after the GrapePick config disables command debug visualization. It
writes to a separate ``checkpoints-clean`` directory by default, preserving the
original videos.

Example:
    uv run python scripts/rerecord_grape_videos_without_arrows.py \
        --checkpoint-dir "$SCRATCH/microduck-rl/grape-pick/tensorboard"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing model_<iteration>.pt checkpoints.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help=(
            "Clean-video destination (default: "
            "<checkpoint-dir>/videos/checkpoints-clean)."
        ),
    )
    parser.add_argument("--video-length", type=int, default=300)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-distance", type=float, default=0.55)
    parser.add_argument("--video-azimuth", type=float, default=90.0)
    parser.add_argument("--video-elevation", type=float, default=-15.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mujoco-gl", choices=("egl", "osmesa"), default="egl")
    parser.add_argument("--uv-command", default="uv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    video_dir = (
        args.video_dir.resolve()
        if args.video_dir is not None
        else checkpoint_dir / "videos" / "checkpoints-clean"
    )
    recorder = Path(__file__).with_name("record_grape_checkpoints.py")
    command = [
        sys.executable,
        str(recorder),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--video-dir",
        str(video_dir),
        "--once",
        "--min-age-seconds",
        "0",
        "--video-length",
        str(args.video_length),
        "--video-width",
        str(args.video_width),
        "--video-height",
        str(args.video_height),
        "--video-distance",
        str(args.video_distance),
        "--video-azimuth",
        str(args.video_azimuth),
        "--video-elevation",
        str(args.video_elevation),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--mujoco-gl",
        args.mujoco_gl,
        "--uv-command",
        args.uv_command,
    ]
    print(f"[clean-render] output: {video_dir}", flush=True)
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
