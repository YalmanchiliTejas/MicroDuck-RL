#!/usr/bin/env python3
"""Render one deterministic GrapePick rollout for every saved checkpoint.

Run this alongside training, not inside the training process.  Rendering one
environment is intentionally kept separate from the 4,096-environment PPO
job so it cannot slow or perturb learning.

Example:
    uv run python scripts/record_grape_checkpoints.py \
        --checkpoint-dir "$SCRATCH/microduck-rl/grape-pick/tensorboard"
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


TASK_ID = "Mjlab-GrapePick-Flat-MicroDuck"
_CHECKPOINT_RE = re.compile(r"^model_(\d+)\.pt$")


def checkpoint_iteration(path: Path) -> int | None:
    """Return the iteration encoded in a standard rsl_rl checkpoint name."""
    match = _CHECKPOINT_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


def find_checkpoints(checkpoint_dir: Path) -> list[Path]:
    """Find checkpoints in chronological order, including timestamped run dirs."""
    checkpoints = [
        path for path in checkpoint_dir.rglob("model_*.pt")
        if checkpoint_iteration(path) is not None
    ]
    return sorted(checkpoints, key=lambda path: (checkpoint_iteration(path), str(path)))


def completion_marker(video_root: Path, iteration: int) -> Path:
    return video_root / f"model_{iteration}" / "complete.json"


def record_checkpoint(args: argparse.Namespace, checkpoint: Path, iteration: int) -> None:
    """Run the existing checkpoint player and leave an iteration-specific video."""
    destination = args.video_dir / f"model_{iteration}"
    destination.mkdir(parents=True, exist_ok=True)

    # export.py owns the correct policy loading and VideoRecorder wiring.  It
    # also exports an ONNX as part of its normal contract; place that transient
    # file in a temporary directory and discard it afterwards.
    with tempfile.TemporaryDirectory(prefix="grape-video-onnx-") as temp_dir:
        command = [
            args.uv_command,
            "run",
            "scripts/export.py",
            TASK_ID,
            "--checkpoint-file", str(checkpoint.resolve()),
            "--onnx-file", str(Path(temp_dir) / "policy.onnx"),
            "--video",
            "--video-folder", str(destination.resolve()),
            "--video-length", str(args.video_length),
            "--video-width", str(args.video_width),
            "--video-height", str(args.video_height),
            "--num-envs", "1",
            "--device", args.device,
            "--seed", str(args.seed),
        ]
        print("[record]", " ".join(command), flush=True)
        subprocess.run(command, check=True)

    if not list(destination.rglob("*.mp4")):
        raise RuntimeError(
            f"Checkpoint {iteration} finished without writing an MP4 to {destination}"
        )
    marker = completion_marker(args.video_dir, iteration)
    marker.write_text(json.dumps({
        "checkpoint": str(checkpoint.resolve()),
        "iteration": iteration,
        "seed": args.seed,
        "video_length": args.video_length,
        "recorded_at_unix_s": time.time(),
    }, indent=2) + "\n")
    print(f"[record] completed checkpoint {iteration}: {destination}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="Directory containing model_<iteration>.pt files.")
    parser.add_argument("--video-dir", type=Path, default=None,
                        help="Destination root (default: <checkpoint-dir>/videos/checkpoints).")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--min-age-seconds", type=float, default=20.0,
                        help="Do not load a checkpoint until this long after its last write.")
    parser.add_argument("--video-length", type=int, default=200)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0,
                        help="Fixed evaluation seed, shared by every checkpoint video.")
    parser.add_argument("--device", default="cpu",
                        help="Evaluation device; CPU avoids contending with the training GPU.")
    parser.add_argument("--uv-command", default="uv")
    parser.add_argument("--once", action="store_true",
                        help="Record the currently stable checkpoints, then exit.")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.min_age_seconds < 0 or args.video_length <= 0:
        parser.error("poll/age values must be non-negative and video length must be positive")
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.video_dir = (args.video_dir or args.checkpoint_dir / "videos" / "checkpoints").resolve()
    return args


def main() -> int:
    args = parse_args()
    if not args.checkpoint_dir.is_dir():
        raise SystemExit(f"Checkpoint directory does not exist: {args.checkpoint_dir}")
    args.video_dir.mkdir(parents=True, exist_ok=True)
    print(f"[watch] checkpoints: {args.checkpoint_dir}")
    print(f"[watch] videos:      {args.video_dir}")

    while True:
        now = time.time()
        for checkpoint in find_checkpoints(args.checkpoint_dir):
            iteration = checkpoint_iteration(checkpoint)
            assert iteration is not None
            if completion_marker(args.video_dir, iteration).exists():
                continue
            if now - checkpoint.stat().st_mtime < args.min_age_seconds:
                continue
            try:
                record_checkpoint(args, checkpoint, iteration)
            except subprocess.CalledProcessError as error:
                # Do not write a completion marker: a transient render failure
                # is retried at the next poll, without losing the checkpoint.
                print(f"[record] checkpoint {iteration} failed ({error}); will retry", file=sys.stderr)

        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
