#!/usr/bin/env python3
"""Evaluate and rank Mario checkpoints with one reproducible command.

The deployment score deliberately rewards balance rather than allowing one
excellent command to hide one unusable command:

* 50% geometric mean of all six clean-success fractions.
* 40% weakest clean-success fraction.
* 10% fraction of completed episodes that did not end in a fall.

Evaluation reports are cached beside the run by default, so rerunning this
script resumes after the last completed checkpoint.
"""

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys


COMMAND_NAMES = (
    "neutral",
    "left",
    "right",
    "jump",
    "left_jump",
    "right_jump",
)
CHECKPOINT_PATTERN = re.compile(r"model_(\d+)\.pt$")


def checkpoint_iteration(path: Path) -> int | None:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def discover_checkpoints(
    run_dir: Path,
    min_iteration: int | None = None,
    max_iteration: int | None = None,
    every: int | None = None,
) -> list[tuple[int, Path]]:
    """Return unique numeric checkpoints beneath ``run_dir`` in order."""
    by_iteration: dict[int, list[Path]] = {}
    for path in run_dir.rglob("model_*.pt"):
        iteration = checkpoint_iteration(path)
        if iteration is None:
            continue
        if min_iteration is not None and iteration < min_iteration:
            continue
        if max_iteration is not None and iteration > max_iteration:
            continue
        if every is not None:
            anchor = min_iteration or 0
            if (iteration - anchor) % every:
                continue
        by_iteration.setdefault(iteration, []).append(path.resolve())

    duplicates = {
        iteration: paths
        for iteration, paths in by_iteration.items()
        if len(set(paths)) > 1
    }
    if duplicates:
        detail = "; ".join(
            f"{iteration}: {', '.join(map(str, paths))}"
            for iteration, paths in sorted(duplicates.items())
        )
        raise ValueError(
            "Multiple checkpoints have the same iteration; pass a more specific "
            f"run directory ({detail})"
        )
    return [
        (iteration, paths[0])
        for iteration, paths in sorted(by_iteration.items())
    ]


def score_report(report: dict) -> dict[str, float]:
    """Compute a balance-first deployment score from an evaluator report."""
    fractions = []
    for name in COMMAND_NAMES:
        try:
            value = report["commands"][name]["clean_fraction"]
        except KeyError as error:
            raise ValueError(f"Report is missing command {name!r}") from error
        if value is None or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid clean fraction for {name}: {value}")
        fractions.append(float(value))

    # All evaluator command buckets contain thousands of samples, so a tiny
    # epsilon only defines the zero-skill edge case; it does not affect normal
    # rankings.
    geometric_mean = math.prod(max(value, 1e-12) for value in fractions) ** (
        1.0 / len(fractions)
    )
    weakest_command = min(fractions)
    completed = int(report.get("completed_episodes", 0))
    falls = int(report.get("falls", 0))
    non_fall_fraction = 1.0 - min(1.0, falls / completed) if completed else 0.0
    overall = (
        0.50 * geometric_mean
        + 0.40 * weakest_command
        + 0.10 * non_fall_fraction
    )
    return {
        "overall": overall,
        "geometric_command_mean": geometric_mean,
        "weakest_command": weakest_command,
        "non_fall_fraction": non_fall_fraction,
    }


def cache_matches(
    report: dict,
    checkpoint: Path,
    *,
    mode: str,
    seed: int,
    num_envs: int,
    steps: int,
) -> bool:
    return (
        Path(report.get("checkpoint", "")) == checkpoint.resolve()
        and report.get("mode") == mode
        and report.get("seed") == seed
        and report.get("num_envs") == num_envs
        and report.get("steps") == steps
        and all(name in report.get("commands", {}) for name in COMMAND_NAMES)
    )


def evaluate_checkpoint(
    checkpoint: Path,
    report_path: Path,
    *,
    evaluator: Path,
    mode: str,
    seed: int,
    num_envs: int,
    steps: int,
    device: str,
) -> dict:
    if report_path.is_file():
        try:
            cached = json.loads(report_path.read_text())
        except (json.JSONDecodeError, OSError):
            cached = None
        if cached is not None and cache_matches(
            cached,
            checkpoint,
            mode=mode,
            seed=seed,
            num_envs=num_envs,
            steps=steps,
        ):
            print(f"[cache] {checkpoint.name}", flush=True)
            return cached

    print(f"[eval]  {checkpoint.name}", flush=True)
    command = [
        sys.executable,
        str(evaluator),
        str(checkpoint),
        "--mode",
        mode,
        "--seed",
        str(seed),
        "--num-envs",
        str(num_envs),
        "--steps",
        str(steps),
        "--device",
        device,
        "--include-combinations",
        "--output",
        str(report_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    report = json.loads(report_path.read_text())
    if not cache_matches(
        report,
        checkpoint,
        mode=mode,
        seed=seed,
        num_envs=num_envs,
        steps=steps,
    ):
        raise RuntimeError(f"Evaluator produced mismatched report: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing model_*.pt files")
    parser.add_argument("--min-iteration", type=int)
    parser.add_argument("--max-iteration", type=int)
    parser.add_argument(
        "--every",
        type=int,
        help="Evaluate every N iterations, anchored at --min-iteration or zero",
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("sampled", "mean"), default="mean")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        help="Cache directory (default: RUN_DIR/mario_checkpoint_evaluations)",
    )
    parser.add_argument(
        "--selection-output",
        type=Path,
        help="Ranking JSON (default: REPORTS_DIR/selection.json)",
    )
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        parser.error(f"Run directory not found: {args.run_dir}")
    if args.every is not None and args.every <= 0:
        parser.error("--every must be positive")
    if args.num_envs <= 0 or args.steps <= 0:
        parser.error("--num-envs and --steps must be positive")
    if (
        args.min_iteration is not None
        and args.max_iteration is not None
        and args.min_iteration > args.max_iteration
    ):
        parser.error("--min-iteration cannot exceed --max-iteration")

    try:
        checkpoints = discover_checkpoints(
            args.run_dir,
            min_iteration=args.min_iteration,
            max_iteration=args.max_iteration,
            every=args.every,
        )
    except ValueError as error:
        parser.error(str(error))
    if not checkpoints:
        parser.error("No matching model_ITERATION.pt checkpoints found")

    reports_dir = args.reports_dir or args.run_dir / "mario_checkpoint_evaluations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    selection_output = args.selection_output or reports_dir / "selection.json"
    evaluator = Path(__file__).with_name("evaluate_mario_controller.py")

    ranking = []
    for iteration, checkpoint in checkpoints:
        cache_name = (
            f"model_{iteration}_{args.mode}_seed{args.seed}_"
            f"envs{args.num_envs}_steps{args.steps}.json"
        )
        report = evaluate_checkpoint(
            checkpoint,
            reports_dir / cache_name,
            evaluator=evaluator,
            mode=args.mode,
            seed=args.seed,
            num_envs=args.num_envs,
            steps=args.steps,
            device=args.device,
        )
        scores = score_report(report)
        ranking.append(
            {
                "iteration": iteration,
                "checkpoint": str(checkpoint),
                "score": scores,
                "falls": report["falls"],
                "completed_episodes": report["completed_episodes"],
                "clean_fractions": {
                    name: report["commands"][name]["clean_fraction"]
                    for name in COMMAND_NAMES
                },
            }
        )

    ranking.sort(key=lambda row: (row["score"]["overall"], row["iteration"]), reverse=True)
    result = {
        "scoring": {
            "geometric_command_mean_weight": 0.50,
            "weakest_command_weight": 0.40,
            "non_fall_fraction_weight": 0.10,
            "note": "Higher is better; all six deployment commands are equally represented.",
        },
        "evaluation": {
            "mode": args.mode,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "steps": args.steps,
        },
        "recommended_checkpoint": ranking[0]["checkpoint"],
        "ranking": ranking,
    }
    selection_output.parent.mkdir(parents=True, exist_ok=True)
    selection_output.write_text(json.dumps(result, indent=2) + "\n")

    header = "rank  iter  score   worst   geom    safe    falls  checkpoint"
    print(f"\n{header}")
    for rank, row in enumerate(ranking, start=1):
        score = row["score"]
        print(
            f"{rank:>4}  {row['iteration']:>4}  "
            f"{score['overall'] * 100:>5.1f}%  "
            f"{score['weakest_command'] * 100:>5.1f}%  "
            f"{score['geometric_command_mean'] * 100:>5.1f}%  "
            f"{score['non_fall_fraction'] * 100:>5.1f}%  "
            f"{row['falls']:>5}  {row['checkpoint']}"
        )
    print(f"\nRecommended checkpoint: {ranking[0]['checkpoint']}")
    print(f"Full ranking: {selection_output}")


if __name__ == "__main__":
    main()
