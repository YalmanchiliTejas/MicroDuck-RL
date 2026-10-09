"""Evaluate and rank numbered Mario PPO checkpoints with frozen MaleCNS."""

from __future__ import annotations

import argparse
import json
from math import log
from pathlib import Path
import shutil

import numpy as np
import torch

import evaluate_flybrain


def _step(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("algorithm") != "ppo":
        raise ValueError(f"not a PPO checkpoint: {path}")
    return int(payload["steps"])


def _entropy(summary: dict) -> float:
    fractions = np.asarray(
        [row["selected_fraction"] for row in summary["actions"].values()],
        dtype=np.float64,
    )
    positive = fractions[fractions > 0.0]
    if not len(positive):
        return 0.0
    return float(-(positive * np.log(positive)).sum() / log(len(fractions)))


def _normalized(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if high <= low:
        return [0.5] * len(values)
    return [(value - low) / (high - low) for value in values]


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="flybrain-ppo-step-*.pt")
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--best-output", type=Path)
    parser.add_argument(
        "--dopamine-checkpoint-dir",
        type=Path,
        help="evaluate each PPO with dopamine-plasticity-step-STEP.npz from this directory",
    )
    parser.add_argument("--best-dopamine-output", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-decisions", type=int, default=300)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--male-cns-data", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--reuse-reports", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_decisions <= 0 or args.action_repeat <= 0:
        parser.error("evaluation counts must be positive")
    if args.best_dopamine_output is not None and args.dopamine_checkpoint_dir is None:
        parser.error("--best-dopamine-output requires --dopamine-checkpoint-dir")

    checkpoints = sorted(args.checkpoint_dir.glob(args.pattern), key=_step)
    if not checkpoints:
        parser.error(f"no checkpoints matching {args.checkpoint_dir / args.pattern}")
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for checkpoint in checkpoints:
        step = _step(checkpoint)
        dopamine_state = None
        if args.dopamine_checkpoint_dir is not None:
            dopamine_state = args.dopamine_checkpoint_dir / (
                f"dopamine-plasticity-step-{step:09d}.npz"
            )
            if not dopamine_state.is_file():
                parser.error(
                    f"missing dopamine state paired with step {step}: {dopamine_state}"
                )
        summaries = {}
        for mode in ("mean", "sampled"):
            state_label = "dopamine" if dopamine_state is not None else "clean"
            report_path = (
                args.reports_dir / f"step-{step:09d}-{state_label}-{mode}.json"
            )
            if args.reuse_reports and report_path.exists():
                report = json.loads(report_path.read_text())
            else:
                evaluation_args = argparse.Namespace(
                    checkpoint=checkpoint,
                    dopamine_state=dopamine_state,
                    dopamine_learning_rate=1.0e-5,
                    male_cns_data=args.male_cns_data,
                    male_cns_device="cpu",
                    env="SuperMarioBros-1-1-v0",
                    episodes=args.episodes,
                    max_decisions_per_episode=args.max_decisions,
                    action_repeat=args.action_repeat,
                    epsilon=0.0,
                    sample_actions=mode == "sampled",
                    seed=args.seed,
                    first_actions=0,
                    output=report_path,
                    allow_legacy_reward_contract=False,
                    device=args.device,
                )
                report = evaluate_flybrain.run(evaluation_args)
            summaries[mode] = report["summary"]

        mean, sampled = summaries["mean"], summaries["sampled"]
        completion = 0.5 * (
            mean["completion_rate"] + sampled["completion_rate"]
        )
        duration_survival = 0.5 * (
            min(1.0, mean["mean_decisions"] / args.max_decisions)
            + min(1.0, sampled["mean_decisions"] / args.max_decisions)
        )
        nondeath = 0.5 * (
            2.0 - mean["death_rate"] - sampled["death_rate"]
        )
        rows.append(
            {
                "step": step,
                "checkpoint": str(checkpoint.resolve()),
                "dopamine_state": (
                    None if dopamine_state is None else str(dopamine_state.resolve())
                ),
                "mean_raw_reward": 0.5
                * (mean["mean_raw_reward"] + sampled["mean_raw_reward"]),
                "mean_max_x": 0.5 * (mean["mean_max_x"] + sampled["mean_max_x"]),
                "completion_rate": completion,
                "survival": max(completion, 0.5 * (nondeath + duration_survival)),
                "sampled_action_entropy": _entropy(sampled),
                "modes": summaries,
            }
        )

    reward_scores = _normalized([row["mean_raw_reward"] for row in rows])
    progress_scores = _normalized([row["mean_max_x"] for row in rows])
    for row, reward_score, progress_score in zip(
        rows, reward_scores, progress_scores, strict=True
    ):
        row["score"] = (
            0.30 * reward_score
            + 0.30 * progress_score
            + 0.20 * row["completion_rate"]
            + 0.10 * row["survival"]
            + 0.10 * row["sampled_action_entropy"]
        )
    rows.sort(key=lambda row: (row["score"], row["step"]), reverse=True)
    result = {
        "recommended_checkpoint": rows[0]["checkpoint"],
        "recommended_dopamine_state": rows[0]["dopamine_state"],
        "scoring": {
            "relative_mean_raw_reward": 0.30,
            "relative_mean_max_x": 0.30,
            "completion_rate": 0.20,
            "survival": 0.10,
            "sampled_action_entropy": 0.10,
            "note": "Reward/progress are min-max normalized within this checkpoint set.",
        },
        "evaluation": {
            "episodes_per_mode": args.episodes,
            "modes": ["mean", "sampled"],
            "seed": args.seed,
            "action_repeat": args.action_repeat,
            "max_decisions": args.max_decisions,
            "dopamine": (
                "paired frozen state"
                if args.dopamine_checkpoint_dir is not None
                else "disabled"
            ),
        },
        "ranking": rows,
    }
    _atomic_json(args.output, result)
    if args.best_output is not None:
        _atomic_copy(Path(rows[0]["checkpoint"]), args.best_output)
    if args.best_dopamine_output is not None:
        _atomic_copy(
            Path(rows[0]["dopamine_state"]), args.best_dopamine_output
        )

    print("\nrank  step       score  reward   max_x  survive entropy complete checkpoint")
    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank:>4}  {row['step']:>9}  {row['score'] * 100:>5.1f}% "
            f"{row['mean_raw_reward']:>7.1f} {row['mean_max_x']:>7.1f} "
            f"{row['survival'] * 100:>6.1f}% {row['sampled_action_entropy'] * 100:>6.1f}% "
            f"{row['completion_rate'] * 100:>6.1f}% {row['checkpoint']}"
        )
    print(f"\nRecommended checkpoint: {rows[0]['checkpoint']}")
    if args.best_output is not None:
        print(f"Copied recommendation to: {args.best_output}")
    if args.best_dopamine_output is not None:
        print(f"Copied paired dopamine state to: {args.best_dopamine_output}")
    print(f"Full ranking: {args.output}")


if __name__ == "__main__":
    main()
