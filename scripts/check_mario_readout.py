"""Explicit smoke-health and level-completion gates for the Mario workflow."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def training_health(text: str) -> dict:
    rows = []
    for line in text.splitlines():
        if not line.startswith("ppo_update "):
            continue
        rows.append({key: float(value) for key, value in
                     (part.split("=", 1) for part in line.split()[1:] if "=" in part)})
    if not rows:
        raise ValueError("No PPO updates found; training health is unverified")
    recent = rows[-5:]
    for row in recent:
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError("Nonfinite training metric")
        if row.get("hidden_saturation", 1) >= 0.1:
            raise ValueError("Hidden saturation exceeded smoke threshold")
        if row.get("update_gate_high", 1) >= 0.5:
            raise ValueError("Most update gates are retaining memory almost completely")
    if not any(row.get("spatial_grad", 0) > 0 for row in recent):
        raise ValueError("Direct spatial branch has no observed learning gradient")
    return {"status": "smoke_health_passed", "updates": len(rows), "last_update": rows[-1],
            "level_completion_verified": False}


def completion_status(reports: list[dict]) -> dict:
    if not reports:
        raise ValueError("No evaluation reports")
    if len({report.get("checkpoint") for report in reports}) != 1:
        raise ValueError("Reports must evaluate the same checkpoint")
    completions = {}
    for report in reports:
        if report.get("visual_ablation") != "normal" or report.get("learning_enabled") is not False:
            raise ValueError("Completion gate requires frozen normal-vision evaluation")
        episodes = report.get("episodes", [])
        if not episodes:
            raise ValueError("No evaluated episodes")
        completions[report["mode"]] = sum(row.get("completion") is True for row in episodes)
    passed = any(count > 0 for count in completions.values())
    return {"status": "level_completion_observed" if passed else "no_level_completion",
            "completions_by_mode": completions, "level_completion_verified": passed,
            "note": "One completion is evidence of capability, not reliable success across seeds."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-log", type=Path)
    parser.add_argument("--reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.training_log) == bool(args.reports):
        parser.error("choose training-log or reports")
    result = training_health(args.training_log.read_text()) if args.training_log else completion_status(
        [json.loads(path.read_text()) for path in args.reports]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] == "no_level_completion":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
