"""Select a frozen deterministic checkpoint; never reward survival duration."""
import argparse
import json
import shutil
from pathlib import Path


def selection_key(report):
    if (report.get("mode") != "mean" or report.get("learning_enabled") is not False
            or report.get("visual_ablation") != "normal" or not report.get("episodes")
            or not report.get("bounded_jump_actions")):
        raise ValueError("Need a frozen, normal, deterministic bounded-action report")
    summary = report["summary"]
    return [summary["completion_rate"], summary["mean_max_x"], summary["mean_raw_reward"],
            -summary["mean_max_no_progress_frames"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--best", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    key = selection_key(report)
    history = json.loads(args.history.read_text()) if args.history.exists() else []
    if history and any(row["seed"] != report["seed"] for row in history):
        raise ValueError("Progress evaluations must use the same seed")
    selected = not history or key > max(row["selection_key"] for row in history)
    row = {"report": str(args.report), "checkpoint": report["checkpoint"],
           "seed": report["seed"], "selection_key": key, "summary": report["summary"],
           "selected": selected}
    if selected:
        tmp = args.best.with_name(args.best.name + ".tmp")
        shutil.copy2(report["checkpoint"], tmp)
        tmp.replace(args.best)
    history.append(row)
    tmp = args.history.with_name(args.history.name + ".tmp")
    tmp.write_text(json.dumps(history, indent=2) + "\n")
    tmp.replace(args.history)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
