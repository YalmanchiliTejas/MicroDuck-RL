"""Measure signed, direction-specific FlyVis responses to controlled moving edges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flyvis_malecns_probe import load_gif
from flyvis_perception_experiment import FlyVisRunner, VISUAL_TYPES, spatial_cell_grids


def run_calibration(edge_index: Path, *, gif_root: Path, model_dir: Path,
                    fps: float = 60.0) -> dict:
    if fps < 50:
        raise ValueError("FlyVis integration needs fps >= 50")
    entries = json.loads(edge_index.read_text(encoding="utf-8"))["edges"]
    runner = FlyVisRunner(model_dir)
    rows = []
    for entry in entries:
        frames = load_gif(gif_root / entry["gif"], start=entry["start"], stop=entry["stop"])
        grids = spatial_cell_grids(runner.run(frames, 1.0 / fps),
                                   runner.cell_types, runner.u, runner.v)
        early = grids[:max(1, len(grids) // 4)].mean(axis=0)
        late = grids[-max(1, len(grids) // 4):].mean(axis=0)
        change = late - early
        rows.append({**entry, "signed_change_by_type": {
            name: float(change[index].mean()) for index, name in enumerate(VISUAL_TYPES)},
            "spatial_change_by_type": {
                name: change[index].round(6).tolist()
                for index, name in enumerate(VISUAL_TYPES)},
        })
    direction_contrasts = []
    for contrast in ("on", "off"):
        speeds = sorted({row["speed_px_per_frame"] for row in rows
                         if row["contrast"] == contrast})
        for speed in speeds:
            pair = {row["direction"]: row for row in rows
                    if row["contrast"] == contrast and row["speed_px_per_frame"] == speed}
            if set(pair) != {"left", "right"}:
                raise ValueError(f"missing matched direction for {contrast}, speed {speed}")
            direction_contrasts.append({"contrast": contrast, "speed_px_per_frame": speed,
                                        "right_minus_left_signed_change": {
                name: pair["right"]["signed_change_by_type"][name]
                - pair["left"]["signed_change_by_type"][name]
                for name in VISUAL_TYPES}})
    return {"note": "Diagnostic directional contrasts, not a formal direction-selectivity index, Mario object recognition, or calibrated geometry.",
            "stimuli": rows, "direction_contrasts": direction_contrasts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-index", type=Path, required=True)
    parser.add_argument("--gif-root", type=Path, required=True)
    parser.add_argument("--flyvis-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=60.0)
    args = parser.parse_args()
    report = run_calibration(args.edge_index, gif_root=args.gif_root,
                             model_dir=args.flyvis_model, fps=args.fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
