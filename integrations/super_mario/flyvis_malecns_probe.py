"""Offline, frozen FlyVis -> MaleCNS probe for recorded Mario GIFs.

This is an experimental interface, not an anatomical synapse map or a Mario
object detector. FlyVis responses are kept spatial until the explicitly
reported left/right projection into MaleCNS visual-projection populations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MOTION_TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
BRIDGE_TARGETS = {
    "small_motion": "LPLC1",
    "loom": "LPLC2",
    "threat": "LC4",
    "target": "LC10a",
}


def load_gif(
    path: Path, *, start: int = 0, stop: int | None = None,
    stride: int = 1, limit: int = 120,
) -> np.ndarray:
    """Load equally spaced recorded frames without changing the source GIF."""

    from PIL import Image

    if stride < 1 or limit < 2 or start < 0 or (stop is not None and stop <= start):
        raise ValueError("invalid GIF window, stride, or limit")
    with Image.open(path) as gif:
        frame_count = getattr(gif, "n_frames", 1)
        end = min(frame_count, stop if stop is not None else frame_count, start + stride * limit)
        indices = range(start, end, stride)
        frames = []
        for index in indices:
            gif.seek(index)
            # Forensic GIFs append text panels below the 256x240 NES image.
            # Never let those labels become visual input to FlyVis.
            rgb = np.asarray(gif.convert("RGB"), dtype=np.uint8)
            frames.append(rgb[:240].copy() if rgb.shape[0] >= 240 else rgb)
    if len(frames) < 2:
        raise ValueError(f"need at least two frames: {path}")
    return np.stack(frames)


def parse_scene(specification: str) -> tuple[str, Path, int, int | None]:
    """Parse LABEL=GIF[@START:STOP]; frame indices refer to source GIF frames."""

    if "=" not in specification:
        raise ValueError("--scene must be LABEL=GIF[@START:STOP]")
    label, value = specification.split("=", 1)
    if not label or not value:
        raise ValueError("scene label and GIF path must be nonempty")
    if "@" not in value:
        return label, Path(value), 0, None
    filename, window = value.rsplit("@", 1)
    try:
        start_text, stop_text = window.split(":", 1)
        start = int(start_text)
        stop = int(stop_text)
    except ValueError as exc:
        raise ValueError("window must be @START:STOP") from exc
    if not filename or start < 0 or stop <= start:
        raise ValueError("scene window must have 0 <= START < STOP")
    return label, Path(filename), start, stop


def _node_table(connectome) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = connectome.nodes.to_df()
    cell_type = table["type"].astype(str).to_numpy()
    return cell_type, table["u"].to_numpy(dtype=float), table["v"].to_numpy(dtype=float)


def motion_grid(
    responses: np.ndarray,
    cell_types: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    rows: int = 3,
    columns: int = 4,
) -> np.ndarray:
    """Bin FlyVis motion-cell activity in visual-lattice coordinates.

    Binning uses the same coordinates at every time step and does not infer
    object identities. It retains height and horizontal position for probing.
    """

    activity = np.asarray(responses, dtype=np.float32)
    if activity.ndim != 2 or activity.shape[1] != len(cell_types):
        raise ValueError("responses must have shape (frames, FlyVis cells)")
    mask = np.isin(cell_types, MOTION_TYPES)
    if not mask.any():
        raise ValueError("FlyVis connectome has no expected T4/T5 motion cells")
    selected = activity[:, mask]
    if not np.isfinite(selected).all():
        raise ValueError("FlyVis activity contains non-finite values")
    # BoxEye renders axial hex coordinates as x=v, y=u+v/2 (up to scale).
    # Using u as screen x would silently swap/tilt the Mario scene.
    x, y = v[mask], u[mask] + v[mask] / 2.0
    x_bin = np.minimum(((x - x.min()) / max(np.ptp(x), 1) * columns).astype(int), columns - 1)
    y_bin = np.minimum(((y - y.min()) / max(np.ptp(y), 1) * rows).astype(int), rows - 1)
    grid = np.zeros((len(activity), rows, columns), dtype=np.float32)
    for row in range(rows):
        for column in range(columns):
            region = (y_bin == row) & (x_bin == column)
            if region.any():
                grid[:, row, column] = np.mean(np.abs(selected[:, region]), axis=1)
    return grid


def bridge_amounts(grid: np.ndarray) -> list[dict[tuple[str, str], float]]:
    """Deterministic, bounded experimental projection into named MaleCNS cells.

    The first half of the lattice is L and the second R. There is no claim
    that this is a measured FlyVis-to-MaleCNS synaptic connection. We report
    the complete grid separately to expose any information lost here.
    """

    values = np.asarray(grid, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (3, 4):
        raise ValueError("grid must have shape (frames, 3, 4)")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("grid must contain finite nonnegative values")
    result = []
    for index, frame in enumerate(values):
        previous = values[index - 1] if index else frame
        amounts = {}
        for side, region in (("L", slice(0, 2)), ("R", slice(2, 4))):
            # Spatially localized response and its rise across recorded frames.
            lower = float(frame[1:, region].mean())
            lower_before = float(previous[1:, region].mean())
            upper = float(frame[:2, region].mean())
            rise = max(0.0, lower - lower_before)
            amounts[("small_motion", side)] = min(0.8, 0.40 * lower)
            amounts[("loom", side)] = min(0.8, 0.55 * rise)
            amounts[("threat", side)] = min(0.8, 0.35 * lower + 0.35 * rise)
            amounts[("target", side)] = min(0.8, 0.20 * upper)
        result.append(amounts)
    return result


def flyvis_responses(frames: np.ndarray, *, model_dir: Path, dt: float) -> tuple[np.ndarray, tuple]:
    """Run the published pretrained FlyVis network on a Mario movie."""

    import flyvis
    import torch
    from flyvis.datasets.rendering import BoxEye

    if not model_dir.exists():
        raise FileNotFoundError(f"FlyVis checkpoint missing: {model_dir}")
    gray = np.asarray(frames, dtype=np.float32) / 255.0
    gray = gray @ np.asarray((0.299, 0.587, 0.114), dtype=np.float32)
    with torch.no_grad():
        retina = BoxEye(extent=15, kernel_size=13)
        movie = retina(torch.from_numpy(gray)[None].to(flyvis.device))
        network = flyvis.NetworkView(model_dir).init_network()
        network.eval()
        initial = network.fade_in_state(1.0, dt, movie[0, [0]])
        activity = network.simulate(movie, dt, initial_state=initial)
    values = activity[0].detach().cpu().numpy()
    return values, _node_table(network.connectome)


def male_cns_trace(
    amounts: list[dict[tuple[str, str], float]],
    *,
    data: Path,
    seed: int,
    dt: float,
    silent: bool = False,
    include_visual: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Run identical-noise MaleCNS trials with or without FlyVis drive."""

    from flybrain import FlyBrain, Trace

    if not (data / "brain.npz").exists() or not (data / "weights.npz").exists():
        raise FileNotFoundError(f"MaleCNS brain.npz/weights.npz missing in {data}")
    brain = FlyBrain(data=data, device="cpu", seed=seed, dt=dt)
    descending = brain.cells(["descending_neuron"])
    trace = Trace(brain, idx=descending, tau=0.12)
    targets = {
        (name, side): brain.cells([cell_type], side=side)
        for name, cell_type in BRIDGE_TARGETS.items()
        for side in ("L", "R")
    }
    if any(len(indices) == 0 for indices in targets.values()):
        raise RuntimeError("MaleCNS lacks an expected visual-projection population")
    visual_trace = None
    if include_visual:
        visual_indices = np.unique(np.concatenate(list(targets.values())))
        visual_trace = Trace(brain, idx=visual_indices, tau=0.12)
    eye_drive = np.zeros_like(brain.azimuth, dtype=np.float32)
    output = []
    visual_output = []
    for frame in amounts:
        inject = [] if silent else [
            (targets[key], value) for key, value in frame.items() if value > 0
        ]
        spikes = brain.step(eye_drive=eye_drive, inject=inject)
        output.append(trace.observe(spikes).astype(np.float32, copy=True))
        if visual_trace is not None:
            visual_output.append(visual_trace.observe(spikes).astype(np.float32, copy=True))
    if visual_trace is not None:
        return np.stack(output), np.stack(visual_output)
    return np.stack(output)


def summarize(
    grid: np.ndarray,
    amounts: list[dict[tuple[str, str], float]],
    driven: np.ndarray,
    silent: np.ndarray,
) -> dict:
    """Compare spatial FlyVis drive and paired MaleCNS descending traces."""

    if driven.shape != silent.shape or driven.shape[0] != len(grid):
        raise ValueError("FlyVis and MaleCNS frame counts must match")
    difference = driven - silent
    mean_change = np.mean(np.abs(difference), axis=0)
    top = np.argsort(mean_change)[-10:][::-1]
    return {
        "frames": len(grid),
        "flyvis_motion_grid_mean": grid.mean(axis=0).round(6).tolist(),
        "flyvis_motion_grid_peak": grid.max(axis=0).round(6).tolist(),
        "bridge_mean": {
            f"{name}_{side.lower()}": float(np.mean([row[(name, side)] for row in amounts]))
            for name in BRIDGE_TARGETS for side in ("L", "R")
        },
        "male_cns_descending_mean_absolute_change_vs_silent": float(mean_change.mean()),
        "male_cns_descending_rms_change_vs_silent": float(np.sqrt(np.mean(difference ** 2))),
        "male_cns_most_changed_descending_indices": top.astype(int).tolist(),
        "male_cns_most_changed_descending_amounts": mean_change[top].round(6).tolist(),
        "male_cns_driven_mean_trace": driven.mean(axis=0).round(6).tolist(),
        "male_cns_mean_delta_vs_silent": difference.mean(axis=0).round(6).tolist(),
    }


def compare_scenes(scenes: dict[str, dict]) -> list[dict]:
    """Pairwise distances show whether scene distinctions survive the bridge."""

    labels = list(scenes)
    comparisons = []
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1:]:
            first = np.asarray(scenes[left]["male_cns_mean_delta_vs_silent"], dtype=float)
            second = np.asarray(scenes[right]["male_cns_mean_delta_vs_silent"], dtype=float)
            left_grid = np.asarray(scenes[left]["flyvis_motion_grid_mean"], dtype=float)
            right_grid = np.asarray(scenes[right]["flyvis_motion_grid_mean"], dtype=float)
            comparisons.append({
                "left": left,
                "right": right,
                "flyvis_grid_rms_distance": float(np.sqrt(np.mean((left_grid - right_grid) ** 2))),
                "male_cns_delta_rms_distance": float(np.sqrt(np.mean((first - second) ** 2))),
            })
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", action="append", required=True, metavar="LABEL=GIF[@START:STOP]")
    parser.add_argument("--flyvis-model", type=Path, required=True)
    parser.add_argument("--male-cns-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.fps <= 0 or args.stride < 1 or args.stride / args.fps > 0.02:
        parser.error("FlyVis integration requires 0 < stride/fps <= 0.02 seconds")
    result = {
        "note": "Exploratory projection; FlyVis does not assign Mario object labels.",
        "model": str(args.flyvis_model),
        "stride": args.stride,
        "scenes": {},
    }
    for item in args.scene:
        try:
            label, filename, start, stop = parse_scene(item)
        except ValueError as exc:
            parser.error(str(exc))
        if label in result["scenes"]:
            parser.error("scene labels must be unique")
        frames = load_gif(filename, start=start, stop=stop, stride=args.stride, limit=args.limit)
        dt = args.stride / args.fps
        responses, (types, u, v) = flyvis_responses(
            frames, model_dir=args.flyvis_model, dt=dt
        )
        grid = motion_grid(responses, types, u, v)
        amounts = bridge_amounts(grid)
        driven = male_cns_trace(amounts, data=args.male_cns_data, seed=args.seed, dt=dt)
        silent = male_cns_trace(
            amounts, data=args.male_cns_data, seed=args.seed, dt=dt, silent=True
        )
        result["scenes"][label] = {
            "source": str(filename),
            "source_frame_window": [start, stop],
            **summarize(grid, amounts, driven, silent),
        }
    result["pairwise_scene_comparisons"] = compare_scenes(result["scenes"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
