"""Frozen visual decodability and MaleCNS bottleneck experiment for Mario GIFs.

Labels come from an investigator-supplied manifest. They are NEVER inputs to
FlyVis or MaleCNS. A result is descriptive until recording-group holdout is
possible for every class under test.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from flyvis_malecns_probe import (
    bridge_amounts,
    compare_scenes,
    load_gif,
    male_cns_trace,
    motion_grid,
    summarize,
)
from flyvis_spatial_bridge import build_target_groups, spatial_injections


VISUAL_TYPES = (
    "R1", "L1", "L2", "T4a", "T4b", "T4c", "T4d",
    "T5a", "T5b", "T5c", "T5d",
)
OBJECT_TYPES = ("floor", "goomba", "pipe", "gap")
BRIDGE_NAMES = ("small_motion", "loom", "threat", "target")


def read_manifest(path: Path, *, gif_root: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
        raise ValueError("manifest must contain a samples array")
    samples = []
    ids = set()
    source_groups = {}
    for raw in data["samples"]:
        if not isinstance(raw, dict):
            raise ValueError("each sample must be an object")
        sample_id = raw.get("id")
        label = raw.get("object")
        recording = raw.get("recording_id")
        source = raw.get("gif")
        start, stop = raw.get("start"), raw.get("stop")
        if not isinstance(sample_id, str) or not sample_id or sample_id in ids:
            raise ValueError("sample ids must be nonempty and unique")
        if label not in OBJECT_TYPES:
            raise ValueError(f"{sample_id}: object must be one of {OBJECT_TYPES}")
        if not isinstance(recording, str) or not recording:
            raise ValueError(f"{sample_id}: recording_id is required")
        if not isinstance(source, str) or not source:
            raise ValueError(f"{sample_id}: gif is required")
        if type(start) is not int or type(stop) is not int or start < 0 or stop - start < 2:
            raise ValueError(f"{sample_id}: need 0 <= start < stop-1")
        gif = (gif_root / source).resolve()
        if not gif.is_file():
            raise FileNotFoundError(gif)
        if gif in source_groups and source_groups[gif] != recording:
            raise ValueError(f"{sample_id}: same GIF cannot have two recording groups")
        source_groups[gif] = recording
        for key, correct_class in (("pipe_height_px", "pipe"), ("gap_width_px", "gap")):
            if key in raw:
                if label != correct_class or not isinstance(raw[key], (int, float)) or raw[key] <= 0:
                    raise ValueError(f"{sample_id}: invalid {key}")
        ids.add(sample_id)
        samples.append({**raw, "gif_path": gif})
    if not samples:
        raise ValueError("manifest contains no samples")
    return samples


def spatial_cell_grids(
    responses: np.ndarray, cell_types: np.ndarray, u: np.ndarray, v: np.ndarray,
    *, rows: int = 6, columns: int = 8,
) -> np.ndarray:
    """Keep cell type, visual position, and time; never pool to one threat value."""

    values = np.asarray(responses, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(cell_types):
        raise ValueError("FlyVis responses have wrong shape")
    # BoxEye maps axial (u,v) to screen x=v, y=u+v/2.
    x = v.astype(float)
    y = u.astype(float) + v.astype(float) / 2
    x_bin = np.minimum(((x - x.min()) / max(np.ptp(x), 1) * columns).astype(int), columns - 1)
    y_bin = np.minimum(((y - y.min()) / max(np.ptp(y), 1) * rows).astype(int), rows - 1)
    grids = np.zeros((len(values), len(VISUAL_TYPES), rows, columns), dtype=np.float32)
    for type_index, name in enumerate(VISUAL_TYPES):
        typed = cell_types == name
        if not typed.any():
            raise ValueError(f"FlyVis model lacks required cell type {name}")
        for row in range(rows):
            for column in range(columns):
                region = typed & (x_bin == column) & (y_bin == row)
                if region.any():
                    grids[:, type_index, row, column] = values[:, region].mean(axis=1)
    return grids


def summarize_temporal(values: np.ndarray) -> np.ndarray:
    """Fixed-size readout retaining signed response, change, and variability."""

    sequence = np.asarray(values, dtype=np.float32)
    if sequence.ndim < 2 or len(sequence) < 2:
        raise ValueError("need at least two frames of feature activity")
    summaries = (sequence.mean(axis=0), sequence.std(axis=0),
                 sequence[-1] - sequence[0],
                 np.abs(np.diff(sequence, axis=0)).mean(axis=0))
    return np.concatenate([part.ravel() for part in summaries]).astype(np.float32)


def pixel_features(frames: np.ndarray) -> np.ndarray:
    """Coarse raw-pixel baseline with the same temporal summary operator."""

    from PIL import Image

    reduced = []
    for frame in frames:
        gray = Image.fromarray(frame).convert("L").resize((16, 12), Image.Resampling.BILINEAR)
        reduced.append(np.asarray(gray, dtype=np.float32) / 255.0)
    return summarize_temporal(np.stack(reduced))


def retina_lite_features(frames: np.ndarray) -> np.ndarray:
    """The eight injected retina_lite_v2 channels, without running MaleCNS."""

    from male_cns import MaleCNS

    encoder = MaleCNS.__new__(MaleCNS)
    previous = None
    output = []
    for frame in frames:
        gray = MaleCNS._gray(frame)
        amounts = encoder._retina_lite_amounts(gray, gray if previous is None else previous)
        output.append([amounts[(name, side)] for name in BRIDGE_NAMES
                       for side in ("L", "R")])
        previous = gray
    return summarize_temporal(np.asarray(output, dtype=np.float32))


def bridge_features(amounts: list[dict[tuple[str, str], float]]) -> np.ndarray:
    """The actual eight injected values, in fixed channel/side order."""

    values = np.asarray([[frame[(name, side)] for name in BRIDGE_NAMES
                          for side in ("L", "R")] for frame in amounts], dtype=np.float32)
    return summarize_temporal(values)


def spatial_male_cns_trace(
    grids: np.ndarray,
    *,
    data: Path,
    seed: int,
    dt: float,
    silent: bool = False,
) -> np.ndarray:
    """Run MaleCNS with the spatial cell-type bridge used by online PPO."""

    from flybrain import FlyBrain, Trace

    values = np.asarray(grids, dtype=np.float32)
    if values.ndim != 4 or values.shape[1] != len(VISUAL_TYPES):
        raise ValueError("spatial grids must have shape (frames, 11, rows, columns)")
    if not (data / "brain.npz").exists() or not (data / "weights.npz").exists():
        raise FileNotFoundError(f"MaleCNS brain.npz/weights.npz missing in {data}")
    brain = FlyBrain(data=data, device="cpu", seed=seed, dt=dt)
    descending = brain.cells(["descending_neuron"])
    trace = Trace(brain, idx=descending, tau=0.12)
    targets = build_target_groups(brain, rows=values.shape[2], columns=values.shape[3])
    eye_drive = np.zeros_like(brain.azimuth, dtype=np.float32)
    output = []
    for frame in values:
        inject = [] if silent else spatial_injections(frame, targets)[0]
        spikes = brain.step(eye_drive=eye_drive, inject=inject)
        output.append(trace.observe(spikes).astype(np.float32, copy=True))
    return np.stack(output)


def grouped_classification(samples: list[dict], features: list[np.ndarray]) -> dict:
    """Leave one recording out; refuse folds missing a class in training."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels = np.asarray([sample["object"] for sample in samples])
    groups = np.asarray([sample["recording_id"] for sample in samples])
    classes = sorted(set(labels))
    counts = Counter(labels)
    if len(classes) < 2 or len(set(groups)) < 2 or min(counts.values()) < 2:
        return {"status": "insufficient_data", "reason":
                "need at least two classes, two recordings, and two samples per class"}
    x = np.stack(features)
    predictions = []
    for group in sorted(set(groups)):
        test = groups == group
        train = ~test
        if set(labels[train]) != set(classes):
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
        model.fit(x[train], labels[train])
        predictions.extend({"id": samples[index]["id"], "group": group,
                            "true": labels[index], "predicted": predicted}
                           for index, predicted in zip(np.flatnonzero(test), model.predict(x[test]), strict=True))
    if not predictions or set(row["true"] for row in predictions) != set(classes):
        return {"status": "insufficient_data", "reason":
                "no valid held-out recording folds cover every class"}
    truth = [row["true"] for row in predictions]
    predicted = [row["predicted"] for row in predictions]
    return {"status": "ok", "classes": classes, "held_out_samples": len(predictions),
            "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
            "confusion_matrix": confusion_matrix(truth, predicted, labels=classes).tolist(),
            "predictions": predictions}


def grouped_regression(samples: list[dict], features: list[np.ndarray], field: str) -> dict:
    """Held-out width/height MAE versus a train-fold median baseline."""

    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    selected = [(sample, feature) for sample, feature in zip(samples, features, strict=True)
                if field in sample]
    if len(selected) < 4 or len({item[0]["recording_id"] for item in selected}) < 2:
        return {"status": "insufficient_data", "reason":
                f"need at least four annotated {field} clips in two recordings"}
    x = np.stack([item[1] for item in selected])
    y = np.asarray([item[0][field] for item in selected], dtype=float)
    groups = np.asarray([item[0]["recording_id"] for item in selected])
    rows = []
    for group in sorted(set(groups)):
        test = groups == group
        train = ~test
        if train.sum() < 2:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(x[train], y[train])
        baseline = float(np.median(y[train]))
        for index, predicted in zip(np.flatnonzero(test), model.predict(x[test]), strict=True):
            rows.append({"id": selected[index][0]["id"], "true": float(y[index]),
                         "predicted": float(predicted), "baseline": baseline})
    if len(rows) < len(selected):
        return {"status": "insufficient_data", "reason": "some recording folds lack training clips"}
    return {"status": "ok", "held_out_samples": len(rows),
            "mean_absolute_error_px": float(np.mean([abs(row["predicted"] - row["true"]) for row in rows])),
            "median_baseline_mae_px": float(np.mean([abs(row["baseline"] - row["true"]) for row in rows])),
            "predictions": rows}


class FlyVisRunner:
    """Load pretrained weights once, then reset network state for each clip."""

    def __init__(self, model_dir: Path):
        import flyvis
        from flyvis.datasets.rendering import BoxEye

        if not model_dir.is_dir():
            raise FileNotFoundError(model_dir)
        self.flyvis = flyvis
        self.retina = BoxEye(extent=15, kernel_size=13)
        self.network = flyvis.NetworkView(model_dir).init_network()
        self.network.eval()
        table = self.network.connectome.nodes.to_df()
        self.cell_types = table["type"].astype(str).to_numpy()
        self.u = table["u"].to_numpy(dtype=float)
        self.v = table["v"].to_numpy(dtype=float)

    def run(self, frames: np.ndarray, dt: float) -> np.ndarray:
        import torch

        gray = np.asarray(frames, dtype=np.float32) / 255.0
        gray = gray @ np.asarray((0.299, 0.587, 0.114), dtype=np.float32)
        with torch.no_grad():
            movie = self.retina(torch.from_numpy(gray)[None].to(self.flyvis.device))
            initial = self.network.fade_in_state(1.0, dt, movie[0, [0]])
            response = self.network.simulate(movie, dt, initial_state=initial)
        return response[0].detach().cpu().numpy()


def run_experiment(samples: list[dict], *, model_dir: Path, male_cns_data: Path,
                   fps: float, seed: int, max_frames: int) -> dict:
    if fps < 50 or max_frames < 2:
        raise ValueError("FlyVis integration needs fps >= 50 and at least two frames")
    runner = FlyVisRunner(model_dir)
    features = {name: [] for name in ("flyvis_spatial", "flyvis_motion_grid",
                                     "bridge_8ch", "male_cns_visual", "male_cns",
                                     "male_cns_spatial", "raw_pixels",
                                     "retina_lite_channels")}
    scenes = {}
    for sample in samples:
        frames = load_gif(sample["gif_path"], start=sample["start"], stop=sample["stop"], limit=max_frames)
        activity = runner.run(frames, 1.0 / fps)
        grids = spatial_cell_grids(activity, runner.cell_types, runner.u, runner.v)
        motion = motion_grid(activity, runner.cell_types, runner.u, runner.v)
        amounts = bridge_amounts(motion)
        driven, visual_driven = male_cns_trace(
            amounts, data=male_cns_data, seed=seed, dt=1.0 / fps, include_visual=True)
        silent, visual_silent = male_cns_trace(
            amounts, data=male_cns_data, seed=seed, dt=1.0 / fps,
            silent=True, include_visual=True)
        spatial_driven = spatial_male_cns_trace(
            grids, data=male_cns_data, seed=seed, dt=1.0 / fps)
        spatial_silent = spatial_male_cns_trace(
            grids, data=male_cns_data, seed=seed, dt=1.0 / fps, silent=True)
        features["flyvis_spatial"].append(summarize_temporal(grids))
        features["flyvis_motion_grid"].append(summarize_temporal(motion))
        features["bridge_8ch"].append(bridge_features(amounts))
        features["male_cns_visual"].append(summarize_temporal(visual_driven - visual_silent))
        features["raw_pixels"].append(pixel_features(frames))
        features["retina_lite_channels"].append(retina_lite_features(frames))
        features["male_cns"].append(summarize_temporal(driven - silent))
        spatial_difference = spatial_driven - spatial_silent
        features["male_cns_spatial"].append(summarize_temporal(spatial_difference))
        scenes[sample["id"]] = {"object": sample["object"], "recording_id": sample["recording_id"],
                                "gif": str(sample["gif_path"]), "source_frame_window":
                                [sample["start"], sample["stop"]],
                                "male_cns_spatial_descending_mean_absolute_change_vs_silent":
                                float(np.mean(np.abs(spatial_difference))),
                                "male_cns_spatial_descending_rms_change_vs_silent":
                                float(np.sqrt(np.mean(spatial_difference ** 2))),
                                **summarize(motion, amounts, driven, silent)}
    evaluations = {}
    for name, vectors in features.items():
        evaluations[name] = {
            "object_classification": grouped_classification(samples, vectors),
            "pipe_height": grouped_regression(samples, vectors, "pipe_height_px"),
            "gap_width": grouped_regression(samples, vectors, "gap_width_px"),
        }
    return {"warning": "Scene labels are annotations, not FlyVis detections; paired backgrounds are needed for causal claims.",
            "samples": scenes, "comparisons": compare_scenes(scenes), "evaluations": evaluations}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gif-root", type=Path, required=True)
    parser.add_argument("--flyvis-model", type=Path, required=True)
    parser.add_argument("--male-cns-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    samples = read_manifest(args.manifest, gif_root=args.gif_root)
    report = run_experiment(samples, model_dir=args.flyvis_model,
                            male_cns_data=args.male_cns_data, fps=args.fps,
                            seed=args.seed, max_frames=args.max_frames)
    report["manifest"] = str(args.manifest)
    report["flyvis_model"] = str(args.flyvis_model)
    report["male_cns_data"] = str(args.male_cns_data)
    report["seed"] = args.seed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
