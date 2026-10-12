"""Tests for manifest validation, spatial readouts, and held-out scoring."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _module():
    directory = Path(__file__).parents[1] / "integrations/super_mario"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(
            "flyvis_perception_experiment_test", directory / "flyvis_perception_experiment.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(directory))


def test_manifest_rejects_same_gif_split_across_groups(tmp_path):
    experiment = _module()
    (tmp_path / "episode.gif").touch()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": [
        {"id": "one", "object": "floor", "recording_id": "A", "gif": "episode.gif", "start": 0, "stop": 5},
        {"id": "two", "object": "gap", "recording_id": "B", "gif": "episode.gif", "start": 10, "stop": 15},
    ]}))
    with pytest.raises(ValueError, match="same GIF"):
        experiment.read_manifest(manifest, gif_root=tmp_path)


def test_spatial_grids_keep_cell_type_and_position():
    experiment = _module()
    cell_types = np.array(experiment.VISUAL_TYPES)
    activity = np.zeros((3, len(cell_types)), dtype=np.float32)
    activity[1, 0] = 2.0
    activity[1, 1] = -3.0
    u = np.arange(len(cell_types), dtype=float)
    v = np.arange(len(cell_types), dtype=float)
    grids = experiment.spatial_cell_grids(activity, cell_types, u, v)
    assert grids.shape == (3, len(cell_types), 6, 8)
    assert np.isclose(grids[1, 0].sum(), 2.0)
    assert np.isclose(grids[1, 1].sum(), -3.0)
    summary = experiment.summarize_temporal(grids)
    assert summary.shape == (4 * len(cell_types) * 6 * 8,)


def test_grouped_scores_refuse_single_examples():
    experiment = _module()
    pytest.importorskip("sklearn")
    samples = [
        {"id": "a", "object": "floor", "recording_id": "1"},
        {"id": "b", "object": "gap", "recording_id": "2"},
    ]
    vectors = [np.array([0.0]), np.array([1.0])]
    assert experiment.grouped_classification(samples, vectors)["status"] == "insufficient_data"
    assert experiment.grouped_regression(samples, vectors, "gap_width_px")["status"] == "insufficient_data"


def test_grouped_classification_holds_out_recordings():
    experiment = _module()
    pytest.importorskip("sklearn")
    samples = []
    vectors = []
    for recording in ("A", "B", "C"):
        for label, value in (("floor", -3), ("gap", 3)):
            samples.append({"id": f"{recording}-{label}", "recording_id": recording, "object": label})
            vectors.append(np.array([value], dtype=np.float32))
    score = experiment.grouped_classification(samples, vectors)
    assert score["status"] == "ok"
    assert score["held_out_samples"] == 6
    assert score["balanced_accuracy"] == 1.0


def test_grouped_regression_reports_median_baseline():
    experiment = _module()
    pytest.importorskip("sklearn")
    samples = [{"id": f"p{i}", "recording_id": str(i // 2), "object": "pipe",
                "pipe_height_px": float(i + 10)} for i in range(6)]
    vectors = [np.array([i], dtype=np.float32) for i in range(6)]
    score = experiment.grouped_regression(samples, vectors, "pipe_height_px")
    assert score["status"] == "ok"
    assert score["held_out_samples"] == 6
    assert score["median_baseline_mae_px"] > 0
