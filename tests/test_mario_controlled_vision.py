"""Controlled stimuli preserve labels and vary geometry independently."""

import importlib.util
import json
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "integrations/super_mario/mario_controlled_vision.py"
    spec = importlib.util.spec_from_file_location("mario_controlled_vision_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gap_and_pipe_geometry_are_visible():
    stimuli = _module()
    floor = np.asarray(stimuli.render_mario_frames("floor", x=125, size=32,
                                                   speed=2, style=0, frame_count=2)[0])
    gap = np.asarray(stimuli.render_mario_frames("gap", x=125, size=32,
                                                 speed=2, style=0, frame_count=2)[0])
    pipe = np.asarray(stimuli.render_mario_frames("pipe", x=125, size=48,
                                                  speed=2, style=0, frame_count=2)[0])
    assert not np.array_equal(floor[220, 140], gap[220, 140])
    assert np.array_equal(floor[220, 165], gap[220, 165])
    assert not np.array_equal(floor[165, 140], pipe[165, 140])


def test_edge_moves_both_directions():
    stimuli = _module()
    left = stimuli.render_edge_frames(direction="left", contrast="on", speed=2,
                                     frame_count=3)
    right = stimuli.render_edge_frames(direction="right", contrast="on", speed=2,
                                      frame_count=3)
    assert np.asarray(left[-1])[100, 126, 0] != np.asarray(right[-1])[100, 126, 0]


def test_dataset_has_factorial_labels_and_manifest(tmp_path):
    stimuli = _module()
    report = stimuli.build_dataset(tmp_path, frame_count=2)
    samples = json.loads((tmp_path / "manifest.json").read_text())["samples"]
    edges = json.loads((tmp_path / "edges.json").read_text())["edges"]
    assert report["hazard_samples"] == len(samples) == 48
    assert report["edge_samples"] == len(edges) == 12
    assert {sample["object"] for sample in samples} == set(stimuli.OBJECTS)
    assert {sample["recording_id"] for sample in samples} == {
        "synthetic-style-0", "synthetic-style-1", "synthetic-style-2"}
    assert {sample["pipe_height_px"] for sample in samples if sample["object"] == "pipe"} == {32, 48}
    assert {sample["gap_width_px"] for sample in samples if sample["object"] == "gap"} == {32, 48}
