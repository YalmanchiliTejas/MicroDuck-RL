"""CPU-only contract tests for the optional FlyVis/MaleCNS comparison probe."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _probe():
    path = Path(__file__).parents[1] / "integrations/super_mario/flyvis_malecns_probe.py"
    spec = importlib.util.spec_from_file_location("flyvis_malecns_probe_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_scene_windows_and_gif_loading(tmp_path):
    probe = _probe()
    frames = [Image.fromarray(np.full((300, 8, 3), number, dtype=np.uint8)) for number in range(6)]
    gif = tmp_path / "mario.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=20)
    assert probe.parse_scene(f"pipe={gif}@1:5") == ("pipe", gif, 1, 5)
    selected = probe.load_gif(gif, start=1, stop=5, stride=2)
    assert selected.shape == (2, 240, 8, 3)
    assert selected[:, 0, 0, 0].tolist() == [1, 3]
    with pytest.raises(ValueError, match="window"):
        probe.parse_scene(f"pipe={gif}@5:1")


def test_spatial_grid_and_deterministic_bridge():
    probe = _probe()
    types = np.array(["T4a"] * 12 + ["unrelated"])
    u = np.tile(np.arange(4), 3).astype(float)
    v = np.repeat(np.arange(3), 4).astype(float)
    responses = np.zeros((2, 13), dtype=np.float32)
    responses[1, 10:12] = 1.0  # lower-right motion
    grid = probe.motion_grid(responses, types, np.r_[u, 0], np.r_[v, 0])
    assert grid.shape == (2, 3, 4)
    assert grid[1, 2, 3] == 1.0
    amounts = probe.bridge_amounts(grid)
    assert amounts[0][("loom", "R")] == 0
    assert amounts[1][("loom", "R")] > amounts[1][("loom", "L")]
    assert amounts[1][("threat", "R")] > amounts[1][("threat", "L")]
    assert probe.bridge_amounts(grid) == amounts


def test_male_cns_paired_silent_trial(monkeypatch, tmp_path):
    probe = _probe()
    (tmp_path / "brain.npz").touch()
    (tmp_path / "weights.npz").touch()
    calls = []

    class Brain:
        def __init__(self, **kwargs):
            self.azimuth = np.zeros(3, dtype=np.float32)
            self.last_inject = []

        def cells(self, names, side=None):
            return np.array([1 if side == "L" else 2])

        def step(self, *, eye_drive, inject):
            assert not eye_drive.any()
            calls.append(len(inject))
            return np.array([len(inject)])

    class Trace:
        def __init__(self, brain, *, idx, tau):
            pass

        def observe(self, fired):
            return np.array([fired[0]], dtype=np.float32)

    monkeypatch.setitem(sys.modules, "flybrain", types.SimpleNamespace(FlyBrain=Brain, Trace=Trace))
    amounts = [{("small_motion", "R"): 0.25}] * 3
    driven = probe.male_cns_trace(amounts, data=tmp_path, seed=123, dt=0.02)
    silent = probe.male_cns_trace(amounts, data=tmp_path, seed=123, dt=0.02, silent=True)
    assert driven[:, 0].tolist() == [1, 1, 1]
    assert silent[:, 0].tolist() == [0, 0, 0]
    assert calls == [1, 1, 1, 0, 0, 0]


def test_probe_summary_reports_effect_without_claiming_object_identity():
    probe = _probe()
    grid = np.zeros((2, 3, 4), dtype=np.float32)
    amounts = probe.bridge_amounts(grid)
    driven = np.array([[1, 0], [1, 0]], dtype=np.float32)
    silent = np.zeros_like(driven)
    report = probe.summarize(grid, amounts, driven, silent)
    assert report["frames"] == 2
    assert report["male_cns_descending_mean_absolute_change_vs_silent"] == 0.5
    assert report["male_cns_most_changed_descending_indices"][0] == 0
