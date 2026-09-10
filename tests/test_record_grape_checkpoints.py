"""Unit tests for checkpoint-video watcher bookkeeping (no simulator needed)."""

import importlib.util
import sys
from pathlib import Path


_SCRIPT = Path(__file__).parents[1] / "scripts" / "record_grape_checkpoints.py"
_SPEC = importlib.util.spec_from_file_location("record_grape_checkpoints", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
watcher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(watcher)


def test_checkpoint_iteration_accepts_only_standard_checkpoint_names():
    assert watcher.checkpoint_iteration(Path("model_250.pt")) == 250
    assert watcher.checkpoint_iteration(Path("model_final.pt")) is None
    assert watcher.checkpoint_iteration(Path("checkpoint_250.pt")) is None


def test_find_checkpoints_sorts_numeric_iterations_across_run_dirs(tmp_path):
    for relative_path in (
        "run_b/model_1000.pt",
        "run_a/model_250.pt",
        "run_c/not_a_checkpoint.pt",
    ):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    assert [path.name for path in watcher.find_checkpoints(tmp_path)] == [
        "model_250.pt",
        "model_1000.pt",
    ]


def test_completion_marker_is_isolated_per_iteration(tmp_path):
    assert watcher.completion_marker(tmp_path, 250) == (
        tmp_path / "model_250" / "complete.json"
    )


def test_default_video_covers_full_six_second_cycle(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["record_grape_checkpoints.py", "--checkpoint-dir", str(tmp_path)],
    )
    assert watcher.parse_args().video_length == 300
