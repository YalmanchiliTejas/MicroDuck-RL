"""Regression tests for automatic Mario checkpoint selection."""

import importlib.util
from pathlib import Path

import pytest


def load_selector():
    path = Path(__file__).parents[1] / "scripts/select_mario_checkpoint.py"
    spec = importlib.util.spec_from_file_location("mario_selector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(fractions, falls=20, completed=100):
    names = ("neutral", "left", "right", "jump", "left_jump", "right_jump")
    return {
        "commands": {
            name: {"clean_fraction": value}
            for name, value in zip(names, fractions, strict=True)
        },
        "falls": falls,
        "completed_episodes": completed,
    }


def test_balance_first_score_rejects_one_broken_command():
    selector = load_selector()
    balanced = selector.score_report(report([0.4] * 6))
    broken = selector.score_report(report([0.8] * 5 + [0.01]))
    assert balanced["overall"] > broken["overall"]
    assert balanced["weakest_command"] == pytest.approx(0.4)


def test_score_rewards_fewer_falls_when_skills_match():
    selector = load_selector()
    safer = selector.score_report(report([0.3] * 6, falls=10))
    riskier = selector.score_report(report([0.3] * 6, falls=50))
    assert safer["overall"] > riskier["overall"]


def test_checkpoint_discovery_is_numeric_filtered_and_sorted(tmp_path):
    selector = load_selector()
    for name in ("model_6500.pt", "model_6250.pt", "model_final.pt", "other.pt"):
        (tmp_path / name).touch()
    found = selector.discover_checkpoints(
        tmp_path, min_iteration=6000, max_iteration=6500, every=250
    )
    assert [(iteration, path.name) for iteration, path in found] == [
        (6250, "model_6250.pt"),
        (6500, "model_6500.pt"),
    ]


def test_checkpoint_discovery_rejects_duplicate_iterations(tmp_path):
    selector = load_selector()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "model_6250.pt").touch()
    (tmp_path / "b" / "model_6250.pt").touch()
    with pytest.raises(ValueError, match="same iteration"):
        selector.discover_checkpoints(tmp_path)
