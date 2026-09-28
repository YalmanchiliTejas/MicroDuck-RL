"""Regression for evaluation denominators and pre-reset command labels."""

import importlib.util
from pathlib import Path

import torch


def test_command_counts_exclude_transition_and_fall_success():
    path = Path(__file__).parents[1] / "scripts/evaluate_mario_controller.py"
    spec = importlib.util.spec_from_file_location("mario_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    commands = torch.tensor([
        [0., 0., 0.], [-1., 0., 0.], [-1., 0., 0.],
        [1., 0., 0.], [0., 0., 1.], [0., 0., 1.],
    ])
    counts = module.command_counts(
        commands, torch.tensor([1, 1, 0, 1, 1, 1]),
        torch.tensor([1, 1, 1, 0, 1, 1]),
        torch.tensor([0, 0, 0, 0, 0, 1]),
    )
    assert counts.tolist() == [[1, 1], [1, 1], [1, 0], [2, 1], [0, 0], [0, 0]]
