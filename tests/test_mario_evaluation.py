"""Regression for evaluation denominators and pre-reset command labels."""

import importlib.util
from pathlib import Path

import torch


def load_evaluator():
    path = Path(__file__).parents[1] / "scripts/evaluate_mario_controller.py"
    spec = importlib.util.spec_from_file_location("mario_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_transitions_track_exposure_and_do_not_cross_episode_resets():
    module = load_evaluator()
    tracker = module.TransitionDiagnostics(1, "cpu")
    def step(command, age, fall=False, done=False):
        tracker.update(torch.tensor([command]), torch.tensor([age]),
                       torch.tensor([fall]), torch.tensor([done]))
    step(module.COMMANDS["left"], 0.)
    step(module.COMMANDS["left"], 0.4)
    step(module.COMMANDS["neutral"], 0.)
    step(module.COMMANDS["neutral"], 0.2, fall=True, done=True)
    step(module.COMMANDS["right"], 0.)
    step(module.COMMANDS["right"], 0.8)
    # A resample of the same category is a separate command window.
    step(module.COMMANDS["right"], 0.)
    step(module.COMMANDS["right"], 0.8, fall=True, done=True)
    report = tracker.report(.02)
    assert "neutral->right" not in report
    assert report["episode_start->right"]["command_windows_started"] == 1
    assert report["left->neutral"]["falls"] == 1
    assert report["left->neutral"]["falls_within_0_5s"] == 1
    assert report["left->neutral"]["exposure_seconds"] == .04
    assert report["right->right"]["falls"] == 1
    assert report["right->right"]["falls_within_0_5s"] == 0


def test_neutral_failure_components_match_success_without_changing_it(monkeypatch):
    from types import SimpleNamespace
    from mjlab_microduck.tasks import mdp
    commands = torch.zeros(6, 3)
    activation = torch.zeros(6, 6)
    activation[1, 2] = 1.  # LEFT stuck; other cases fail one readiness gate.
    env = SimpleNamespace(command_manager=SimpleNamespace(get_command=lambda _: commands))
    monkeypatch.setattr(mdp, "mario_nes_activation", lambda *a, **k: activation)
    monkeypatch.setattr(mdp, "_mario_foot_anchor_gate",
                        lambda *a, **k: torch.tensor([1., 1., 0., 1., 1., 1.]))
    monkeypatch.setattr(mdp, "mario_camera_ready",
                        lambda *a, **k: torch.tensor([1., 1., 1., .2, 1., 1.]))
    monkeypatch.setattr(mdp, "mario_standing_pose_ready",
                        lambda *a, **k: torch.tensor([1., 1., 1., 1., .2, 1.]))
    monkeypatch.setattr(mdp, "mario_command_ready",
                        lambda *a, **k: torch.tensor([1., 1., 1., 1., 1., 0.]))
    params = dict(camera_cfg=object(), standing_pose_cfg=object(),
                  transition_grace_s=.25,
                  enabled_buttons=(False, False, True, True, True, True))
    success = mdp.mario_clean_button_success(env, **params)
    assert success.tolist() == [1., 0., 0., 0., 0., 0.]
    combined = torch.ones(6, dtype=torch.bool)
    for name in ("requested_pressed", "wrong_buttons_released", "support",
                 "camera", "leg_pose", "foot_pose", "command_ready"):
        combined &= mdp.mario_clean_button_success(env, component=name, **params).bool()
    assert torch.equal(combined, success.bool())
    released = mdp.mario_clean_button_success(env, component="left_released_if_unrequested", **params)
    assert released.tolist() == [1., 0., 1., 1., 1., 1.]
    assert mdp.mario_clean_button_success(env, component="jump_released_if_unrequested", **params).all()
