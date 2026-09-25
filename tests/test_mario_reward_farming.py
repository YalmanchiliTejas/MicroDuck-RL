from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_mario_env_cfg import make_microduck_mario_env_cfg


def fixture():
    cfg = make_microduck_mario_env_cfg()
    command = torch.tensor([[1., 0., 1.]])
    positions = torch.zeros(1, 2, 3)
    q = torch.tensor([[[1., 0., 0., 0.]] * 2])
    robot = SimpleNamespace(data=SimpleNamespace(site_pos_w=positions, site_quat_w=q))
    controller = SimpleNamespace(data=SimpleNamespace(
        body_link_pos_w=torch.zeros_like(positions), body_link_quat_w=q))
    term = SimpleNamespace(command_age=torch.tensor([1.]))
    env = SimpleNamespace(
        scene={"robot": robot, "nes_controller": controller}, step_dt=0.02,
        episode_length_buf=torch.tensor([10]),
        command_manager=SimpleNamespace(get_command=lambda _: command, get_term=lambda _: term),
    )
    params = dict(cfg.rewards["commanded_foot_pose"].params)
    params["robot_cfg"].site_ids = [0, 1]
    params["controller_cfg"].body_ids = [0, 1]
    params["left_nominal_roll"] = params["right_nominal_roll"] = 0.
    return env, command, positions, term, params


def test_neutral_and_half_combo_cannot_farm_foot_pose():
    env, command, pos, term, params = fixture()
    assert mdp.mario_commanded_foot_pose_reward(env, **params).item() == 0
    pos[0, 0, 1] = -params["lateral_offset"]
    assert mdp.mario_commanded_foot_pose_reward(env, **params).item() == 0
    pos[0, 1, 0] = params["position_offset"]
    assert mdp.mario_commanded_foot_pose_reward(env, **params).item() == pytest.approx(1)
    command.zero_()
    pos.zero_()
    assert mdp.mario_commanded_foot_pose_reward(env, **params).item() == 0


def test_approach_credit_cannot_be_farmed_by_holding_or_oscillation():
    env, command, pos, term, params = fixture()
    params = {k: v for k, v in params.items() if k not in (
        "target_tilt", "position_std", "angle_std", "left_nominal_roll", "right_nominal_roll")}
    def step():
        term.command_age += env.step_dt
        return mdp.mario_foot_approach_reward(env, **params).item()
    assert step() == 0
    assert step() == 0
    pos[0, 0, 1] -= .002
    assert step() > 0
    assert step() == 0
    pos[0, 0, 1] += .002
    assert step() == 0
    pos[0, 0, 1] -= .002
    assert step() == 0
    term.command_age.zero_()
    pos[0, 0, 1] -= .002
    assert step() == 0
    env.episode_length_buf.zero_()
    pos[0, 0, 1] -= .002
    assert step() == 0


def test_combo_reward_requires_both_switches_and_no_wrong_key(monkeypatch):
    command = torch.tensor([[1., 0., 1.]] * 4)
    activation = torch.tensor([
        [0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 1., 1., 0.],
        [0., 0., 1., 1., 1., 0.],
    ])
    env = SimpleNamespace(command_manager=SimpleNamespace(get_command=lambda _: command))
    monkeypatch.setattr(mdp, "mario_nes_activation", lambda *a, **k: activation)
    score = mdp.mario_requested_button_reward(env, require_exclusive=True)
    assert score.tolist() == [0., 0., 1., 0.]
    monkeypatch.setattr(mdp, "mario_nes_progress", lambda *a, **k: activation)
    progress = mdp.mario_requested_button_progress_reward(env, require_exclusive=True)
    assert progress.tolist() == [0., 0., 1., 0.]


def test_unloaded_switch_sag_has_zero_progress(monkeypatch):
    monkeypatch.setattr(mdp, "mario_nes_joint_state", lambda *a: torch.full((2, 6), .000082))
    assert not mdp.mario_nes_progress(SimpleNamespace()).any()


def test_active_success_excludes_neutral_and_requires_complete_combos(monkeypatch):
    commands = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 0., 1.]])
    env = SimpleNamespace(command_manager=SimpleNamespace(get_command=lambda _: commands))
    monkeypatch.setattr(mdp, "mario_clean_button_success",
                        lambda *a, **k: torch.tensor([1., 1., 0.]))
    assert mdp.mario_active_success_rate(env).tolist() == [.5, .5, .5]
    assert mdp.mario_active_success_rate(env, combinations_only=True).tolist() == [0., 0., 0.]


def test_foot_targets_match_actual_xml_key_centers():
    import mujoco
    from pathlib import Path

    model = mujoco.MjModel.from_xml_path(str(
        Path(__file__).parents[1] / "src/mjlab_microduck/robot/microduck/controller_nes.xml"))
    cfg = make_microduck_mario_env_cfg()
    p = cfg.rewards["commanded_foot_pose"].params
    assert model.body("dpad_left_key").pos[1] == pytest.approx(p["lateral_offset"])
    assert model.body("dpad_right_key").pos[1] == pytest.approx(-p["lateral_offset"])
    assert model.body("button_a_key").pos[0] == pytest.approx(p["position_offset"])
    assert model.body("button_b_key").pos[0] == pytest.approx(-p["position_offset"])
    assert cfg.rewards["commanded_foot_pose"].weight == 0
    assert cfg.rewards["commanded_foot_clearance"].weight == 0
    assert cfg.rewards["pose"].weight == 0
