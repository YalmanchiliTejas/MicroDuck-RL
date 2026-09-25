from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_mario_env_cfg import make_microduck_mario_env_cfg


@pytest.mark.parametrize("leg,slot", [("left", 3), ("right", 4)])
def test_one_leg_gets_discovery_credit_but_not_combo_credit(monkeypatch, leg, slot):
    command = torch.tensor([[1., 0., 1.], [0., 0., 0.]])
    env = SimpleNamespace(command_manager=SimpleNamespace(get_command=lambda _: command))
    activation = torch.zeros(2, 6)
    activation[:, slot] = 1
    monkeypatch.setattr(mdp, "mario_nes_activation", lambda *a, **k: activation)
    monkeypatch.setattr(mdp, "mario_nes_progress", lambda *a, **k: activation)
    other = "right" if leg == "left" else "left"
    for func in (mdp.mario_requested_button_reward, mdp.mario_requested_button_progress_reward):
        assert func(env, leg=leg, require_exclusive=True).tolist() == [1., 0.]
        assert func(env, leg=other, require_exclusive=True).tolist() == [0., 0.]
        assert func(env, require_exclusive=True).tolist() == [0., 0.]
    assert mdp.mario_active_success_rate(env, leg=leg).tolist() == [1., 1.]
    assert mdp.mario_active_success_rate(env, leg=leg, combinations_only=True).tolist() == [1., 1.]
    assert mdp.mario_active_success_rate(env, leg=other).tolist() == [0., 0.]
    assert mdp.mario_active_success_rate(env, combinations_only=True).tolist() == [0., 0.]
    # Complete combo is allowed by both leg terms; the other requested key
    # must never be mistaken for an unrequested key after selecting one leg.
    activation[0, 3:5] = 1
    assert mdp.mario_requested_button_reward(env, leg=leg, require_exclusive=True)[0] == 1
    assert mdp.mario_requested_button_reward(env, require_exclusive=True)[0] == 1
    # Truly wrong LEFT blocks both leg rewards under a RIGHT+A request.
    activation[0, 2] = 1
    assert mdp.mario_requested_button_reward(env, leg=leg, require_exclusive=True)[0] == 0


def test_per_leg_approach_history_is_independent_and_not_rechargeable():
    cfg = make_microduck_mario_env_cfg()
    command = torch.tensor([[1., 0., 1.], [1., 0., 1.]])
    pos = torch.zeros(2, 2, 3)
    term = SimpleNamespace(command_age=torch.ones(2))
    q = torch.tensor([[[1., 0., 0., 0.]] * 2] * 2)
    env = SimpleNamespace(
        step_dt=.02, episode_length_buf=torch.full((2,), 10),
        command_manager=SimpleNamespace(get_command=lambda _: command, get_term=lambda _: term),
        scene={
            "robot": SimpleNamespace(data=SimpleNamespace(site_pos_w=pos)),
            "nes_controller": SimpleNamespace(data=SimpleNamespace(
                body_link_pos_w=torch.zeros_like(pos), body_link_quat_w=q)),
        },
    )
    params = {}
    for leg in ("left", "right"):
        params[leg] = cfg.rewards[f"{leg}_foot_approach"].params
        params[leg]["robot_cfg"].site_ids = [0, 1]
        params[leg]["controller_cfg"].body_ids = [0, 1]
    def step(order=("left", "right")):
        term.command_age += .02
        return {leg: mdp.mario_foot_approach_reward(env, **params[leg]) for leg in order}
    assert all(not value.any() for value in step().values())
    pos[:, 1, 0] += .002
    result = step()
    assert not result["left"].any()
    assert (result["right"] > 0).all()
    pos[:, 0, 1] -= .002
    result = step(("right", "left"))
    assert (result["left"] > 0).all()
    assert not result["right"].any()
    assert all(not value.any() for value in step().values())
    pos[:, 0, 1] += .002
    step()
    pos[:, 0, 1] -= .002
    assert all(not value.any() for value in step().values())
    # Reset only one world: the other world's improvement survives.
    env.episode_length_buf[0] = 0
    pos[:, 0, 1] -= .002
    result = step()
    assert result["left"][0] == 0
    assert result["left"][1] > 0


def test_split_reward_budget_and_no_duplicate_aggregate_shaping():
    rewards = make_microduck_mario_env_cfg().rewards
    assert "requested_button_progress" not in rewards
    assert "foot_approach" not in rewards
    single = rewards["left_requested_button"].weight + rewards["left_button_progress"].weight
    assert single == pytest.approx(1.25)
    complete = rewards["requested_button"].weight + 2 * single
    assert complete == pytest.approx(8.5)
    for leg in ("left", "right"):
        for name in (f"{leg}_requested_button", f"{leg}_button_progress", f"{leg}_foot_approach"):
            assert rewards[name].params["leg"] == leg
