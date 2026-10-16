"""Matched control contracts; fixtures are not evidence of Mario completion."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integrations/super_mario"))
from mario_action_macros import action_frames, append_executed_frames
from mario_ppo import PPOAgent, PPOConfig, VisionFusionGRUController, HOLD_DURATION_SCALE


def test_direct_buttons_feedback_reaches_actor_and_critic_and_resets(tmp_path):
    torch.set_num_threads(1)
    config = PPOConfig(feature_dim=3, stack_depth=1, controller_feature_dim=8,
        visual_encoder="flyvis_spatial_v1", temporal_encoder="vision_fusion_gru_v2",
        stabilized_vision_gru=True, direct_four_frame=True)
    agent = PPOAgent(config)
    stack = VisionFusionGRUController(3)
    stack.reset(np.zeros(3, np.float32))
    # Five consecutive decisions -> a continuous 20-frame press, not five jumps.
    groups = []
    for _ in range(5):
        frames = action_frames(9, bounded=False, repeat=4)
        groups.append(append_executed_frames(stack, [np.zeros(3)] * 4, frames))
    tokens = np.concatenate(groups)
    assert tokens[:, 3 + 2].tolist() == [1.] * 20
    assert tokens[:, 3 + 4].sum() == 1
    assert tokens[:, 3 + 5].sum() == 0
    np.testing.assert_allclose(tokens[:, -2], np.arange(1, 21) / HOLD_DURATION_SCALE)
    np.testing.assert_allclose(tokens[-1, 3:], [0, 1, 1, 1, 0, 0, 20 / 32, 20 / 32])
    seen = []
    hook = agent.network.controller_projection.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach().clone()))
    tensor = torch.tensor(tokens[None, -4:], requires_grad=True)
    logits, values, _ = agent.network.recurrent_tokens(tensor)
    hook.remove()
    np.testing.assert_allclose(seen[-1][0].numpy(), tokens[-1, 3:])
    actor_grad = torch.autograd.grad(logits.square().sum(), tensor, retain_graph=True)[0]
    critic_grad = torch.autograd.grad(values.sum(), tensor)[0]
    assert actor_grad[0, -1, 3:].abs().sum() > 0
    assert critic_grad[0, -1, 3:].abs().sum() > 0
    release = append_executed_frames(stack, [np.zeros(3)] * 4, [7] * 4)
    assert release[:, 3 + 5].sum() == 1
    assert release[:, 3 + 2].sum() == 0
    assert not stack.reset(np.ones(3))[0, 3:].any()
    agent.save(tmp_path / "new.pt")
    assert PPOAgent.load(tmp_path / "new.pt").config.direct_four_frame
    payload = torch.load(tmp_path / "new.pt", weights_only=False)
    payload["schema"] = 9
    del payload["config"]["direct_four_frame"]
    torch.save(payload, tmp_path / "old.pt")
    assert not PPOAgent.load(tmp_path / "old.pt").config.direct_four_frame
    with pytest.raises(ValueError):
        PPOConfig(direct_four_frame=True)


def test_direct_four_frame_keeps_reward_and_visual_contract():
    from reward_contract import REWARD_CONTRACT
    from rollouts import REWARD_COMPONENTS, training_reward
    # Same shaped reward both arms use; direct mode must not fork it.
    components = {"progress": 1.0, "score": 0.2, "completion": 0.0, "death": 0.0}
    assert training_reward(components, raw_reward=1.0) == pytest.approx((1.0 + 0.2 * 2.5) / 10.0)
    assert REWARD_CONTRACT == "vietnh1009-shaped-action-interval-v1"
    assert tuple(REWARD_COMPONENTS) == (
        "progress", "time", "score", "coins", "powerup", "completion", "death")
    direct = PPOConfig(feature_dim=3, stack_depth=1, controller_feature_dim=8,
        visual_encoder="flyvis_spatial_v1", temporal_encoder="vision_fusion_gru_v2",
        stabilized_vision_gru=True, direct_four_frame=True)
    bounded = PPOConfig(feature_dim=3, stack_depth=1, controller_feature_dim=8,
        visual_encoder="flyvis_spatial_v1", temporal_encoder="vision_fusion_gru_v2",
        stabilized_vision_gru=True, bounded_jump_actions=True)
    # Visual stack is identical; only the button timing/feedback flag differs.
    assert direct.visual_encoder == bounded.visual_encoder == "flyvis_spatial_v1"
    assert direct.temporal_encoder == bounded.temporal_encoder == "vision_fusion_gru_v2"
    assert (direct.stabilized_vision_gru, direct.spatial_readout) == (
        bounded.stabilized_vision_gru, bounded.spatial_readout)
    # Direct decisions are exactly four emulator frames with raw holds.
    assert action_frames(9, bounded=False, repeat=4) == (9,) * 4
    assert len(action_frames(9, bounded=True, repeat=4)) == 16


@pytest.mark.parametrize("arm", ["male", "cnn"])
def test_launcher_submits_one_smoke_job_and_requires_gate_for_training(tmp_path, arm):
    (tmp_path / "brain.npz").touch()
    (tmp_path / "weights.npz").touch()
    env = {**os.environ, "SCRATCH": str(tmp_path), "FLY_DATA": str(tmp_path),
           "MARIO_PIPELINE_DRY_RUN": "1"}
    env.pop("SLURM_JOB_ID", None)
    cmd = ["bash", str(ROOT / "slurm_mario_comparison.sh"), arm]
    result = subprocess.run(cmd + ["smoke"], env=env, text=True, capture_output=True, check=True)
    assert len(result.stdout.splitlines()) == 1
    assert result.stdout.startswith("sbatch ")
    assert "--dependency" not in result.stdout
    assert subprocess.run(cmd + ["train"], env=env, capture_output=True).returncode != 0


def fake_pixels_env():
    gym = pytest.importorskip("gymnasium")
    class Env(gym.Env):
        action_space = gym.spaces.Discrete(10)
        observation_space = gym.spaces.Box(0, 255, (84, 84, 3), np.uint8)
        def __init__(self):
            self.actions = []
        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.frame = 0
            return np.zeros((84, 84, 3), np.uint8), {"x_pos": 40}
        def step(self, action):
            self.actions.append(action)
            self.frame += 1
            components = {"progress": 1., "score": .2}
            done = self.frame == 23
            if done:
                components["death"] = -25.
            return np.full((84, 84, 3), self.frame, np.uint8), float(self.frame % 3), done, False, {
                "x_pos": 40 + self.frame, "reward_components": components}
    return Env()


def test_cnn_wrapper_buttons_reward_stack_and_terminal_match():
    pytest.importorskip("stable_baselines3")
    from mario_cnn_baseline import MarioPixels
    from rollouts import training_reward
    base = fake_pixels_env()
    env = MarioPixels(base)
    obs, _ = env.reset(seed=123)
    assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8
    assert not obs.any()
    for i in range(5):
        obs, reward, done, truncated, info = env.step(9)
        assert reward == training_reward(info["reward_components"], raw_reward=info["raw_reward"])
        assert info["jump_press_edges"] == (1 if i == 0 else 0)
        assert not done and not truncated
    assert base.actions == [9] * 20
    assert info["max_jump_hold_frames"] == 20
    assert obs[:, 0, 0].tolist() == [8, 12, 16, 20]
    _, _, done, _, info = env.step(7)
    assert done and info["emulator_frames"] == 3 and info["death"]
    assert info["jump_release_edges"] == 1
    env.reset()
    _, _, _, _, info = env.step(9)
    assert info["max_jump_hold_frames"] == 4
    env = MarioPixels(fake_pixels_env(), max_decisions=1)
    env.reset()
    _, _, done, truncated, info = env.step(9)
    assert not done and truncated and info["evaluator_truncated"]


def test_actual_sb3_cnn_train_save_load_rank_evaluate(monkeypatch, tmp_path):
    pytest.importorskip("stable_baselines3")
    import mario_cnn_baseline as baseline
    torch.set_num_threads(1)
    monkeypatch.setattr(baseline, "make_env", lambda env_id, limit: baseline.MarioPixels(fake_pixels_env(), limit))
    real_version = baseline.version
    monkeypatch.setattr(baseline, "version", lambda name: "fixture" if name == "gym-super-mario-bros" else real_version(name))
    args = SimpleNamespace(run_dir=tmp_path, env="fixture", max_decisions=16,
        n_envs=8, seed=123, resume=False, device="cpu", save_every=1024, steps=1024)
    baseline.train(args)
    assert json.loads((tmp_path / "cnn-training.json").read_text())["actual_decisions"] == 1024
    reports = baseline.evaluate(tmp_path / "cnn-ppo.zip", tmp_path / "eval", seed=123,
        episodes=1, record_episodes=1, device="cpu")
    assert [r["mode"] for r in reports] == ["mean", "sampled"]
    assert reports[0]["summary"]["completion_rate"] == 0
    assert (tmp_path / "eval/mean-episode-000/rollout.gif").is_file()
    assert (tmp_path / "eval/mean-episode-000/frames.jsonl").is_file()
    baseline.rank(SimpleNamespace(run_dir=tmp_path, output=tmp_path / "ranking",
        max_checkpoints=6, episodes=1, seed=42, device="cpu"))
    assert (tmp_path / "best-cnn-ppo.zip").is_file()
    args.steps, args.resume = 2048, True
    baseline.train(args)
    assert json.loads((tmp_path / "cnn-training.json").read_text())["actual_decisions"] == 2048
