"""CPU checks for the bounded NES action contract and PPO replay guard."""
import importlib.util
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
from mario_dqn import ACTION_LEVELS, FlybrainAction
from mario_ppo import PPOAgent, PPOConfig, VisionFusionGRUController


@pytest.mark.parametrize("action", range(10))
def test_macros_have_equal_duration_and_release_jump(action):
    frames = action_frames(action, bounded=True, repeat=4)
    assert len(frames) == 16
    assert action_frames(action, bounded=False, repeat=4) == (action,) * 4
    levels = ACTION_LEVELS[action]
    for a in frames:
        assert ACTION_LEVELS[a][:2] == levels[:2]
        assert ACTION_LEVELS[a][3] == levels[3]
    assert not ACTION_LEVELS[frames[0]][2]
    assert not ACTION_LEVELS[frames[-1]][2]
    expected = (12 if levels[3] else 4) if levels[2] else 0
    assert sum(ACTION_LEVELS[a][2] for a in frames) == expected


def test_repeated_jump_macro_releases_and_feedback_tracks_actual_buttons():
    stack = VisionFusionGRUController(3)
    stack.reset(np.zeros(3, np.float32))
    for _ in range(100):
        frames = action_frames(FlybrainAction.RIGHT_RUN_JUMP, bounded=True, repeat=4)
        tokens = append_executed_frames(stack, [np.zeros(3, np.float32)] * 16, frames)
        assert tokens[:, 3 + 2].tolist() == [0.] + [1.] * 12 + [0.] * 3
        assert tokens[:, 3 + 4].sum() == 1  # physical press
        assert tokens[:, 3 + 5].sum() == 1  # physical release
        assert stack.last_diagnostics["jump_hold_decisions"] == 12
        assert stack.jump_hold_decisions == 0  # live state, not interval maximum
    stack.reset(np.zeros(3, np.float32))
    tokens = append_executed_frames(stack, [np.zeros(3, np.float32)] * 3, frames[:3])
    assert len(tokens) == 3  # environment may terminate before macro ends
    assert stack.jump_hold_decisions == 2


def test_replay_checks_episode_boundaries_and_fails_before_update(tmp_path):
    torch.set_num_threads(1)
    config = PPOConfig(feature_dim=3, stack_depth=1, controller_feature_dim=8,
        visual_encoder="flyvis_spatial_v1", temporal_encoder="vision_fusion_gru_v2",
        stabilized_vision_gru=True, bounded_jump_actions=True,
        recurrent_sequence_length=3, update_epochs=1, minibatch_size=6)
    agent = PPOAgent(config, seed=41)
    stack = VisionFusionGRUController(3)
    tokens, hiddens, actions, logps, dones = [], [], [], [], []
    rng = np.random.default_rng(5)
    for i in range(6):
        if i in (0, 2):
            agent.reset_recurrent()
            token = stack.reset(rng.normal(size=3).astype(np.float32))
        hidden = agent.observe_recurrent(token)
        action, logp, _ = agent.act_recurrent()
        tokens.append(token)
        hiddens.append(hidden)
        actions.append(action)
        logps.append(logp)
        dones.append(float(i in (1, 5)))
        frames = action_frames(action, bounded=True, repeat=4)
        token = append_executed_frames(stack,
            [rng.normal(size=3).astype(np.float32) for _ in frames], frames)
    hiddens = np.asarray(hiddens)
    assert agent.check_recurrent_replay(tokens, hiddens, actions, logps, dones) < 1e-5
    before = {k: v.clone() for k, v in agent.network.state_dict().items()}
    args = dict(token_sequences=tokens, initial_hiddens=hiddens,
        actions=np.array(actions), old_log_probabilities=np.array(logps),
        returns=np.arange(6, dtype=np.float32), advantages=np.arange(6, dtype=np.float32),
        dones=np.array(dones), visual_targets=[np.zeros((len(t),528),np.float32) for t in tokens])
    args["old_log_probabilities"][3] += .1
    with pytest.raises(RuntimeError, match="replay mismatch"):
        agent.update_recurrent(**args)
    assert all(torch.equal(before[k], v) for k, v in agent.network.state_dict().items())
    args["old_log_probabilities"] = np.array(logps)
    metrics = agent.update_recurrent(**args)
    assert metrics["replay_max_log_probability_error"] < 1e-5
    agent.save(tmp_path / "bounded.pt")
    assert PPOAgent.load(tmp_path / "bounded.pt").config.bounded_jump_actions
    # Old checkpoint metadata defaults to raw button holds, never auto-migrates.
    payload = torch.load(tmp_path / "bounded.pt", weights_only=False)
    del payload["config"]["bounded_jump_actions"]
    payload["schema"] = 8
    torch.save(payload, tmp_path / "old.pt")
    assert not PPOAgent.load(tmp_path / "old.pt").config.bounded_jump_actions


def test_bounded_workflow_has_progress_checks_and_no_remote_submission(tmp_path):
    (tmp_path / "brain.npz").touch()
    (tmp_path / "weights.npz").touch()
    result = subprocess.run(["bash", str(ROOT / "scripts/train_mario_bounded_jumps.sh"), "25000"],
        env={**os.environ, "SCRATCH": str(tmp_path), "FLY_DATA": str(tmp_path),
             "MARIO_PIPELINE_DRY_RUN": "1"}, text=True, capture_output=True, check=True)
    lines = [l for l in result.stdout.splitlines() if l.startswith("sbatch")]
    assert [l.split()[-1] for l in lines] == [
        "smoke", "progress", "train", "progress", "train", "progress", "train", "progress", "evaluate"]
    assert "MARIO_READOUT_STAGE_TARGET=25000" in lines[-2]
    assert all("--dependency=afterok:12345" in l for l in lines[1:])


def test_progress_selection_rewards_success_not_waiting():
    spec = importlib.util.spec_from_file_location("progress", ROOT / "scripts/select_mario_progress.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = dict(mode="mean", learning_enabled=False, visual_ablation="normal",
        episodes=[{}], bounded_jump_actions=True, summary=dict(completion_rate=0.,
        mean_max_x=434., mean_raw_reward=-20., mean_max_no_progress_frames=7000))
    stalled = module.selection_key(report)
    report["summary"]["mean_max_x"] = 800.
    assert module.selection_key(report) > stalled
    report["summary"]["completion_rate"] = .2
    assert module.selection_key(report)[0] == .2


def fake_nes(monkeypatch):
    """An API fixture, not a Mario emulator or evidence of gameplay success."""
    class Env:
        def __init__(self):
            self.actions = []
        def reset(self, seed=None):
            self.frame = 0
            return np.zeros((8, 8, 3), np.uint8), {"x_pos": 40}
        def step(self, action):
            self.actions.append(action)
            self.frame += 1
            return np.zeros((8, 8, 3), np.uint8), 1., self.frame >= 19, False, {
                "x_pos": 40 + self.frame, "reward_components": {"progress": 1.}}
        def close(self):
            pass
    class Brain:
        feature_dim = 3
        visual_encoder = "flyvis_spatial_v1"
        dopamine = None
        def __init__(self, **kwargs):
            pass
        def reset(self, observation, seed=None):
            return np.zeros(3, np.float32)
        def observe(self, observation, **kwargs):
            return np.ones(3, np.float32)
        def visual_stats(self):
            return {}
        def retina_stats(self):
            return {}
        def flyvis_spatial_activity(self):
            return np.zeros((11, 6, 8), np.float32)
        def close(self):
            pass
        def save_plasticity(self):
            pass
        def reinforce(self, prediction_error):
            pass
        def dopamine_stats(self):
            return None
    env = Env()
    monkeypatch.setitem(sys.modules, "gym_super_mario_bros", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "gymnasium", SimpleNamespace(make=lambda *a, **k: env))
    monkeypatch.setitem(sys.modules, "nes_py.wrappers", SimpleNamespace(JoypadSpace=lambda e, a: e))
    return env, Brain


@pytest.mark.parametrize("direct", [False, True])
def test_evaluation_sends_macro_buttons_and_counts_physical_edges(monkeypatch, tmp_path, direct):
    import evaluate_flybrain as evaluator
    env, brain = fake_nes(monkeypatch)
    monkeypatch.setattr(evaluator, "MaleCNS", brain)
    config = PPOConfig(feature_dim=3, stack_depth=1, controller_feature_dim=8,
        visual_encoder="flyvis_spatial_v1", temporal_encoder="vision_fusion_gru_v2",
        stabilized_vision_gru=True, bounded_jump_actions=not direct, direct_four_frame=direct)
    agent = PPOAgent(config)
    with torch.no_grad():
        agent.network.policy.weight.zero_()
        agent.network.policy.bias.zero_()
        agent.network.policy.bias[FlybrainAction.RIGHT_RUN_JUMP] = 10.
    checkpoint = tmp_path / "eval.pt"
    agent.save(checkpoint)
    args = SimpleNamespace(checkpoint=checkpoint, device="cpu", male_cns_data=None,
        male_cns_device="cpu", seed=123, dopamine_state=None, dopamine_learning_rate=0.,
        env="fixture", episodes=1, max_decisions_per_episode=10, action_repeat=4,
        sample_actions=False, epsilon=0., first_actions=10, output=None)
    report = evaluator.run(args)
    expected = action_frames(9, bounded=not direct, repeat=4)
    assert env.actions == (list(expected + expected[:3]) if not direct else [9] * 19)
    assert report["hold_counter_units"] == "frames"
    assert report["summary"]["mean_jump_press_edges"] == (1 if direct else 2)
    assert report["summary"]["mean_jump_release_edges"] == (0 if direct else 1)
    assert report["summary"]["mean_max_jump_hold_decisions"] == (19 if direct else 12)
    assert report["direct_four_frame"] == direct
    assert report["summary"]["mean_max_no_progress_frames"] == 0


@pytest.mark.parametrize("direct", [False, True])
def test_training_api_smoke_collects_and_replays_bounded_rollouts(monkeypatch, tmp_path, direct):
    import train_ppo_flybrain as trainer
    env, brain = fake_nes(monkeypatch)
    monkeypatch.setattr(trainer, "MaleCNS", brain)
    monkeypatch.setattr(sys, "argv", ["train", "--steps", "6", "--rollout-steps", "2",
        "--vision-fusion-gru", "--stabilized-vision-gru", "--direct-four-frame" if direct else "--bounded-jump-actions",
        "--visual-encoder", "flyvis_spatial_v1", "--device", "cpu",
        "--flyvis-model", "fixture", "--flyvis-python", "fixture", "--flyvis-worker", "fixture",
        "--tensorboard-dir", str(tmp_path / "tb"), "--output", str(tmp_path / "trained.pt")])
    trainer.main()
    trained = PPOAgent.load(tmp_path / "trained.pt")
    assert trained.steps == 6
    # Bounded macros terminate the 19-frame fixture every two decisions (three
    # episodes -> three flushes). Direct four-frame holds need five decisions
    # for the first 19-frame episode, so the same six decisions flush twice.
    assert trained.updates == (2 if direct else 3)
    assert trained.config.bounded_jump_actions == (not direct)
    assert trained.config.direct_four_frame == direct
    assert len(env.actions) == (23 if direct else 3 * 19)
