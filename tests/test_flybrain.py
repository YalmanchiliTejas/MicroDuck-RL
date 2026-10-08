import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_flybrain():
    path = Path(__file__).parents[1] / "integrations/super_mario/mario_dqn.py"
    spec = importlib.util.spec_from_file_location("microduck_test_flybrain", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def _load_rollouts():
    path = Path(__file__).parents[1] / "integrations/super_mario/rollouts.py"
    spec = importlib.util.spec_from_file_location("microduck_test_rollouts", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def _load_rollout_trainer(flybrain, rollouts):
    sys.modules["mario_dqn"] = flybrain
    sys.modules["rollouts"] = rollouts
    path = Path(__file__).parents[1] / "integrations/super_mario/train_rollouts.py"
    spec = importlib.util.spec_from_file_location("microduck_test_rollout_trainer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_evaluator(flybrain, rollouts):
    injected = {
        "mario_dqn": flybrain,
        "rollouts": rollouts,
        "male_cns": types.SimpleNamespace(MaleCNS=object),
        "mario_sidecar": types.SimpleNamespace(nes_actions=list),
    }
    previous = {name: sys.modules.get(name) for name in injected}
    sys.modules.update(injected)
    path = Path(__file__).parents[1] / "integrations/super_mario/evaluate_flybrain.py"
    spec = importlib.util.spec_from_file_location("microduck_test_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
    return module


def _load_visualizer():
    path = Path(__file__).parents[1] / "integrations/super_mario/visualize_flybrain.py"
    spec = importlib.util.spec_from_file_location("microduck_test_flybrain_visualizer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def _load_male_cns():
    path = Path(__file__).parents[1] / "integrations/super_mario/male_cns.py"
    spec = importlib.util.spec_from_file_location("microduck_test_male_cns", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_dopamine():
    path = Path(__file__).parents[1] / "integrations/super_mario/dopamine.py"
    spec = importlib.util.spec_from_file_location("microduck_test_dopamine", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_ppo(flybrain):
    previous = sys.modules.get("mario_dqn")
    sys.modules["mario_dqn"] = flybrain
    path = Path(__file__).parents[1] / "integrations/super_mario/mario_ppo.py"
    spec = importlib.util.spec_from_file_location("microduck_test_mario_ppo", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
        if previous is None:
            sys.modules.pop("mario_dqn", None)
        else:
            sys.modules["mario_dqn"] = previous
    return module


def _load_ppo_rollout_trainer(ppo, rollouts):
    previous_ppo = sys.modules.get("mario_ppo")
    previous_rollouts = sys.modules.get("rollouts")
    sys.modules["mario_ppo"] = ppo
    sys.modules["rollouts"] = rollouts
    path = Path(__file__).parents[1] / "integrations/super_mario/train_ppo_rollouts.py"
    spec = importlib.util.spec_from_file_location("microduck_test_ppo_rollout_trainer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_ppo is None:
            sys.modules.pop("mario_ppo", None)
        else:
            sys.modules["mario_ppo"] = previous_ppo
        if previous_rollouts is None:
            sys.modules.pop("rollouts", None)
        else:
            sys.modules["rollouts"] = previous_rollouts
    return module


def test_flybrain_actions_cover_combinations_without_opposite_directions():
    flybrain = _load_flybrain()
    assert [flybrain.action_levels(i) for i in range(10)] == [
        (False, False, False, False),
        (True, False, False, False),
        (False, True, False, False),
        (False, False, True, False),
        (True, False, True, False),
        (False, True, True, False),
        (True, False, False, True),
        (False, True, False, True),
        (True, False, True, True),
        (False, True, True, True),
    ]
    for invalid in (-1, 10):
        with pytest.raises(ValueError):
            flybrain.action_levels(invalid)


def test_activity_stack_has_four_malecns_trace_frames():
    flybrain = _load_flybrain()
    frame = np.arange(12, dtype=np.float32)
    stack = flybrain.ActivityStack(4)
    state = stack.reset(frame)
    assert state.shape == (4, 12)
    assert state.dtype == np.float32
    assert np.array_equal(state[0], state[-1])


def test_flybrain_config_locks_four_frames_and_ten_intents():
    flybrain = _load_flybrain()
    config = flybrain.FlybrainConfig()
    assert config.stack_depth == 4
    assert config.num_actions == 10
    assert config.replay_start == 500
    assert config.target_update_every == 1_000
    assert config.per_beta_steps == 10_000
    assert config.epsilon_steps == 10_000
    with pytest.raises(ValueError, match="exactly four frames"):
        flybrain.FlybrainConfig(stack_depth=3)
    with pytest.raises(ValueError, match="exactly 10 intents"):
        flybrain.FlybrainConfig(num_actions=6)
    with pytest.raises(ValueError, match="algorithm"):
        flybrain.FlybrainConfig(algorithm="not-an-algorithm")


def test_dqn_and_double_dqn_differ_only_in_target_action_selection():
    flybrain = _load_flybrain()

    class Fixed(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

        def forward(self, states):
            return self.values.expand(len(states), -1)

    state = torch.zeros((1, 4, 6))
    dqn = flybrain.FlybrainAgent(
        flybrain.FlybrainConfig(feature_dim=6, algorithm="dqn")
    )
    double = flybrain.FlybrainAgent(
        flybrain.FlybrainConfig(feature_dim=6, algorithm="double_dqn")
    )
    for agent in (dqn, double):
        agent.online = Fixed([0.0, 10.0] + [0.0] * 8)
        agent.target = Fixed([5.0, 1.0] + [0.0] * 8)

    assert dqn._bootstrap_values(state).item() == 5.0
    assert double._bootstrap_values(state).item() == 1.0


def test_ppo_gae_uses_raw_rewards_and_terminal_boundaries():
    flybrain = _load_flybrain()
    ppo = _load_ppo(flybrain)
    advantages, returns = ppo.generalized_advantage_estimates(
        np.asarray([1.0, 1.0]),
        np.asarray([0.0, 0.0]),
        np.asarray([0.0, 1.0]),
        next_value=100.0,
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert np.allclose(advantages, [2.0, 1.0])
    assert np.allclose(returns, [2.0, 1.0])


def test_ppo_gae_keeps_multiple_episodes_separate_in_one_rollout():
    flybrain = _load_flybrain()
    ppo = _load_ppo(flybrain)
    advantages, returns = ppo.generalized_advantage_estimates(
        np.asarray([1.0, 1.0, 1.0, 1.0]),
        np.zeros(4),
        np.asarray([0.0, 1.0, 0.0, 1.0]),
        next_value=100.0,
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert np.allclose(advantages, [2.0, 1.0, 2.0, 1.0])
    assert np.allclose(returns, [2.0, 1.0, 2.0, 1.0])


def test_ppo_waits_for_a_complete_rollout_across_short_episodes():
    flybrain = _load_flybrain()
    ppo = _load_ppo(flybrain)
    assert not ppo.rollout_ready(4, 256)
    assert not ppo.rollout_ready(255, 256)
    assert ppo.rollout_ready(256, 256)


def test_ppo_checkpoint_restores_policy_and_sampling_rng(tmp_path):
    flybrain = _load_flybrain()
    ppo = _load_ppo(flybrain)
    config = ppo.PPOConfig(feature_dim=6, rollout_steps=4, minibatch_size=2)
    agent = ppo.PPOAgent(config, seed=8)
    state = np.zeros((4, 6), dtype=np.float32)
    agent.act(state)
    checkpoint = tmp_path / "ppo.pt"
    agent.save(checkpoint)
    expected = [agent.act(state)[0] for _ in range(10)]
    restored = ppo.PPOAgent.load(checkpoint)
    assert [restored.act(state)[0] for _ in range(10)] == expected


def test_factorized_ppo_maps_three_heads_to_ten_action_probabilities():
    flybrain = _load_flybrain()
    ppo = _load_ppo(flybrain)
    logits = torch.zeros((1, 7), dtype=torch.float32)
    probabilities = ppo._factorized_probabilities(logits)[0]
    assert probabilities.shape == (10,)
    assert probabilities.sum().item() == pytest.approx(1.0)
    # Neutral actions marginalize over the irrelevant run head.
    assert probabilities[flybrain.FlybrainAction.IDLE].item() == pytest.approx(1 / 6)
    assert probabilities[flybrain.FlybrainAction.JUMP].item() == pytest.approx(1 / 6)
    # Directional combinations share direction, jump, and run probability.
    assert probabilities[flybrain.FlybrainAction.RIGHT].item() == pytest.approx(1 / 12)
    assert probabilities[flybrain.FlybrainAction.RIGHT_JUMP].item() == pytest.approx(1 / 12)


def test_categorical_ppo_checkpoint_migrates_to_factorized_heads(tmp_path):
    flybrain = _load_flybrain()
    ppo = _load_ppo(flybrain)
    state = np.zeros((4, 6), dtype=np.float32)
    source = ppo.PPOAgent(ppo.PPOConfig(feature_dim=6), seed=8)
    source.steps = 123
    checkpoint = tmp_path / "categorical.pt"
    source.save(checkpoint)

    migrated = ppo.PPOAgent.load(checkpoint, policy_mode="factorized")
    assert migrated.config.policy_mode == "factorized"
    assert migrated.steps == 123
    assert migrated.action_probabilities(state).sum() == pytest.approx(1.0)
    assert torch.equal(
        migrated.network.frame_projection[0].weight,
        source.network.frame_projection[0].weight,
    )
    assert torch.equal(migrated.network.value.weight, source.network.value.weight)
    assert migrated.optimizer.state_dict()["state"] == {}


def test_schema_one_categorical_ppo_checkpoint_still_loads(tmp_path):
    flybrain = _load_flybrain()
    ppo = _load_ppo(flybrain)
    checkpoint = tmp_path / "schema-one.pt"
    ppo.PPOAgent(ppo.PPOConfig(feature_dim=6), seed=8).save(checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    payload["schema"] = 1
    payload["config"].pop("policy_mode")
    torch.save(payload, checkpoint)
    assert ppo.PPOAgent.load(checkpoint).config.policy_mode == "categorical"


def test_ppo_continuation_overrides_are_checkpointed(tmp_path):
    flybrain = _load_flybrain()
    ppo = _load_ppo(flybrain)
    agent = ppo.PPOAgent(ppo.PPOConfig(feature_dim=6), seed=8)
    agent.configure_continuation(
        learning_rate=2.5e-5,
        value_coefficient=0.05,
        entropy_coefficient=0.02,
        target_kl=0.01,
    )
    checkpoint = tmp_path / "ppo-safe.pt"
    agent.save(checkpoint)
    restored = ppo.PPOAgent.load(checkpoint)
    assert restored.config.learning_rate == 2.5e-5
    assert restored.config.value_coefficient == 0.05
    assert restored.config.entropy_coefficient == 0.02
    assert restored.config.target_kl == 0.01
    assert restored.optimizer.param_groups[0]["lr"] == 2.5e-5


def test_ppo_update_accepts_unmodified_reward_returns():
    flybrain = _load_flybrain()
    ppo = _load_ppo(flybrain)
    config = ppo.PPOConfig(
        feature_dim=6,
        rollout_steps=4,
        update_epochs=1,
        minibatch_size=2,
    )
    agent = ppo.PPOAgent(config, seed=9)
    states = np.zeros((4, 4, 6), dtype=np.float32)
    samples = [agent.act(state) for state in states]
    metrics = agent.update(
        states=states,
        actions=np.asarray([sample[0] for sample in samples]),
        old_log_probabilities=np.asarray([sample[1] for sample in samples]),
        returns=np.asarray([100.0, -25.0, 50.0, 10.0], dtype=np.float32),
        advantages=np.asarray([100.0, -25.0, 50.0, 10.0], dtype=np.float32),
    )
    assert agent.updates == 1
    assert metrics["batch_size"] == 4.0
    assert all(np.isfinite(value) for value in metrics.values())


def test_factorized_ppo_update_reports_per_head_entropy_and_kl():
    flybrain = _load_flybrain()
    ppo = _load_ppo(flybrain)
    config = ppo.PPOConfig(
        feature_dim=6,
        policy_mode="factorized",
        rollout_steps=4,
        update_epochs=1,
        minibatch_size=2,
    )
    agent = ppo.PPOAgent(config, seed=9)
    states = np.zeros((4, 4, 6), dtype=np.float32)
    samples = [agent.act(state) for state in states]
    metrics = agent.update(
        states=states,
        actions=np.asarray([sample[0] for sample in samples]),
        old_log_probabilities=np.asarray([sample[1] for sample in samples]),
        returns=np.asarray([100.0, -25.0, 50.0, 10.0], dtype=np.float32),
        advantages=np.asarray([100.0, -25.0, 50.0, 10.0], dtype=np.float32),
    )
    assert metrics["approx_kl"] >= 0.0
    for name in ("direction_entropy", "jump_entropy", "run_entropy"):
        assert metrics[name] > 0.0


def test_replay_samples_self_contained_pre_and_post_action_states():
    flybrain = _load_flybrain()
    replay = flybrain.PrioritizedReplay(2, (4, 6), seed=3)

    def stack(*values):
        return np.stack([np.full(6, value, dtype=np.float32) for value in values])

    replay.add(stack(1, 2, 3, 4), 0, 0.0, stack(5, 6, 7, 8), False)
    replay.add(stack(11, 12, 13, 14), 1, 1.0, stack(15, 16, 17, 18), True)
    # Overwrite the oldest circular-buffer entry. The remaining transition must
    # not need that old entry to reconstruct its own current state.
    replay.add(stack(21, 22, 23, 24), 2, 2.0, stack(25, 26, 27, 28), False)
    sample = replay.sample(2, beta=0.4)
    assert sample[0].shape == (2, 4, 6)
    assert sample[3].shape == (2, 4, 6)
    transitions = {
        int(action): (state[:, 0].tolist(), next_state[:, 0].tolist())
        for state, action, next_state in zip(sample[0], sample[1], sample[3], strict=True)
    }
    assert transitions == {
        1: ([11, 12, 13, 14], [15, 16, 17, 18]),
        2: ([21, 22, 23, 24], [25, 26, 27, 28]),
    }


def test_replay_accepts_non_overlapping_activity_stacks():
    flybrain = _load_flybrain()
    replay = flybrain.PrioritizedReplay(2, (4, 6))
    state = np.zeros((4, 6), dtype=np.float32)
    next_state = np.ones((4, 6), dtype=np.float32)
    replay.add(state, 0, 0.0, next_state, False)
    sample = replay.sample(1, beta=0.4)
    assert np.array_equal(sample[0][0], state)
    assert np.array_equal(sample[3][0], next_state)


def test_prioritized_replay_checkpoint_restores_data_position_and_rng(tmp_path):
    flybrain = _load_flybrain()
    replay = flybrain.PrioritizedReplay(3, (4, 6), seed=19)
    for index in range(4):
        state = np.full((4, 6), index, dtype=np.float32)
        next_state = np.full((4, 6), index + 1, dtype=np.float32)
        replay.add(state, index, float(index), next_state, index == 3)
    path = tmp_path / "replay.npz"
    replay.save(path)

    restored = flybrain.PrioritizedReplay(3, (4, 6), seed=999)
    restored.load(path)
    assert len(restored) == 3
    assert restored._position == replay._position
    assert np.array_equal(restored.states, replay.states)
    assert np.array_equal(restored.next_states, replay.next_states)
    assert np.array_equal(restored.actions, replay.actions)
    assert np.array_equal(restored.rewards, replay.rewards)
    assert np.array_equal(restored.dones, replay.dones)
    assert np.array_equal(restored.priorities, replay.priorities)
    # The RNG state is part of the checkpoint, so PER sampling also continues.
    original_sample = replay.sample(2, beta=0.4)
    restored_sample = restored.sample(2, beta=0.4)
    assert np.array_equal(original_sample[-1], restored_sample[-1])


def test_dueling_network_outputs_one_q_value_per_action():
    flybrain = _load_flybrain()
    network = flybrain.DuelingQNetwork(feature_dim=16)
    output = network(torch.zeros(2, 4, 16))
    assert output.shape == (2, 10)


def test_male_cns_backend_uses_real_step_and_only_descending_trace(tmp_path, monkeypatch):
    class FakeBrain:
        def __init__(self, **_kwargs):
            self.n = 8
            self.steps = 0
            self.azimuth = np.asarray((-1.0, 1.0), dtype=np.float32)

        def cells(self, names, side=None):
            if names == ["descending_neuron"]:
                return np.asarray((4, 5), dtype=np.int64)
            return np.asarray((0 if side == "L" else 1,), dtype=np.int64)

        def step(self, **_kwargs):
            self.steps += 1
            return np.asarray((0, 4), dtype=np.int64)

        def reset(self):
            self.steps = 0

    class FakeTrace:
        def __init__(self, _brain, idx, **_kwargs):
            assert idx.tolist() == [4, 5]

        def observe(self, fired):
            assert fired.tolist() == [0, 4]
            return np.asarray((1.0, 0.0), dtype=np.float32)

        def reset(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "flybrain",
        types.SimpleNamespace(FlyBrain=FakeBrain, Trace=FakeTrace),
    )
    male_cns = _load_male_cns()
    spikes = tmp_path / "spikes.jsonl"
    backend = male_cns.MaleCNS(spike_file=spikes)
    frame = np.zeros((12, 16, 3), dtype=np.uint8)[:, ::-1]
    assert frame.strides[1] < 0
    assert backend.observe(frame, action_sequence=7).tolist() == [1.0, 0.0]
    row = json.loads(spikes.read_text())
    assert row["population"] == "MaleCNS descending_neuron"
    assert row["neuron_ids"] == [4]
    assert row["all_spikes"] == 2


def test_reward_packet_preserves_action_sequence_components_and_terminal():
    rollouts = _load_rollouts()
    event = {
        "run_id": "test-run",
        "episode": 2,
        "transition": 7,
        "action_sequence": 41,
        "action": 5,
        "reward": -25.0,
        "training_reward": -1.0,
        "terminated": True,
        "truncated": False,
        "reward_components": {"death": -25.0},
        "emulator_steps": 9,
        "execution_fraction": 0.75,
    }
    assert rollouts.decode_reward_packet(rollouts.encode_reward_packet(event)) == event


def test_training_reward_is_exact_unmodified_gymnasium_reward():
    rollouts = _load_rollouts()
    components = {"progress": 150.0, "death": -25.0}
    assert rollouts.training_reward(components, raw_reward=123.5) == 123.5
    assert rollouts.training_reward({}, raw_reward=-17.0) == -17.0
    with pytest.raises(ValueError, match="raw Gymnasium reward"):
        rollouts.training_reward(components)


def test_frozen_evaluator_exposes_old_fatal_progress_reward_bug():
    flybrain = _load_flybrain()
    rollouts = _load_rollouts()
    evaluator = _load_evaluator(flybrain, rollouts)
    fatal_progress = {"progress": 150.0, "death": -25.0}

    assert evaluator._legacy_training_reward(fatal_progress, 0.0) == 5.0
    assert rollouts.training_reward(fatal_progress, raw_reward=-7.0) == -7.0


def test_dopamine_uses_anatomical_pam_ppl1_gates_and_persists(tmp_path):
    dopamine = _load_dopamine()

    class TinyBrain:
        device = "cpu"
        dt = 0.02
        n = 6
        # KC0, KC1, MBON0, MBON1, PAM0, PPL1-0
        cell_type = np.asarray(("KCg-m", "KCab", "MBON01", "MBON02", "PAM01", "PPL101"))
        # CSC edges: KC0->MBON0, KC1->MBON1, PAM0->MBON0, PPL1->MBON1.
        indptr = np.asarray((0, 1, 2, 2, 2, 3, 4), dtype=np.int64)
        indices = np.asarray((2, 3, 2, 3), dtype=np.int64)
        weights = np.asarray((0.5, 0.6, 0.8, 0.9), dtype=np.float32)

    state = tmp_path / "dopamine.npz"
    brain = TinyBrain()
    plasticity = dopamine.DopaminePlasticity(
        brain, state_path=state, learning_rate=0.1, recovery_rate=0.0
    )
    plasticity.observe(np.asarray((0,), dtype=np.int64))
    positive_signal = plasticity.reinforce(+100.0)
    assert 0.0 < positive_signal < 1.0
    assert plasticity.stats()["normalized_prediction_error"] == pytest.approx(1.0)
    assert brain.weights[0] < 0.5
    assert brain.weights[1] == pytest.approx(0.6)
    assert plasticity.consume_injection()[0][0].tolist() == [4]

    plasticity.reset_episode()
    plasticity.observe(np.asarray((1,), dtype=np.int64))
    negative_signal = plasticity.reinforce(-100.0)
    assert -1.0 < negative_signal < 0.0
    assert brain.weights[1] < 0.6
    assert plasticity.consume_injection()[0][0].tolist() == [5]
    plasticity.save()

    restored_brain = TinyBrain()
    restored = dopamine.DopaminePlasticity(
        restored_brain, state_path=state, learning_rate=0.1, recovery_rate=0.0
    )
    assert restored.updates == 2
    assert restored.rpe_second_moment == pytest.approx(
        plasticity.rpe_second_moment
    )
    assert np.allclose(restored_brain.weights[:2], brain.weights[:2])


def test_rollout_recorder_writes_atomic_replay_ready_episode(tmp_path):
    rollouts = _load_rollouts()
    recorder = rollouts.RolloutRecorder(tmp_path, 4, 6, run_id="run")

    def stack(*values):
        return np.stack([np.full(6, value, dtype=np.float32) for value in values])

    recorder.add(
        state=stack(1, 2, 3, 4),
        action=2,
        reward=3.0,
        next_state=stack(2, 3, 4, 5),
        terminated=False,
        truncated=False,
        reward_components={"progress": 3.0},
        action_sequence=10,
        emulator_steps=4,
    )
    terminal_event = recorder.add(
        state=stack(2, 3, 4, 5),
        action=4,
        reward=-25.0,
        next_state=stack(3, 4, 5, 6),
        terminated=True,
        truncated=False,
        reward_components={"death": -25.0},
        action_sequence=11,
        emulator_steps=3,
    )
    assert terminal_event["action_sequence"] == 11
    paths = list(tmp_path.glob("rollout-*.npz"))
    assert len(paths) == 1
    metadata, arrays = rollouts.load_rollout(paths[0])
    assert metadata["complete"] is True
    assert metadata["reward_contract"] == "gymnasium-raw-action-interval-v1"
    assert arrays["actions"].tolist() == [2, 4]
    assert np.allclose(arrays["training_rewards"], [3.0, -25.0])
    assert arrays["action_sequences"].tolist() == [10, 11]
    assert arrays["next_states"][1, :, 0].tolist() == [3, 4, 5, 6]
    assert not list(tmp_path.glob("*.tmp"))
    transitions = [
        json.loads(line)
        for line in (tmp_path / "transitions.jsonl").read_text().splitlines()
    ]
    assert [row["action_sequence"] for row in transitions] == [10, 11]


def test_visualizer_reads_only_real_spike_telemetry(tmp_path):
    visualizer = _load_visualizer()
    spike_file = tmp_path / "spikes.jsonl"
    spike_file.write_text(
        '{"time_s":0.1,"population":"KC","neuron_ids":[3,8]}\n'
        'not-json\n'
        '{"time_s":0.2,"population":"MBON","neuron_ids":[13]}\n'
    )
    spikes = visualizer._tail_json(spike_file, 10)
    assert [row["population"] for row in spikes] == ["KC", "MBON"]
    assert visualizer._tail_json(tmp_path / "missing.jsonl", 10) == []


def test_visualizer_encodes_a_consistent_shared_frame_as_jpeg():
    visualizer = _load_visualizer()
    width, height, channels = 3, 2, 3
    rgb = np.arange(width * height * channels, dtype=np.uint8).reshape(
        height, width, channels
    )
    buffer = bytearray(visualizer.FRAME_HEADER.size + rgb.nbytes)
    visualizer.FRAME_HEADER.pack_into(
        buffer,
        0,
        visualizer.FRAME_MAGIC,
        width,
        height,
        channels,
        rgb.nbytes,
        2,
    )
    buffer[visualizer.FRAME_HEADER.size :] = rgb.tobytes()
    reader = visualizer.FrameReader("unused")
    reader._shm = type("FakeShm", (), {"buf": buffer, "close": lambda self: None})()
    jpeg = reader.read_jpeg()
    assert jpeg is not None
    assert jpeg.startswith(b"\xff\xd8")
    assert jpeg.endswith(b"\xff\xd9")


def test_completed_rollout_can_be_ingested_into_per_and_trained(tmp_path):
    flybrain = _load_flybrain()
    rollouts = _load_rollouts()
    trainer = _load_rollout_trainer(flybrain, rollouts)
    recorder = rollouts.RolloutRecorder(tmp_path, 4, 6, run_id="training")

    def stack(*values):
        return np.stack([np.full(6, value, dtype=np.float32) for value in values])

    recorder.add(
        state=stack(1, 2, 3, 4), action=2, reward=2.0,
        next_state=stack(2, 3, 4, 5), terminated=False, truncated=False,
        reward_components={"progress": 2.0}, action_sequence=0, emulator_steps=4,
    )
    recorder.add(
        state=stack(2, 3, 4, 5), action=3, reward=-25.0,
        next_state=stack(3, 4, 5, 6), terminated=True, truncated=False,
        reward_components={"death": -25.0}, action_sequence=1, emulator_steps=2,
    )
    config = flybrain.FlybrainConfig(
        feature_dim=6,
        stack_depth=4,
        batch_size=1,
        replay_capacity=3,
        replay_start=1,
        train_every=1,
    )
    agent = flybrain.FlybrainAgent(config, seed=4)
    replay = flybrain.PrioritizedReplay(3, (4, 6), seed=4)
    count, reward, loss = trainer.ingest_rollout(
        next(tmp_path.glob("rollout-*.npz")), replay, agent
    )
    assert count == 2
    assert reward == -23.0
    assert loss is not None
    assert len(replay) == 2


def test_completed_physical_rollout_updates_ppo_only_for_executed_segments(tmp_path):
    flybrain = _load_flybrain()
    ppo = _load_ppo(flybrain)
    rollouts = _load_rollouts()
    trainer = _load_ppo_rollout_trainer(ppo, rollouts)
    agent = ppo.PPOAgent(
        ppo.PPOConfig(feature_dim=6, minibatch_size=2, update_epochs=1), seed=5
    )
    recorder = rollouts.RolloutRecorder(tmp_path, 4, 6, run_id="ppo")
    state = np.zeros((4, 6), dtype=np.float32)
    for index, execution_fraction in enumerate((1.0, 0.1, 1.0)):
        action, log_probability, value = agent.act(state)
        next_state = np.full((4, 6), index + 1, dtype=np.float32)
        recorder.add(
            state=state,
            action=action,
            reward=float(index + 1),
            next_state=next_state,
            terminated=index == 2,
            truncated=False,
            reward_components={"progress": float(index + 1)},
            action_sequence=index,
            emulator_steps=30,
            execution_fraction=execution_fraction,
            behavior_log_probability=log_probability,
            behavior_value=value,
        )
        state = next_state

    count, reward, metrics = trainer.ingest_rollout(
        next(tmp_path.glob("rollout-*.npz")), agent
    )
    assert count == 2
    assert reward == 6.0
    assert metrics is not None
    assert agent.steps == 2


def test_checkpoint_continues_random_exploration_sequence(tmp_path):
    flybrain = _load_flybrain()
    config = flybrain.FlybrainConfig(
        feature_dim=6,
        batch_size=1,
        replay_capacity=3,
        replay_start=1,
    )
    state = np.zeros((4, 6), dtype=np.float32)
    live = flybrain.FlybrainAgent(config, seed=17)
    live.act(state, epsilon=1.0)

    checkpoint = tmp_path / "flybrain.pt"
    live.save(checkpoint)
    expected = [live.act(state, epsilon=1.0) for _ in range(20)]
    reloaded = flybrain.FlybrainAgent.load(checkpoint)
    actual = [reloaded.act(state, epsilon=1.0) for _ in range(20)]
    assert actual == expected


def test_checkpoint_rejects_legacy_reward_contract(tmp_path):
    flybrain = _load_flybrain()
    checkpoint = tmp_path / "legacy.pt"
    flybrain.FlybrainAgent(flybrain.FlybrainConfig(feature_dim=6)).save(checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    payload["schema"] = 4
    payload.pop("reward_contract")
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="unmodified Gymnasium reward contract"):
        flybrain.FlybrainAgent.load(checkpoint)
    restored = flybrain.FlybrainAgent.load(
        checkpoint, allow_legacy_reward_contract=True
    )
    assert restored.config.feature_dim == 6


def test_agent_construction_does_not_reset_process_torch_rng():
    flybrain = _load_flybrain()
    config = flybrain.FlybrainConfig(feature_dim=6)
    torch.manual_seed(91)
    expected = torch.rand(5)
    torch.manual_seed(91)
    flybrain.FlybrainAgent(config, seed=0)
    actual = torch.rand(5)
    assert torch.equal(actual, expected)


def test_continuous_exploration_is_not_directionally_skewed():
    flybrain = _load_flybrain()
    config = flybrain.FlybrainConfig(feature_dim=6)
    agent = flybrain.FlybrainAgent(config, seed=0)
    state = np.zeros((4, 6), dtype=np.float32)
    counts = np.bincount(
        [agent.act(state, epsilon=1.0) for _ in range(10_000)],
        minlength=config.num_actions,
    )
    # A continuously advancing seeded stream stays close to 10% per action;
    # the old episode-by-episode RNG reset failed this operational property.
    assert counts.max() - counts.min() < 150


def test_rebuild_replay_restores_processed_data_without_advancing_schedule(tmp_path):
    flybrain = _load_flybrain()
    rollouts = _load_rollouts()
    trainer = _load_rollout_trainer(flybrain, rollouts)
    recorder = rollouts.RolloutRecorder(tmp_path, 4, 6, run_id="resume")
    state = np.zeros((4, 6), dtype=np.float32)
    next_state = np.ones((4, 6), dtype=np.float32)
    recorder.add(
        state=state,
        action=1,
        reward=5.0,
        next_state=next_state,
        terminated=True,
        truncated=False,
        reward_components={"progress": 5.0},
        action_sequence=0,
        emulator_steps=4,
    )
    path = next(tmp_path.glob("rollout-*.npz"))
    config = flybrain.FlybrainConfig(
        feature_dim=6,
        batch_size=1,
        replay_capacity=3,
        replay_start=1,
        train_every=1,
    )
    agent = flybrain.FlybrainAgent(config, seed=2)
    agent.steps = 123
    replay = flybrain.PrioritizedReplay(3, (4, 6), seed=2)

    restored = trainer.rebuild_replay(
        tmp_path, {path.name}, replay, agent
    )
    assert restored == 1
    assert len(replay) == 1
    assert agent.steps == 123
