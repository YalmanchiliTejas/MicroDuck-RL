import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch


def _load_flybrain():
    path = Path(__file__).parents[1] / "integrations/super_mario/mario_dqn.py"
    spec = importlib.util.spec_from_file_location("microduck_test_flybrain", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_rollouts():
    path = Path(__file__).parents[1] / "integrations/super_mario/rollouts.py"
    spec = importlib.util.spec_from_file_location("microduck_test_rollouts", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
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


def test_training_reward_preserves_progress_and_terminal_magnitude():
    rollouts = _load_rollouts()
    assert rollouts.training_reward({"progress": 3.0}) == pytest.approx(0.3)
    assert rollouts.training_reward({"death": -25.0}) == pytest.approx(-2.5)
    assert rollouts.training_reward({"completion": 50.0}) == pytest.approx(5.0)
    assert rollouts.training_reward({}, raw_reward=12.0) == pytest.approx(1.2)


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
    plasticity.reinforce(+1.0)
    assert brain.weights[0] < 0.5
    assert brain.weights[1] == pytest.approx(0.6)
    assert plasticity.consume_injection()[0][0].tolist() == [4]

    plasticity.reset_episode()
    plasticity.observe(np.asarray((1,), dtype=np.int64))
    plasticity.reinforce(-1.0)
    assert brain.weights[1] < 0.6
    assert plasticity.consume_injection()[0][0].tolist() == [5]
    plasticity.save()

    restored_brain = TinyBrain()
    restored = dopamine.DopaminePlasticity(
        restored_brain, state_path=state, learning_rate=0.1, recovery_rate=0.0
    )
    assert restored.updates == 2
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
    assert arrays["actions"].tolist() == [2, 4]
    assert np.allclose(arrays["training_rewards"], [0.3, -2.5])
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
