import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


def _load_flybrain():
    path = Path(__file__).parents[1] / "integrations/super_mario/flybrain.py"
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
    sys.modules["flybrain"] = flybrain
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
    with pytest.raises(ValueError):
        flybrain.action_levels(10)


def test_frame_preprocessing_and_stack_have_cnn_shape():
    flybrain = _load_flybrain()
    rgb = np.zeros((240, 256, 3), dtype=np.uint8)
    rgb[:, :, 1] = 200
    frame = flybrain.preprocess_frame(rgb)
    stack = flybrain.FrameStack(4)
    state = stack.reset(frame)
    assert frame.shape == (84, 84)
    assert frame.dtype == np.uint8
    assert state.shape == (4, 84, 84)
    assert np.array_equal(state[0], state[-1])


def test_replay_samples_self_contained_pre_and_post_action_states():
    flybrain = _load_flybrain()
    replay = flybrain.PrioritizedReplay(2, (4, 36, 36), seed=3)

    def stack(*values):
        return np.stack(
            [np.full((36, 36), value, dtype=np.uint8) for value in values]
        )

    replay.add(stack(1, 2, 3, 4), 0, 0.0, stack(2, 3, 4, 5), False)
    replay.add(stack(11, 12, 13, 14), 1, 1.0, stack(12, 13, 14, 15), True)
    # Overwrite the oldest circular-buffer entry. The remaining transition must
    # not need that old entry to reconstruct its own current state.
    replay.add(stack(21, 22, 23, 24), 2, 2.0, stack(22, 23, 24, 25), False)
    sample = replay.sample(2, beta=0.4)
    assert sample[0].shape == (2, 4, 36, 36)
    assert sample[3].shape == (2, 4, 36, 36)
    transitions = {
        int(action): (state[:, 0, 0].tolist(), next_state[:, 0, 0].tolist())
        for state, action, next_state in zip(sample[0], sample[1], sample[3], strict=True)
    }
    assert transitions == {
        1: ([11, 12, 13, 14], [12, 13, 14, 15]),
        2: ([21, 22, 23, 24], [22, 23, 24, 25]),
    }


def test_replay_rejects_a_next_state_that_is_not_after_the_action():
    flybrain = _load_flybrain()
    replay = flybrain.PrioritizedReplay(2, (4, 36, 36))
    state = np.zeros((4, 36, 36), dtype=np.uint8)
    invalid_next_state = np.ones((4, 36, 36), dtype=np.uint8)
    with pytest.raises(ValueError, match="observation after the action"):
        replay.add(state, 0, 0.0, invalid_next_state, False)


def test_dueling_network_outputs_one_q_value_per_action():
    flybrain = _load_flybrain()
    network = flybrain.DuelingQNetwork(frame_size=84)
    output = network(torch.zeros(2, 4, 84, 84, dtype=torch.uint8))
    assert output.shape == (2, 10)


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
    }
    assert rollouts.decode_reward_packet(rollouts.encode_reward_packet(event)) == event


def test_rollout_recorder_writes_atomic_replay_ready_episode(tmp_path):
    rollouts = _load_rollouts()
    recorder = rollouts.RolloutRecorder(tmp_path, 4, 36, run_id="run")

    def stack(*values):
        return np.stack(
            [np.full((36, 36), value, dtype=np.uint8) for value in values]
        )

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
    assert arrays["training_rewards"].tolist() == [1.0, -1.0]
    assert arrays["action_sequences"].tolist() == [10, 11]
    reconstructed = np.concatenate(
        (arrays["states"][1, 1:], arrays["post_action_frames"][1, None]), axis=0
    )
    assert reconstructed[:, 0, 0].tolist() == [3, 4, 5, 6]
    assert not list(tmp_path.glob("*.tmp"))
    transitions = [json.loads(line) for line in (tmp_path / "transitions.jsonl").read_text().splitlines()]
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


def test_completed_rollout_can_be_ingested_into_per_and_trained(tmp_path):
    flybrain = _load_flybrain()
    rollouts = _load_rollouts()
    trainer = _load_rollout_trainer(flybrain, rollouts)
    recorder = rollouts.RolloutRecorder(tmp_path, 4, 36, run_id="training")

    def stack(*values):
        return np.stack(
            [np.full((36, 36), value, dtype=np.uint8) for value in values]
        )

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
        frame_size=36,
        stack_depth=4,
        batch_size=1,
        replay_capacity=3,
        replay_start=1,
        train_every=1,
    )
    agent = flybrain.FlybrainAgent(config, seed=4)
    replay = flybrain.PrioritizedReplay(3, (4, 36, 36), seed=4)
    count, reward, loss = trainer.ingest_rollout(
        next(tmp_path.glob("rollout-*.npz")), replay, agent
    )
    assert count == 2
    assert reward == -23.0
    assert loss is not None
    assert len(replay) == 2
