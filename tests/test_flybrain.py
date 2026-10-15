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
        "male_cns": types.SimpleNamespace(
            MaleCNS=object,
            VISUAL_ABLATIONS=(
                "normal", "threat", "loom", "small_motion", "target",
                "all", "eye", "all_visual",
            ),
        ),
        "mario_sidecar": types.SimpleNamespace(nes_actions=list),
    }
    previous = {name: sys.modules.get(name) for name in injected}
    sys.modules.update(injected)
    path = Path(__file__).parents[1] / "integrations/super_mario/evaluate_flybrain.py"
    spec = importlib.util.spec_from_file_location("microduck_test_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
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


def test_controller_temporal_stack_tracks_jump_edges_and_holds():
    flybrain = _load_flybrain()
    ppo = _load_ppo(flybrain)
    stack = ppo.ControllerTemporalStack(feature_dim=6, depth=16)
    state = stack.reset(np.ones(6, dtype=np.float32))
    assert state.shape == (16, 6 + ppo.CONTROLLER_FEATURE_DIM)

    right_jump = int(flybrain.FlybrainAction.RIGHT_JUMP)
    right = int(flybrain.FlybrainAction.RIGHT)
    state = stack.append_interval(
        [np.full(6, 1.0, dtype=np.float32), np.full(6, 3.0, dtype=np.float32)],
        right_jump,
    )
    assert np.allclose(state[-1, :6], 2.0)
    assert stack.last_diagnostics == {
        "action_hold_decisions": 1,
        "jump_hold_decisions": 1,
        "jump_pressed_edge": True,
        "jump_released_edge": False,
    }
    stack.append_interval([np.ones(6, dtype=np.float32)], right_jump)
    assert stack.last_diagnostics["action_hold_decisions"] == 2
    assert stack.last_diagnostics["jump_hold_decisions"] == 2
    assert not stack.last_diagnostics["jump_pressed_edge"]
    stack.append_interval([np.ones(6, dtype=np.float32)], right)
    assert stack.last_diagnostics["action_hold_decisions"] == 1
    assert stack.last_diagnostics["jump_hold_decisions"] == 0
    assert stack.last_diagnostics["jump_released_edge"]


def test_controller_gru_ppo_checkpoint_round_trip(tmp_path):
    flybrain = _load_flybrain()
    ppo = _load_ppo(flybrain)
    config = ppo.PPOConfig(
        feature_dim=6,
        stack_depth=16,
        temporal_encoder="controller_gru",
        controller_feature_dim=ppo.CONTROLLER_FEATURE_DIM,
        rollout_steps=4,
        minibatch_size=2,
    )
    agent = ppo.PPOAgent(config, seed=11)
    state = np.zeros((16, 6 + ppo.CONTROLLER_FEATURE_DIM), dtype=np.float32)
    assert agent.action_probabilities(state).sum() == pytest.approx(1.0)
    checkpoint = tmp_path / "controller-gru.pt"
    agent.save(checkpoint)
    restored = ppo.PPOAgent.load(checkpoint)
    assert restored.config.temporal_encoder == "controller_gru"
    assert restored.config.stack_depth == 16
    assert restored.action_probabilities(state).sum() == pytest.approx(1.0)

    states = np.zeros(
        (4, 16, 6 + ppo.CONTROLLER_FEATURE_DIM), dtype=np.float32
    )
    samples = [restored.act(value) for value in states]
    metrics = restored.update(
        states=states,
        actions=np.asarray([sample[0] for sample in samples]),
        old_log_probabilities=np.asarray([sample[1] for sample in samples]),
        returns=np.asarray([1.0, 0.5, -0.5, 2.0], dtype=np.float32),
        advantages=np.asarray([1.0, 0.5, -0.5, 2.0], dtype=np.float32),
    )
    assert metrics["batch_size"] == 4.0
    assert all(np.isfinite(value) for value in metrics.values())


def test_episode_gru_controller_preserves_each_emulator_frame():
    flybrain = _load_flybrain()
    ppo = _load_ppo(flybrain)
    controller = ppo.EpisodeGRUController(feature_dim=3)
    initial = controller.reset(np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    assert initial.shape == (1, 3 + ppo.CONTROLLER_FEATURE_DIM)
    assert np.allclose(initial[0, :3], [1.0, 2.0, 3.0])

    right_jump = int(flybrain.FlybrainAction.RIGHT_JUMP)
    tokens = controller.append_interval(
        [
            np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
            np.asarray([3.0, 3.0, 3.0], dtype=np.float32),
            np.asarray([4.0, 4.0, 4.0], dtype=np.float32),
        ],
        right_jump,
    )
    assert tokens.shape == (4, 3 + ppo.CONTROLLER_FEATURE_DIM)
    assert np.allclose(tokens[:, :3], np.arange(1, 5)[:, None])
    assert controller.last_diagnostics["jump_pressed_edge"]
    assert controller.last_diagnostics["jump_hold_decisions"] == 1


def test_episode_gru_checkpoint_and_recurrent_ppo_update(tmp_path):
    flybrain = _load_flybrain()
    ppo = _load_ppo(flybrain)
    config = ppo.PPOConfig(
        feature_dim=3,
        stack_depth=1,
        temporal_encoder="episode_gru_v1",
        controller_feature_dim=ppo.CONTROLLER_FEATURE_DIM,
        recurrent_sequence_length=2,
        rollout_steps=4,
        minibatch_size=4,
        update_epochs=2,
    )
    agent = ppo.PPOAgent(config, seed=19)
    controller = ppo.EpisodeGRUController(feature_dim=3)
    agent.reset_recurrent()
    tokens = controller.reset(np.ones(3, dtype=np.float32))
    token_sequences = []
    initial_hiddens = []
    actions = []
    log_probabilities = []
    values = []
    for index in range(4):
        initial_hiddens.append(agent.observe_recurrent(tokens))
        action, log_probability, value = agent.act_recurrent()
        token_sequences.append(tokens)
        actions.append(action)
        log_probabilities.append(log_probability)
        values.append(value)
        tokens = controller.append_interval(
            [
                np.full(3, index + frame + 1, dtype=np.float32)
                for frame in range(4)
            ],
            action,
        )
    before = {
        name: value.detach().clone() for name, value in agent.network.named_parameters()
    }
    metrics = agent.update_recurrent(
        token_sequences=token_sequences,
        initial_hiddens=np.asarray(initial_hiddens, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        old_log_probabilities=np.asarray(log_probabilities, dtype=np.float32),
        returns=np.asarray([1.0, 0.5, -0.5, 2.0], dtype=np.float32),
        advantages=np.asarray([1.0, 0.5, -0.5, 2.0], dtype=np.float32),
        dones=np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
    )
    assert metrics["batch_size"] == 4.0
    assert metrics["sequence_length"] == 2.0
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(before[name], value)
        for name, value in agent.network.named_parameters()
    )

    checkpoint = tmp_path / "episode-gru.pt"
    agent.save(checkpoint)
    restored = ppo.PPOAgent.load(checkpoint)
    assert restored.config.temporal_encoder == "episode_gru_v1"
    restored.reset_recurrent()
    restored.observe_recurrent(controller.reset(np.ones(3, dtype=np.float32)))
    probabilities = restored.policy_diagnostics_recurrent()["action_probabilities"]
    assert probabilities.sum() == pytest.approx(1.0)


def test_episode_gru_memory_persists_until_explicit_reset():
    flybrain = _load_flybrain()
    ppo = _load_ppo(flybrain)
    agent = ppo.PPOAgent(
        ppo.PPOConfig(
            feature_dim=3,
            stack_depth=1,
            temporal_encoder="episode_gru_v1",
            controller_feature_dim=ppo.CONTROLLER_FEATURE_DIM,
        ),
        seed=23,
    )
    controller = ppo.EpisodeGRUController(feature_dim=3)
    tokens = controller.reset(np.ones(3, dtype=np.float32))
    agent.reset_recurrent()
    agent.observe_recurrent(tokens)
    once = agent.recurrent_hidden()
    agent.observe_recurrent(tokens)
    twice = agent.recurrent_hidden()
    assert not np.allclose(once, twice)

    agent.reset_recurrent()
    agent.observe_recurrent(tokens)
    assert np.allclose(agent.recurrent_hidden(), once)


def test_vision_fusion_controller_separates_visual_and_compact_state():
    flybrain = _load_flybrain()
    ppo = _load_ppo(flybrain)
    controller = ppo.VisionFusionGRUController(feature_dim=3)
    initial = controller.reset(np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    assert initial.shape == (1, 3 + ppo.COMPACT_CONTROLLER_FEATURE_DIM)
    assert np.allclose(initial[0, 3:], 0.0)

    tokens = controller.append_interval(
        [np.full(3, frame, dtype=np.float32) for frame in range(1, 5)],
        int(flybrain.FlybrainAction.RIGHT_JUMP),
    )
    assert tokens.shape == (4, 3 + ppo.COMPACT_CONTROLLER_FEATURE_DIM)
    assert np.allclose(tokens[:, :3], np.arange(1, 5)[:, None])
    assert np.allclose(tokens[:, 3:], tokens[0, 3:])
    assert controller.last_diagnostics["jump_pressed_edge"]


@pytest.mark.parametrize("stabilized", [False, True])
def test_vision_fusion_recurrent_update_logs_auxiliary_and_dependence_metrics(tmp_path, stabilized):
    flybrain = _load_flybrain()
    ppo = _load_ppo(flybrain)
    config = ppo.PPOConfig(
        feature_dim=3,
        stack_depth=1,
        visual_encoder="flyvis_spatial_v1",
        temporal_encoder="vision_fusion_gru_v2",
        controller_feature_dim=ppo.COMPACT_CONTROLLER_FEATURE_DIM,
        recurrent_sequence_length=2,
        rollout_steps=4,
        minibatch_size=4,
        update_epochs=1,
        fusion_gate_warmup_steps=20_000,
        stabilized_vision_gru=stabilized,
    )
    agent = ppo.PPOAgent(config, seed=29)
    controller = ppo.VisionFusionGRUController(feature_dim=3)
    agent.reset_recurrent()
    tokens = controller.reset(np.ones(3, dtype=np.float32))
    token_sequences = []
    targets = []
    hiddens = []
    actions = []
    log_probabilities = []
    values = []
    rng = np.random.default_rng(7)
    for index in range(4):
        hiddens.append(agent.observe_recurrent(tokens))
        action, log_probability, value = agent.act_recurrent()
        token_sequences.append(tokens)
        targets.append(
            rng.normal(size=(len(tokens), ppo.FLYVIS_SPATIAL_TARGET_DIM)).astype(
                np.float32
            )
        )
        actions.append(action)
        log_probabilities.append(log_probability)
        values.append(value)
        tokens = controller.append_interval(
            [np.full(3, index + frame + 1, dtype=np.float32) for frame in range(4)],
            action,
        )
    gate_before = agent.network.fusion_gate_logit.detach().clone()
    metrics = agent.update_recurrent(
        token_sequences=token_sequences,
        visual_targets=targets,
        initial_hiddens=np.asarray(hiddens, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        old_log_probabilities=np.asarray(log_probabilities, dtype=np.float32),
        returns=np.asarray([1.0, 0.5, -0.5, 2.0], dtype=np.float32),
        advantages=np.asarray([1.0, 0.5, -0.5, 2.0], dtype=np.float32),
        dones=np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
    )

    assert metrics["vision_auxiliary_loss"] > 0
    assert metrics["vision_zero_kl"] >= 0
    assert metrics["fusion_vision_weight"] == pytest.approx(0.2 if stabilized else 0.8)
    assert metrics["hidden_saturation_fraction"] < 0.5
    assert torch.equal(gate_before, agent.network.fusion_gate_logit)
    assert isinstance(agent.network.frame_projection[1], torch.nn.LayerNorm)
    checkpoint = tmp_path / "vision-fusion.pt"
    agent.save(checkpoint)
    restored = ppo.PPOAgent.load(checkpoint)
    assert restored.config.temporal_encoder == "vision_fusion_gru_v2"
    assert restored.config.vision_auxiliary_coefficient == pytest.approx(0.1)
    assert restored.config.stabilized_vision_gru == stabilized
    if stabilized:
        assert metrics["update_gate_mean"] < 0.4
        assert metrics["update_gate_above_095_fraction"] < 0.01
        assert metrics["gru_grad_norm"] > 0
        assert metrics["vision_decoder_grad_norm"] > 0
        with torch.no_grad():
            token_t = torch.as_tensor(tokens).unsqueeze(0)
            assert torch.allclose(
                agent.network.recurrent_tokens(token_t)[0],
                restored.network.recurrent_tokens(token_t)[0],
            )


def test_observable_gru_matches_pytorch_outputs_and_gradients():
    ppo = _load_ppo(_load_flybrain())
    torch.manual_seed(12)
    stock = torch.nn.GRU(4, 6, batch_first=True)
    observable = ppo.ObservableGRU(4, 6, batch_first=True)
    observable.load_state_dict(stock.state_dict())
    inputs = torch.randn(2, 7, 4, requires_grad=True)
    hidden = torch.randn(1, 2, 6)
    output, final = stock(inputs, hidden)
    actual, actual_final = observable(inputs, hidden)
    assert torch.allclose(output, actual, atol=1e-6)
    assert torch.allclose(final, actual_final, atol=1e-6)
    expected_gradient = torch.autograd.grad(output.sum(), inputs, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual.sum(), inputs)[0]
    assert torch.allclose(expected_gradient, actual_gradient, atol=1e-6)


def test_stabilized_gru_tracks_changed_vision_without_initial_saturation():
    ppo = _load_ppo(_load_flybrain())
    config = ppo.PPOConfig(
        feature_dim=32, stack_depth=1, visual_encoder="flyvis_spatial_v1",
        temporal_encoder="vision_fusion_gru_v2", controller_feature_dim=8,
        stabilized_vision_gru=True,
    )
    agent = ppo.PPOAgent(config, seed=19)
    network = agent.network
    with torch.no_grad():
        zero = torch.zeros(1, 128, 40)
        changed = zero.clone()
        changed[:, 64:, :32] = torch.linspace(-10, 10, 32)
        _, _, baseline = network.recurrent_tokens(zero)
        _, _, response = network.recurrent_tokens(changed)
        assert response.square().mean().sqrt() < 0.6
        assert (response.abs() > 0.95).float().mean() == 0
        assert (response - baseline).square().mean().sqrt() > 0.01
        assert network.temporal.gate_metrics["update_gate_above_095_fraction"] == 0
        # Live action selection and batched training must use identical normalization.
        logits, _, hidden = network.recurrent_tokens(changed)
        live_logits, _ = network.recurrent_heads(hidden, changed[:, -1, 32:])
        assert torch.allclose(logits, live_logits, atol=1e-6)


def test_stabilized_gru_learns_visual_targets_without_saturating():
    """Exercise the auxiliary learning path, not just an untrained forward pass."""
    ppo = _load_ppo(_load_flybrain())
    config = ppo.PPOConfig(
        feature_dim=16, stack_depth=1, visual_encoder="flyvis_spatial_v1",
        temporal_encoder="vision_fusion_gru_v2", controller_feature_dim=8,
        stabilized_vision_gru=True, hidden_saturation_coefficient=0.3,
        fusion_gate_warmup_steps=0,
    )
    agent = ppo.PPOAgent(config, seed=7)
    generator = torch.Generator().manual_seed(5)
    visual = torch.randn(8, 12, 16, generator=generator)
    tokens = torch.cat((visual, torch.zeros(8, 12, 8)), dim=-1)
    mapping = torch.randn(16, ppo.FLYVIS_SPATIAL_TARGET_DIM, generator=generator) / 4
    target = torch.tanh(visual[:, -1] @ mapping)
    optimizer = torch.optim.Adam(agent.network.parameters(), lr=1e-3)
    errors = []
    for _ in range(40):
        _, _, hidden, reconstruction = agent.network.recurrent_tokens_aux(tokens)
        error = torch.nn.functional.mse_loss(reconstruction, target)
        rms = hidden.square().mean(-1).add(1e-8).sqrt()
        loss = error + 0.3 * torch.relu(rms - 0.6).square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.network.parameters(), 0.5)
        optimizer.step()
        errors.append(error.item())
    assert errors[-1] < errors[0] * 0.2
    assert (hidden.abs() > 0.95).float().mean() < 0.05
    assert torch.isfinite(hidden).all()
    assert agent.network.temporal.gate_metrics["update_gate_above_095_fraction"] < 0.05


def test_flyvis_target_normalization_is_finite_and_spatial():
    flybrain = _load_flybrain()
    ppo = _load_ppo(flybrain)
    grids = np.arange(ppo.FLYVIS_SPATIAL_TARGET_DIM, dtype=np.float32).reshape(11, 6, 8)
    target = ppo.normalize_flyvis_targets(grids)
    assert target.shape == (ppo.FLYVIS_SPATIAL_TARGET_DIM,)
    assert np.isfinite(target).all()
    assert target.std() == pytest.approx(1.0, rel=1.0e-3)


def test_spatial_readout_ppo_learns_opposite_actions_from_visual_input(tmp_path):
    """Toy visual control contract; this is not an emulator completion test."""
    ppo = _load_ppo(_load_flybrain())
    config = ppo.PPOConfig(
        feature_dim=3 + 528, stack_depth=1, controller_feature_dim=8,
        visual_encoder="flyvis_spatial_v1", temporal_encoder="vision_fusion_gru_v2",
        stabilized_vision_gru=True, spatial_readout=True,
        learning_rate=1e-3, recurrent_sequence_length=1,
        minibatch_size=32, update_epochs=2, vision_auxiliary_coefficient=0,
        fusion_gate_warmup_steps=0, value_coefficient=0.5,
    )
    agent = ppo.PPOAgent(config, seed=73)
    generator = torch.Generator().manual_seed(17)
    tokens = torch.zeros(32, 1, 539)
    tokens[:16, :, 3:267] = 0.8
    tokens[16:, :, 267:531] = 0.8
    expected = torch.tensor([2] * 16 + [5] * 16)
    for _ in range(35):
        with torch.no_grad():
            logits, values, _ = agent.network.recurrent_tokens(tokens)
            probabilities = logits.softmax(-1)
            actions = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
            rewards = (actions == expected).float() * 2 - 1
            logs = probabilities.gather(1, actions[:, None]).log().squeeze(1)
        metrics = agent.update_recurrent(
            token_sequences=list(tokens.numpy()), initial_hiddens=np.zeros((32, 256), np.float32),
            actions=actions.numpy(), old_log_probabilities=logs.numpy(),
            returns=rewards.numpy(), advantages=(rewards - values).numpy(),
            dones=np.ones(32, np.float32), visual_targets=[np.zeros((1, 528), np.float32)] * 32,
        )
    with torch.no_grad():
        predictions = agent.network.recurrent_tokens(tokens)[0].argmax(-1)
    assert (predictions == expected).float().mean() >= 0.9
    assert metrics["spatial_projection_grad_norm"] > 0
    assert metrics["value_loss"] == pytest.approx(metrics["value_loss_unscaled"])
    path = tmp_path / "spatial-readout.pt"
    agent.save(path)
    restored = ppo.PPOAgent.load(path)
    assert restored.config.spatial_readout
    with torch.no_grad():
        assert torch.allclose(agent.network.recurrent_tokens(tokens)[0], restored.network.recurrent_tokens(tokens)[0])


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


def test_factorized_ppo_exposes_forensic_head_probabilities():
    flybrain = _load_flybrain()
    ppo = _load_ppo(flybrain)
    agent = ppo.PPOAgent(
        ppo.PPOConfig(feature_dim=6, policy_mode="factorized"), seed=8
    )
    diagnostics = agent.policy_diagnostics(np.zeros((4, 6), dtype=np.float32))
    assert diagnostics["action_probabilities"].sum() == pytest.approx(1.0)
    assert diagnostics["direction_probabilities"].sum() == pytest.approx(1.0)
    assert diagnostics["jump_probabilities"].sum() == pytest.approx(1.0)
    assert diagnostics["run_probabilities"].sum() == pytest.approx(1.0)
    assert np.isfinite(diagnostics["value"])


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
    payload["config"].pop("visual_encoder")
    torch.save(payload, checkpoint)
    restored = ppo.PPOAgent.load(checkpoint)
    assert restored.config.policy_mode == "categorical"
    assert restored.config.visual_encoder == "column_v1"


def test_retina_lite_visual_contract_round_trips_with_ppo(tmp_path):
    flybrain = _load_flybrain()
    ppo = _load_ppo(flybrain)
    checkpoint = tmp_path / "retina-lite.pt"
    ppo.PPOAgent(
        ppo.PPOConfig(feature_dim=6, visual_encoder="retina_lite_v2"), seed=8
    ).save(checkpoint)

    assert ppo.PPOAgent.load(checkpoint).config.visual_encoder == "retina_lite_v2"


def test_retina_lite_preserves_lower_field_and_side_motion():
    male_cns = _load_male_cns()
    encoder = male_cns.MaleCNS.__new__(male_cns.MaleCNS)
    previous = np.ones((100, 120), dtype=np.float32)
    current = previous.copy()
    current[65:82, 86:101] = 0.0

    amounts = encoder._retina_lite_amounts(current, previous)

    assert amounts[("small_motion", "R")] > amounts[("small_motion", "L")]
    assert amounts[("loom", "R")] > amounts[("loom", "L")]
    assert amounts[("threat", "R")] > amounts[("threat", "L")]


def test_retina_diagnostic_shows_selected_right_pixels_without_state_change():
    male_cns = _load_male_cns()
    encoder = male_cns.MaleCNS.__new__(male_cns.MaleCNS)
    encoder.visual_encoder = "retina_lite_v2"
    encoder.previous_gray = np.ones((100, 120), dtype=np.float32)
    frame = np.full((100, 120, 3), 255, dtype=np.uint8)
    frame[65:82, 86:101] = 0

    diagnostic = encoder.retina_diagnostic(frame)

    assert diagnostic is not None
    assert diagnostic["camera_shift_pixels"] == 0
    assert diagnostic["motion"].shape == (100, 120)
    assert diagnostic["dark_growth"][70, 90] > 0.0
    assert diagnostic["threat"][:, 60:].sum() > 0.0
    assert diagnostic["threat"][:, :60].sum() == 0.0
    assert diagnostic["target"][:, 60:].sum() > 0.0
    np.testing.assert_array_equal(encoder.previous_gray, 1.0)


def _spatial_mario_frame(*, pipe: bool = False, gap: bool = False):
    frame = np.full((240, 256, 3), (104, 136, 252), dtype=np.uint8)
    frame[208:] = (228, 92, 16)
    frame[188:205, 68:80] = (248, 56, 0)
    if pipe:
        frame[150:208, 98:120] = (0, 168, 0)
    if gap:
        frame[208:, 140:172] = (104, 136, 252)
    return frame


def test_spatial_retina_detects_pipe_and_gap_at_separate_distances():
    male_cns = _load_male_cns()
    frame = _spatial_mario_frame(pipe=True, gap=True)

    spatial = male_cns.MaleCNS._spatial_retina(frame)

    assert 65 <= spatial["player_screen_x"] <= 82
    assert spatial["obstacle_mid"] > 0.4
    assert spatial["obstacle_near"] < spatial["obstacle_mid"]
    assert spatial["gap_far"] > 0.25
    assert spatial["gap_near"] < 0.1
    assert spatial["ground_near"] > 0.9


def test_spatial_encoder_routes_geometry_through_malecns_visual_drive():
    male_cns = _load_male_cns()
    encoder = male_cns.MaleCNS.__new__(male_cns.MaleCNS)
    encoder.visual_encoder = "retina_spatial_v3"
    encoder.visual_ablation = "normal"
    encoder.previous_gray = None
    encoder.last_visual_drive = {}
    encoder.last_retina_measurements = {}
    encoder.brain = types.SimpleNamespace(azimuth=np.linspace(-1, 1, 16))
    encoder.visual = {
        name: {side: np.asarray([index], dtype=np.int64) for side in ("L", "R")}
        for index, name in enumerate(male_cns.VISUAL_CHANNELS)
    }
    frame = _spatial_mario_frame(pipe=True, gap=True)

    eye_drive, injected = encoder._encode(frame)

    assert len(injected) > 0
    assert encoder.visual_stats()["loom_r"] > encoder.visual_stats()["loom_l"]
    assert encoder.visual_stats()["threat_r"] > encoder.visual_stats()["threat_l"]
    assert encoder.retina_stats()["gap_far"] > 0.25
    assert encoder.retina_stats()["obstacle_mid"] > 0.4
    assert np.any(eye_drive)

    encoder.visual_ablation = "all_visual"
    eye_drive, injected = encoder._encode(frame)
    assert not np.any(eye_drive)
    assert injected == []


def test_retina_lite_suppresses_horizontal_camera_scroll():
    male_cns = _load_male_cns()
    rng = np.random.default_rng(4)
    previous = rng.random((80, 120), dtype=np.float32)
    current = np.roll(previous, 3, axis=1)
    current[:, :3] = previous[:, :1]

    aligned = male_cns._align_previous(current, previous)

    assert np.mean(np.abs(current - aligned)) < 1.0e-6


def test_retina_ablation_silences_only_named_injection():
    male_cns = _load_male_cns()
    encoder = male_cns.MaleCNS.__new__(male_cns.MaleCNS)
    encoder.visual_encoder = "retina_lite_v2"
    encoder.visual_ablation = "threat"
    encoder.previous_gray = np.ones((100, 120), dtype=np.float32)
    encoder.last_visual_drive = {}
    encoder.last_retina_measurements = {}
    encoder.brain = types.SimpleNamespace(azimuth=np.asarray((-1.0, 1.0)))
    encoder.visual = {
        name: {side: np.asarray([index], dtype=np.int64) for side in ("L", "R")}
        for index, name in enumerate(male_cns.VISUAL_CHANNELS)
    }
    frame = np.full((100, 120, 3), 255, dtype=np.uint8)
    frame[65:82, 86:101] = 0

    _, injected = encoder._encode(frame)

    assert encoder.visual_stats()["threat_r"] == 0.0
    assert encoder.visual_stats()["loom_r"] > 0.0
    assert encoder.retina_stats()["dark_growth_r"] > 0.0
    assert all(not np.array_equal(indices, encoder.visual["threat"]["R"])
               for indices, _ in injected)

    encoder.visual_ablation = "all_visual"
    eye_drive, injected = encoder._encode(frame)
    assert not np.any(eye_drive)
    assert injected == []


def test_male_cns_seeded_reset_reproduces_neural_noise():
    male_cns = _load_male_cns()

    class FakeBrain:
        def reset(self, seed=None):
            self.rng = np.random.default_rng(seed)

    encoder = male_cns.MaleCNS.__new__(male_cns.MaleCNS)
    encoder.brain = FakeBrain()
    encoder.trace = types.SimpleNamespace(reset=lambda: None)
    encoder.dopamine = None
    encoder.observe = lambda _frame: np.asarray([encoder.brain.rng.random()])
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    first = encoder.reset(frame, seed=42)
    second = encoder.reset(frame, seed=42)

    assert np.array_equal(first, second)


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


@pytest.mark.parametrize("readout, expected_dt", [(False, 0.001), (True, 1 / 60)])
def test_new_spatial_readout_uses_video_clock_preserving_legacy(tmp_path, monkeypatch, readout, expected_dt):
    class Brain:
        dt = 0.001
        def __init__(self, **kwargs):
            pass
        def cells(self, names, side=None):
            return np.array([0, 1])
    class Bridge:
        def __init__(self, **kwargs):
            self.dt = kwargs["dt"]
        def close(self):
            pass
    monkeypatch.setitem(sys.modules, "flybrain", types.SimpleNamespace(
        FlyBrain=Brain, Trace=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, "flyvis_spatial_bridge", types.SimpleNamespace(FlyVisSpatialBridge=Bridge))
    male_cns = _load_male_cns()
    backend = male_cns.MaleCNS(
        visual_encoder="flyvis_spatial_v1", spatial_readout=readout,
        flyvis_model=tmp_path, flyvis_python=tmp_path, flyvis_worker=tmp_path,
    )
    assert backend.feature_dim == 2 + (528 if readout else 0)
    assert backend.flyvis_frame_dt == pytest.approx(expected_dt)
    backend.close()
    assert backend.flyvis_frame_dt == pytest.approx(expected_dt)


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


def test_training_reward_matches_vietnh1009_shaping():
    rollouts = _load_rollouts()
    assert rollouts.training_reward({}, raw_reward=12.0) == pytest.approx(1.2)
    assert rollouts.training_reward(
        {"score": 1.0}, raw_reward=1.0
    ) == pytest.approx(0.35)
    assert rollouts.training_reward(
        {"completion": 50.0}, raw_reward=15.0
    ) == pytest.approx(6.5)
    assert rollouts.training_reward(
        {"death": -25.0}, raw_reward=-15.0
    ) == pytest.approx(-6.5)
    with pytest.raises(ValueError, match="raw Gymnasium reward"):
        rollouts.training_reward({"death": -25.0})


def test_frozen_evaluator_exposes_old_fatal_progress_reward_bug():
    flybrain = _load_flybrain()
    rollouts = _load_rollouts()
    evaluator = _load_evaluator(flybrain, rollouts)
    fatal_progress = {"progress": 150.0, "death": -25.0}

    assert evaluator._legacy_training_reward(fatal_progress, 0.0) == 5.0
    assert rollouts.training_reward(
        fatal_progress, raw_reward=-7.0
    ) == pytest.approx(-5.7)


def test_forensic_recorder_writes_gif_and_synchronized_jsonl(tmp_path):
    flybrain = _load_flybrain()
    rollouts = _load_rollouts()
    evaluator = _load_evaluator(flybrain, rollouts)
    recorder = evaluator.ForensicRecorder(tmp_path, episode=2, fps=60)
    recorder.add_frame(
        np.zeros((16, 20, 3), dtype=np.uint8),
        {"episode": 2, "decision": 1, "x": 42},
        ["action=right", "x=42"],
    )
    recorder.add_decision(
        {"episode": 2, "decision": 1, "action": "right", "x_after": 42}
    )
    artifacts = recorder.close()
    for path in artifacts.values():
        assert Path(path).is_file()
    assert json.loads((tmp_path / "episode-002-frames.jsonl").read_text()) == {
        "decision": 1,
        "episode": 2,
        "x": 42,
    }
    assert json.loads((tmp_path / "episode-002-decisions.jsonl").read_text())[
        "action"
    ] == "right"


def test_forensic_recorder_writes_opt_in_retina_view_and_timeline(tmp_path):
    flybrain = _load_flybrain()
    rollouts = _load_rollouts()
    evaluator = _load_evaluator(flybrain, rollouts)
    recorder = evaluator.ForensicRecorder(
        tmp_path, episode=0, fps=60, retina_view=True
    )
    frame = np.zeros((32, 40, 3), dtype=np.uint8)
    diagnostic = {
        name: np.zeros((32, 40), dtype=np.float32)
        for name in ("motion", "dark_growth", "edges", "small_motion", "loom", "threat", "target")
    }
    diagnostic["threat"][20, 30] = 0.2
    diagnostic["camera_shift_pixels"] = -2
    visual = {
        f"{name}_{side}": 0.02
        for name in ("small_motion", "loom", "threat", "target")
        for side in ("l", "r")
    }
    row = {
        "episode": 0, "decision": 1, "frame_in_decision": 4,
        "x": 42, "y": 79, "action": "right_jump", "death": False,
        "visual_drive": visual, "jump_pressed_edge": True,
        "jump_released_edge": False, "jump_hold_decisions": 1,
        "jump_metrics": {"total_jump_probability": 0.62},
    }
    recorder.add_frame(frame, row, ["action=right_jump"])
    recorder.add_retina_frame(frame, diagnostic, row)
    recorder.add_decision({"decision": 1, "action": "right_jump"})

    artifacts = recorder.close()

    assert Path(artifacts["retina_video"]).is_file()
    assert Path(artifacts["retina_timeline"]).is_file()


def test_forensic_recorder_writes_spatial_retina_timeline(tmp_path):
    flybrain = _load_flybrain()
    rollouts = _load_rollouts()
    evaluator = _load_evaluator(flybrain, rollouts)
    recorder = evaluator.ForensicRecorder(
        tmp_path, episode=1, fps=60, retina_view=True
    )
    frame = _spatial_mario_frame(pipe=True, gap=True)
    male_cns = _load_male_cns()
    encoder = male_cns.MaleCNS.__new__(male_cns.MaleCNS)
    encoder.visual_encoder = "retina_spatial_v3"
    encoder.previous_gray = None
    diagnostic = encoder.retina_diagnostic(frame)
    spatial = encoder._spatial_retina(frame)
    row = {
        "episode": 1, "decision": 1, "frame_in_decision": 4,
        "x": 70, "y": 79, "action": "right_jump", "death": False,
        "visual_drive": {
            f"{name}_{side}": 0.1
            for name in ("small_motion", "loom", "threat", "target")
            for side in ("l", "r")
        },
        "retina_measurements": {
            key: value for key, value in spatial.items()
            if key not in ("ground_profile", "obstacle_profile")
        },
        "jump_pressed_edge": True, "jump_released_edge": False,
        "jump_hold_decisions": 1,
        "jump_metrics": {"total_jump_probability": 0.62},
    }
    recorder.add_frame(frame, row, ["action=right_jump"])
    recorder.add_retina_frame(frame, diagnostic, row)
    recorder.add_decision({"decision": 1, "action": "right_jump"})

    artifacts = recorder.close()

    assert Path(artifacts["retina_video"]).is_file()
    assert Path(artifacts["spatial_timeline"]).is_file()


def test_forensic_recorder_writes_opt_in_flyvis_view(tmp_path):
    flybrain = _load_flybrain()
    rollouts = _load_rollouts()
    evaluator = _load_evaluator(flybrain, rollouts)
    recorder = evaluator.ForensicRecorder(
        tmp_path, episode=3, fps=12, flyvis_view=True
    )
    frame = np.zeros((32, 40, 3), dtype=np.uint8)
    row = {
        "episode": 3,
        "decision": 1,
        "x": 42,
        "action": "right_jump",
        "death": False,
    }
    grids = np.zeros((11, 6, 8), dtype=np.float32)
    grids[3, 2, 5] = 0.7
    recorder.add_frame(frame, row, ["action=right_jump"])
    recorder.add_flyvis_frame(frame, grids, row)
    recorder.add_decision({"decision": 1, "action": "right_jump"})

    artifacts = recorder.close()

    assert Path(artifacts["flyvis_video"]).is_file()


def test_paired_replay_uses_same_frames_and_actions_for_channel_masks(monkeypatch):
    flybrain = _load_flybrain()
    rollouts = _load_rollouts()
    evaluator = _load_evaluator(flybrain, rollouts)
    ppo = _load_ppo(flybrain)
    monkeypatch.setitem(sys.modules, "mario_ppo", ppo)

    class FakeConnectome:
        visual_ablation = "normal"

        def reset(self, _frame, *, seed=None):
            assert seed == 19
            return np.zeros(2, dtype=np.float32)

        def observe(self, frame):
            value = float(frame[0, 0, 0]) / 255.0
            if self.visual_ablation == "all":
                value = 0.0
            return np.asarray((value, value), dtype=np.float32)

    class FakeAgent:
        config = types.SimpleNamespace(
            feature_dim=2, stack_depth=4, temporal_encoder="conv4"
        )

        def policy_diagnostics(self, state):
            jump = float(np.mean(state)) * 0.5
            scores = np.full(10, (1.0 - jump) / 5.0, dtype=np.float64)
            scores[list(evaluator.JUMP_ACTIONS)] = jump / 5.0
            return {"action_probabilities": scores}

    connectome = FakeConnectome()
    agent = FakeAgent()
    initial = np.zeros((2, 2, 3), dtype=np.uint8)
    bright = np.full((2, 2, 3), 255, dtype=np.uint8)
    intervals = [(int(flybrain.FlybrainAction.RIGHT), [bright] * 4)] * 2
    rows = [
        {"action_probabilities": dict(zip(
            [action.name.lower() for action in flybrain.FlybrainAction],
            agent.policy_diagnostics(
                np.zeros((4, 2), dtype=np.float32)
                if index == 0 else np.ones((4, 2), dtype=np.float32)
            )["action_probabilities"].tolist(),
            strict=True,
        ))}
        for index in range(2)
    ]

    baseline_error = evaluator._replay_counterfactuals(
        agent, connectome, initial, intervals, rows, episode_seed=19
    )

    assert baseline_error < 1.0e-8
    assert connectome.visual_ablation == "normal"
    assert rows[1]["counterfactual"]["normal"]["total_jump_probability"] == pytest.approx(0.5)
    assert rows[1]["counterfactual"]["all"]["total_jump_probability"] == pytest.approx(0.0)


def test_ppo_jump_sensitivity_reports_each_input_frame(monkeypatch):
    flybrain = _load_flybrain()
    evaluator = _load_evaluator(flybrain, _load_rollouts())
    ppo = _load_ppo(flybrain)
    monkeypatch.setitem(sys.modules, "mario_ppo", ppo)
    agent = ppo.PPOAgent(ppo.PPOConfig(feature_dim=6), seed=3)

    sensitivity = evaluator._ppo_input_sensitivity(
        agent, np.ones((4, 6), dtype=np.float32)
    )

    assert len(sensitivity["jump_probability_gradient_rms_by_frame"]) == 4
    assert all(np.isfinite(value) for values in sensitivity.values() for value in values)


def test_episode_gru_sensitivity_restores_evaluation_mode(monkeypatch):
    flybrain = _load_flybrain()
    evaluator = _load_evaluator(flybrain, _load_rollouts())
    ppo = _load_ppo(flybrain)
    monkeypatch.setitem(sys.modules, "mario_ppo", ppo)
    agent = ppo.PPOAgent(
        ppo.PPOConfig(
            feature_dim=6,
            stack_depth=1,
            temporal_encoder="episode_gru_v1",
            controller_feature_dim=ppo.CONTROLLER_FEATURE_DIM,
        ),
        seed=5,
    )
    agent.network.eval()
    tokens = np.ones(
        (4, 6 + ppo.CONTROLLER_FEATURE_DIM), dtype=np.float32
    )
    initial_hidden = np.zeros(ppo.EPISODE_GRU_HIDDEN_DIM, dtype=np.float32)

    sensitivity = evaluator._ppo_recurrent_input_sensitivity(
        agent, tokens, initial_hidden
    )

    assert not agent.network.training
    assert len(sensitivity["jump_probability_gradient_rms_by_frame"]) == 4
    assert all(np.isfinite(value) for values in sensitivity.values() for value in values)


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
    assert metadata["reward_contract"] == "vietnh1009-shaped-action-interval-v1"
    assert arrays["actions"].tolist() == [2, 4]
    assert np.allclose(arrays["training_rewards"], [0.3, -7.5])
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

    with pytest.raises(ValueError, match="current Mario reward contract"):
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
