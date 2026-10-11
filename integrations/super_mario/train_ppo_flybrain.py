"""Pretrain the MaleCNS Mario readout with discrete PPO."""

from __future__ import annotations

import argparse
import shutil
import signal
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
from male_cns import VISUAL_ENCODERS, MaleCNS
from mario_dqn import ActivityStack, FlybrainAction
from mario_ppo import (
    CONTROLLER_FEATURE_DIM,
    CONTROLLER_GRU_DEPTH,
    ControllerTemporalStack,
    PPOAgent,
    PPOConfig,
    generalized_advantage_estimates,
    rollout_ready,
)
from mario_sidecar import nes_actions
from reward_contract import REWARD_CONTRACT
from rollouts import REWARD_COMPONENTS, training_reward
from torch.utils.tensorboard import SummaryWriter


def _device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _atomic_agent_save(agent: PPOAgent, output: Path) -> None:
    temporary = output.with_name(f".{output.name}.tmp")
    agent.save(temporary)
    temporary.replace(output)


def _save(
    agent: PPOAgent,
    connectome: MaleCNS,
    output: Path,
    snapshot_dir: Path | None = None,
) -> None:
    _atomic_agent_save(agent, output)
    connectome.save_plasticity()
    if snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot = snapshot_dir / f"flybrain-ppo-step-{agent.steps:09d}.pt"
        _atomic_agent_save(agent, snapshot)
        if connectome.dopamine is not None:
            dopamine_snapshot = (
                snapshot_dir / f"dopamine-plasticity-step-{agent.steps:09d}.npz"
            )
            temporary = dopamine_snapshot.with_name(f".{dopamine_snapshot.name}.tmp")
            shutil.copy2(connectome.dopamine.state_path, temporary)
            temporary.replace(dopamine_snapshot)


def _flush(
    agent: PPOAgent,
    rollout: dict[str, list],
    next_state: np.ndarray,
    done: bool,
) -> dict[str, float] | None:
    if not rollout["states"]:
        return None
    next_value = 0.0 if done else agent.value(next_state)
    advantages, returns = generalized_advantage_estimates(
        np.asarray(rollout["rewards"], dtype=np.float32),
        np.asarray(rollout["values"], dtype=np.float32),
        np.asarray(rollout["dones"], dtype=np.float32),
        next_value=next_value,
        gamma=agent.config.gamma,
        gae_lambda=agent.config.gae_lambda,
    )
    metrics = agent.update(
        states=np.asarray(rollout["states"], dtype=np.float32),
        actions=np.asarray(rollout["actions"], dtype=np.int64),
        old_log_probabilities=np.asarray(rollout["log_probabilities"], dtype=np.float32),
        returns=returns,
        advantages=advantages,
    )
    for values in rollout.values():
        values.clear()
    return metrics


def _record_update_metrics(writer, step: int, metrics: dict[str, float]) -> None:
    """Persist PPO update diagnostics and mirror the key values to stdout."""
    for name, metric in metrics.items():
        writer.add_scalar(f"loss/{name}", metric, step)
    writer.flush()
    print(
        "ppo_update "
        f"step={step} "
        f"approx_kl={metrics['approx_kl']:.6f} "
        f"entropy={metrics['entropy']:.6f} "
        f"direction_entropy={metrics['direction_entropy']:.6f} "
        f"jump_entropy={metrics['jump_entropy']:.6f} "
        f"run_entropy={metrics['run_entropy']:.6f} "
        f"clip_fraction={metrics['clip_fraction']:.6f} "
        f"batch_size={int(metrics['batch_size'])} "
        f"epochs={int(metrics['epochs_completed'])} "
        f"early_stop={int(metrics['early_stop'])}",
        flush=True,
    )


def run(args: argparse.Namespace) -> None:
    import gym_super_mario_bros  # noqa: F401
    import gymnasium as gym
    from nes_py.wrappers import JoypadSpace

    connectome = MaleCNS(
        data=args.male_cns_data,
        device=args.male_cns_device,
        seed=args.seed,
        spike_file=args.spike_file,
        dopamine_state=args.dopamine_state,
        dopamine_learning_rate=args.dopamine_learning_rate,
        visual_encoder=args.visual_encoder,
    )
    if args.resume is not None:
        requested_policy_mode = "factorized" if args.factorized_policy else None
        agent = PPOAgent.load(
            args.resume,
            device=_device(args.device),
            policy_mode=requested_policy_mode,
        )
        print(f"resumed PPO={args.resume} step={agent.steps}", flush=True)
        if args.temporal_controller and agent.config.temporal_encoder != "controller_gru":
            raise ValueError(
                "--temporal-controller cannot convert an existing conv4 checkpoint; "
                "start a fresh run"
            )
        if agent.config.visual_encoder != args.visual_encoder:
            raise ValueError(
                f"checkpoint requires visual encoder {agent.config.visual_encoder!r}, "
                f"not {args.visual_encoder!r}"
            )
    else:
        agent = PPOAgent(
            PPOConfig(
                feature_dim=connectome.feature_dim,
                rollout_steps=args.rollout_steps,
                policy_mode=(
                    "factorized" if args.factorized_policy else "categorical"
                ),
                visual_encoder=args.visual_encoder,
                temporal_encoder=(
                    "controller_gru" if args.temporal_controller else "conv4"
                ),
                stack_depth=(
                    args.temporal_decisions if args.temporal_controller else 4
                ),
                controller_feature_dim=(
                    CONTROLLER_FEATURE_DIM if args.temporal_controller else 0
                ),
            ),
            device=_device(args.device),
            seed=args.seed,
        )
    agent.configure_continuation(
        learning_rate=args.continuation_learning_rate,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        target_kl=args.target_kl,
    )
    print(
        "PPO optimizer "
        f"policy={agent.config.policy_mode} "
        f"visual={agent.config.visual_encoder} "
        f"temporal={agent.config.temporal_encoder} "
        f"history={agent.config.stack_depth} "
        f"lr={agent.config.learning_rate:g} value_coef={agent.config.value_coefficient:g} "
        f"entropy_coef={agent.config.entropy_coefficient:g} "
        f"target_kl={agent.config.target_kl:g}",
        flush=True,
    )
    if agent.config.feature_dim != connectome.feature_dim:
        raise ValueError("PPO checkpoint and MaleCNS feature dimensions differ")
    target_steps = (
        agent.steps + args.additional_steps
        if args.additional_steps is not None
        else args.steps
    )

    env = JoypadSpace(
        gym.make(args.env, render_mode="rgb_array"),
        nes_actions(),
    )
    observation, _ = env.reset(seed=args.seed + agent.steps)
    stack = (
        ControllerTemporalStack(agent.config.feature_dim, agent.config.stack_depth)
        if agent.config.temporal_encoder == "controller_gru"
        else ActivityStack(agent.config.stack_depth)
    )
    state = stack.reset(connectome.reset(observation))
    writer = SummaryWriter(
        log_dir=str(args.tensorboard_dir),
        purge_step=agent.steps if agent.steps else None,
    )
    writer.add_text("training/reward_contract", REWARD_CONTRACT, 0)
    writer.add_scalar("training/action_repeat", args.action_repeat, agent.steps)
    writer.add_scalar("training/ppo_frozen", float(args.freeze_ppo), agent.steps)
    writer.add_scalar(
        "training/dopamine_frozen", float(args.freeze_dopamine), agent.steps
    )
    writer.flush()
    rollout = {
        "states": [],
        "actions": [],
        "log_probabilities": [],
        "rewards": [],
        "dones": [],
        "values": [],
    }
    recent_returns: deque[float] = deque(maxlen=100)
    action_counts: Counter[int] = Counter()
    recent_actions: deque[int] = deque(maxlen=1_000)
    episode_raw_return = 0.0
    episode_training_return = 0.0
    episode_decisions = 0
    episode_max_x = 0
    episode_jump_press_edges = 0
    episode_jump_release_edges = 0
    episode_max_action_hold = 0
    episode_max_jump_hold = 0
    episode = 0
    stop = False
    last_save = agent.steps
    started = time.monotonic()

    def request_stop(_signum, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        while agent.steps < target_steps and not stop:
            action, log_probability, value = agent.act(state)
            raw_reward = 0.0
            reward_components: dict[str, float] = {}
            interval_activities: list[np.ndarray] = []
            terminated = truncated = False
            info: dict = {}
            for _ in range(args.action_repeat):
                observation, reward, terminated, truncated, info = env.step(action)
                raw_reward += float(reward)
                for key, component_value in info.get(
                    "reward_components", {}
                ).items():
                    if key in REWARD_COMPONENTS:
                        reward_components[key] = reward_components.get(
                            key, 0.0
                        ) + float(component_value)
                activity = connectome.observe(
                    observation, action_sequence=agent.steps
                )
                if agent.config.temporal_encoder == "controller_gru":
                    interval_activities.append(activity)
                else:
                    stack.append(activity)
                if terminated or truncated:
                    break
            if agent.config.temporal_encoder == "controller_gru":
                next_state = stack.append_interval(interval_activities, action)
                controller = stack.last_diagnostics
                episode_jump_press_edges += int(controller["jump_pressed_edge"])
                episode_jump_release_edges += int(controller["jump_released_edge"])
                episode_max_action_hold = max(
                    episode_max_action_hold,
                    int(controller["action_hold_decisions"]),
                )
                episode_max_jump_hold = max(
                    episode_max_jump_hold,
                    int(controller["jump_hold_decisions"]),
                )
                writer.add_scalar(
                    "controller/action_hold_decisions",
                    controller["action_hold_decisions"],
                    agent.steps,
                )
                writer.add_scalar(
                    "controller/jump_hold_decisions",
                    controller["jump_hold_decisions"],
                    agent.steps,
                )
                writer.add_scalar(
                    "controller/jump_pressed_edge",
                    float(controller["jump_pressed_edge"]),
                    agent.steps,
                )
                writer.add_scalar(
                    "controller/jump_released_edge",
                    float(controller["jump_released_edge"]),
                    agent.steps,
                )
            else:
                next_state = stack.state
            done = bool(terminated or truncated)
            learning_reward = training_reward(
                reward_components, raw_reward=raw_reward
            )
            if not args.freeze_ppo:
                rollout["states"].append(state.copy())
                rollout["actions"].append(action)
                rollout["log_probabilities"].append(log_probability)
                rollout["rewards"].append(learning_reward)
                rollout["dones"].append(done)
                rollout["values"].append(value)
            agent.steps += 1
            action_counts[action] += 1
            recent_actions.append(action)
            episode_raw_return += raw_reward
            episode_training_return += learning_reward
            episode_decisions += 1
            episode_max_x = max(episode_max_x, int(info.get("x_pos", 0)))
            writer.add_scalar(
                "reward/action_interval", learning_reward, agent.steps
            )
            writer.add_scalar(
                "reward/action_interval_raw", raw_reward, agent.steps
            )
            writer.add_scalar("training/x_pos", int(info.get("x_pos", 0)), agent.steps)
            writer.add_scalar("training/action", action, agent.steps)
            for channel, amount in connectome.visual_stats().items():
                writer.add_scalar(f"retina/{channel}", amount, agent.steps)
            if not args.freeze_dopamine:
                connectome.reinforce(
                    agent.prediction_error(
                        learning_reward, next_state, done, value=value
                    )
                )
            state = next_state

            # Accumulate across episode boundaries. The stored done mask keeps
            # GAE from bootstrapping through a terminal state, while waiting
            # for a full rollout avoids noisy 4-10 transition PPO updates when
            # an exploratory policy dies early.
            if (
                not args.freeze_ppo
                and rollout_ready(
                    len(rollout["states"]), agent.config.rollout_steps
                )
            ):
                metrics = _flush(agent, rollout, state, done)
                if metrics is not None:
                    _record_update_metrics(writer, agent.steps, metrics)

            if done:
                episode += 1
                recent_returns.append(episode_training_return)
                average = float(np.mean(recent_returns))
                writer.add_scalar(
                    "reward/episode_return",
                    episode_training_return,
                    agent.steps,
                )
                writer.add_scalar(
                    "reward/episode_raw_return", episode_raw_return, agent.steps
                )
                writer.add_scalar("reward/average_100_episodes", average, agent.steps)
                writer.add_scalar("episode/max_x", episode_max_x, agent.steps)
                recent_action_counts = Counter(recent_actions)
                for action_id, action_name in enumerate(FlybrainAction):
                    writer.add_scalar(
                        f"actions/recent_fraction_{action_name.name.lower()}",
                        recent_action_counts[action_id] / max(1, len(recent_actions)),
                        agent.steps,
                    )
                dopamine = connectome.dopamine_stats()
                if dopamine is not None:
                    writer.add_scalar("dopamine/rpe", dopamine["signal"], agent.steps)
                    writer.add_scalar(
                        "dopamine/raw_prediction_error",
                        dopamine["raw_prediction_error"],
                        agent.steps,
                    )
                    writer.add_scalar(
                        "dopamine/normalized_prediction_error",
                        dopamine["normalized_prediction_error"],
                        agent.steps,
                    )
                    writer.add_scalar(
                        "dopamine/prediction_error_rms",
                        dopamine["prediction_error_rms"],
                        agent.steps,
                    )
                    writer.add_scalar(
                        "dopamine/mean_kc_mbon_scale",
                        dopamine["mean_kc_mbon_scale"],
                        agent.steps,
                    )
                print(
                    f"episode={episode} step={agent.steps} decisions={episode_decisions} "
                    f"raw_return={episode_raw_return:+.1f} "
                    f"training_return={episode_training_return:+.3f} "
                    f"avg100={average:+.3f} "
                    f"max_x={episode_max_x}"
                    + (
                        f" jump_edges={episode_jump_press_edges}/{episode_jump_release_edges} "
                        f"max_action_hold={episode_max_action_hold} "
                        f"max_jump_hold={episode_max_jump_hold}"
                        if agent.config.temporal_encoder == "controller_gru"
                        else ""
                    ),
                    flush=True,
                )
                observation, _ = env.reset()
                state = stack.reset(connectome.reset(observation))
                episode_raw_return = 0.0
                episode_training_return = 0.0
                episode_decisions = 0
                episode_max_x = 0
                episode_jump_press_edges = 0
                episode_jump_release_edges = 0
                episode_max_action_hold = 0
                episode_max_jump_hold = 0

            if agent.steps - last_save >= args.save_every:
                _save(agent, connectome, args.output, args.snapshot_dir)
                writer.flush()
                last_save = agent.steps
                rate = agent.steps / max(time.monotonic() - started, 1.0e-6)
                print(
                    f"saved={args.output} step={agent.steps} decisions_per_s={rate:.2f}",
                    flush=True,
                )
    finally:
        metrics = _flush(agent, rollout, state, False)
        if metrics is not None:
            _record_update_metrics(writer, agent.steps, metrics)
        for action, count in action_counts.items():
            writer.add_scalar(f"actions/count_{action}", count, agent.steps)
        _save(agent, connectome, args.output, args.snapshot_dir)
        writer.flush()
        writer.close()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument(
        "--additional-steps",
        type=int,
        help="train this many more decisions from either a fresh or resumed checkpoint",
    )
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--output", type=Path, default=Path("flybrain-ppo.pt"))
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("tensorboard/ppo"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--male-cns-data", type=Path)
    parser.add_argument(
        "--visual-encoder",
        choices=VISUAL_ENCODERS,
        default="column_v1",
        help="fixed visual preprocessing contract stored in PPO checkpoints",
    )
    parser.add_argument("--male-cns-device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--spike-file", type=Path)
    parser.add_argument("--dopamine-state", type=Path)
    parser.add_argument("--dopamine-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--continuation-learning-rate", type=float)
    parser.add_argument("--value-coefficient", type=float)
    parser.add_argument("--entropy-coefficient", type=float)
    parser.add_argument("--target-kl", type=float)
    parser.add_argument(
        "--temporal-controller",
        action="store_true",
        help="use decision-scale MaleCNS aggregation, controller feedback, and a GRU",
    )
    parser.add_argument(
        "--temporal-decisions",
        type=int,
        default=CONTROLLER_GRU_DEPTH,
        help="number of action intervals visible to the controller GRU",
    )
    parser.add_argument(
        "--factorized-policy",
        action="store_true",
        help=(
            "use direction/jump/run PPO heads; categorical checkpoints retain "
            "their encoder and critic and migrate the action head"
        ),
    )
    parser.add_argument(
        "--freeze-ppo",
        action="store_true",
        help="update dopamine plasticity while preserving PPO network/optimizer weights",
    )
    parser.add_argument(
        "--freeze-dopamine",
        action="store_true",
        help="load learned dopamine synapses but do not update them",
    )
    args = parser.parse_args()
    if min(args.steps, args.action_repeat, args.rollout_steps, args.save_every) <= 0:
        parser.error("steps, action repeat, rollout steps, and save interval must be positive")
    if args.additional_steps is not None and args.additional_steps <= 0:
        parser.error("--additional-steps must be positive")
    if args.temporal_decisions < 2:
        parser.error("--temporal-decisions must be at least two")
    if args.dopamine_learning_rate <= 0:
        parser.error("--dopamine-learning-rate must be positive")
    continuation_values = (
        args.continuation_learning_rate,
        args.value_coefficient,
        args.entropy_coefficient,
        args.target_kl,
    )
    if any(value is not None and value <= 0 for value in continuation_values):
        parser.error("PPO continuation overrides must be positive")
    if args.dopamine_state is not None and args.male_cns_device != "cpu":
        parser.error("--dopamine-state requires --male-cns-device cpu")
    if args.freeze_ppo and args.dopamine_state is None:
        parser.error("--freeze-ppo requires --dopamine-state")
    if args.freeze_dopamine and args.dopamine_state is None:
        parser.error("--freeze-dopamine requires --dopamine-state")
    if args.freeze_ppo and args.freeze_dopamine:
        parser.error("cannot freeze both PPO and dopamine; there would be no learning")
    run(args)


if __name__ == "__main__":
    main()
