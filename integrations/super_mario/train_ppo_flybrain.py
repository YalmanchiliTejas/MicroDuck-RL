"""Pretrain the MaleCNS Mario readout with discrete PPO and raw Gym rewards."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from male_cns import MaleCNS
from mario_dqn import ActivityStack, FlybrainAction
from mario_ppo import PPOAgent, PPOConfig, generalized_advantage_estimates
from mario_sidecar import nes_actions


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
    if snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot = snapshot_dir / f"flybrain-ppo-step-{agent.steps:09d}.pt"
        _atomic_agent_save(agent, snapshot)
    connectome.save_plasticity()


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


def run(args: argparse.Namespace) -> None:
    import gymnasium as gym
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros  # noqa: F401

    connectome = MaleCNS(
        data=args.male_cns_data,
        device=args.male_cns_device,
        seed=args.seed,
        spike_file=args.spike_file,
        dopamine_state=args.dopamine_state,
        dopamine_learning_rate=args.dopamine_learning_rate,
    )
    if args.resume is not None:
        agent = PPOAgent.load(args.resume, device=_device(args.device))
        print(f"resumed PPO={args.resume} step={agent.steps}", flush=True)
    else:
        agent = PPOAgent(
            PPOConfig(
                feature_dim=connectome.feature_dim,
                rollout_steps=args.rollout_steps,
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
    stack = ActivityStack(agent.config.stack_depth)
    state = stack.reset(connectome.reset(observation))
    writer = SummaryWriter(
        log_dir=str(args.tensorboard_dir),
        purge_step=agent.steps if agent.steps else None,
    )
    writer.add_text("training/reward_contract", "raw sum of env.step rewards", 0)
    writer.add_scalar("training/action_repeat", args.action_repeat, agent.steps)
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
    episode_return = 0.0
    episode_decisions = 0
    episode_max_x = 0
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
            terminated = truncated = False
            info: dict = {}
            for _ in range(args.action_repeat):
                observation, reward, terminated, truncated, info = env.step(action)
                raw_reward += float(reward)
                stack.append(connectome.observe(observation, action_sequence=agent.steps))
                if terminated or truncated:
                    break
            next_state = stack.state
            done = bool(terminated or truncated)
            rollout["states"].append(state.copy())
            rollout["actions"].append(action)
            rollout["log_probabilities"].append(log_probability)
            rollout["rewards"].append(raw_reward)
            rollout["dones"].append(done)
            rollout["values"].append(value)
            agent.steps += 1
            action_counts[action] += 1
            recent_actions.append(action)
            episode_return += raw_reward
            episode_decisions += 1
            episode_max_x = max(episode_max_x, int(info.get("x_pos", 0)))
            writer.add_scalar("reward/action_interval", raw_reward, agent.steps)
            writer.add_scalar("training/x_pos", int(info.get("x_pos", 0)), agent.steps)
            writer.add_scalar("training/action", action, agent.steps)
            connectome.reinforce(
                agent.prediction_error(raw_reward, next_state, done, value=value)
            )
            state = next_state

            if len(rollout["states"]) >= agent.config.rollout_steps or done:
                metrics = _flush(agent, rollout, state, done)
                if metrics is not None:
                    for name, metric in metrics.items():
                        writer.add_scalar(f"loss/{name}", metric, agent.steps)
                    writer.flush()

            if done:
                episode += 1
                recent_returns.append(episode_return)
                average = float(np.mean(recent_returns))
                writer.add_scalar("reward/episode_return", episode_return, agent.steps)
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
                        "dopamine/mean_kc_mbon_scale",
                        dopamine["mean_kc_mbon_scale"],
                        agent.steps,
                    )
                print(
                    f"episode={episode} step={agent.steps} decisions={episode_decisions} "
                    f"raw_return={episode_return:+.1f} avg100={average:+.1f} "
                    f"max_x={episode_max_x}",
                    flush=True,
                )
                observation, _ = env.reset()
                state = stack.reset(connectome.reset(observation))
                episode_return = 0.0
                episode_decisions = 0
                episode_max_x = 0

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
            for name, metric in metrics.items():
                writer.add_scalar(f"loss/{name}", metric, agent.steps)
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
    parser.add_argument("--action-repeat", type=int, default=30)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--output", type=Path, default=Path("flybrain-ppo.pt"))
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("tensorboard/ppo"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--male-cns-data", type=Path)
    parser.add_argument("--male-cns-device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--spike-file", type=Path)
    parser.add_argument("--dopamine-state", type=Path)
    parser.add_argument("--dopamine-learning-rate", type=float, default=0.001)
    parser.add_argument("--continuation-learning-rate", type=float)
    parser.add_argument("--value-coefficient", type=float)
    parser.add_argument("--entropy-coefficient", type=float)
    parser.add_argument("--target-kl", type=float)
    args = parser.parse_args()
    if min(args.steps, args.action_repeat, args.rollout_steps, args.save_every) <= 0:
        parser.error("steps, action repeat, rollout steps, and save interval must be positive")
    if args.additional_steps is not None and args.additional_steps <= 0:
        parser.error("--additional-steps must be positive")
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
    run(args)


if __name__ == "__main__":
    main()
