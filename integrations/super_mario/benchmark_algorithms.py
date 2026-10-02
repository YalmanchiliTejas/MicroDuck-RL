"""Train and evaluate DQN, Double DQN, or PPO on identical MaleCNS inputs."""

from __future__ import annotations

import argparse
from collections import Counter, deque
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from male_cns import MaleCNS
from mario_dqn import ActivityStack, FlybrainAgent, FlybrainConfig, PrioritizedReplay
from mario_ppo import PPOAgent, PPOConfig, generalized_advantage_estimates
from mario_sidecar import nes_actions


ALGORITHMS = ("dqn", "double_dqn", "ppo")


def _device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _make_env(name: str):
    import gymnasium as gym
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros  # noqa: F401 -- registers environments

    return JoypadSpace(gym.make(name, render_mode="rgb_array"), nes_actions())


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_progress(path: Path) -> tuple[int, deque[float]]:
    if not path.exists():
        return 0, deque(maxlen=100)
    payload = json.loads(path.read_text())
    return int(payload.get("episodes", 0)), deque(
        (float(value) for value in payload.get("recent_returns", [])), maxlen=100
    )


def _save_progress(path: Path, episodes: int, recent_returns: deque[float]) -> None:
    _atomic_json(
        path,
        {"episodes": episodes, "recent_returns": list(recent_returns)},
    )


def _collect_interval(
    env,
    connectome: MaleCNS,
    stack: ActivityStack,
    action: int,
    action_repeat: int,
    action_sequence: int,
) -> tuple[np.ndarray, float, bool, bool, dict]:
    raw_reward = 0.0
    terminated = truncated = False
    last_info: dict = {}
    for _ in range(action_repeat):
        observation, reward, terminated, truncated, last_info = env.step(action)
        raw_reward += float(reward)
        stack.append(
            connectome.observe(observation, action_sequence=action_sequence)
        )
        if terminated or truncated:
            break
    return stack.state, raw_reward, terminated, truncated, last_info


def _save_q(
    agent: FlybrainAgent,
    replay: PrioritizedReplay,
    checkpoint: Path,
    replay_path: Path,
) -> None:
    temporary = checkpoint.with_name(f".{checkpoint.name}.tmp")
    agent.save(temporary)
    temporary.replace(checkpoint)
    replay.save(replay_path)


def _save_ppo(agent: PPOAgent, checkpoint: Path) -> None:
    temporary = checkpoint.with_name(f".{checkpoint.name}.tmp")
    agent.save(temporary)
    temporary.replace(checkpoint)


def _flush_ppo(
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
        old_log_probabilities=np.asarray(
            rollout["log_probabilities"], dtype=np.float32
        ),
        returns=returns,
        advantages=advantages,
    )
    for values in rollout.values():
        values.clear()
    return metrics


def train(args: argparse.Namespace) -> tuple[object, int]:
    """Train one algorithm without dopamine plasticity or reward shaping."""

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"{args.algorithm}.pt"
    replay_path = output_dir / f"{args.algorithm}-replay.npz"
    progress_path = output_dir / f"{args.algorithm}-progress.json"
    tensorboard_dir = args.tensorboard_dir / args.algorithm / f"seed_{args.seed}"
    connectome = MaleCNS(
        data=args.male_cns_data,
        device=args.male_cns_device,
        seed=args.seed,
    )
    device = _device(args.device)

    replay = None
    if args.algorithm in {"dqn", "double_dqn"}:
        if checkpoint.exists():
            agent: FlybrainAgent | PPOAgent = FlybrainAgent.load(
                checkpoint, device=device
            )
            if agent.config.algorithm != args.algorithm:
                raise ValueError("Q checkpoint algorithm does not match request")
        else:
            agent = FlybrainAgent(
                FlybrainConfig(
                    feature_dim=connectome.feature_dim,
                    replay_capacity=args.replay_capacity,
                    replay_start=args.replay_start,
                    algorithm=args.algorithm,
                ),
                device=device,
                seed=args.seed,
            )
        replay = PrioritizedReplay(
            agent.config.replay_capacity,
            (agent.config.stack_depth, agent.config.feature_dim),
            alpha=agent.config.per_alpha,
            seed=args.seed,
        )
        if replay_path.exists():
            replay.load(replay_path)
    else:
        if checkpoint.exists():
            agent = PPOAgent.load(checkpoint, device=device)
        else:
            agent = PPOAgent(
                PPOConfig(
                    feature_dim=connectome.feature_dim,
                    rollout_steps=args.ppo_rollout_steps,
                ),
                device=device,
                seed=args.seed,
            )

    if agent.steps >= args.steps:
        print(
            f"algorithm={args.algorithm} already_complete step={agent.steps}",
            flush=True,
        )
        return agent, agent.steps

    episode_count, recent_returns = _load_progress(progress_path)
    writer = SummaryWriter(
        log_dir=str(tensorboard_dir),
        purge_step=agent.steps if agent.steps else None,
    )
    writer.add_text("benchmark/reward_contract", "raw sum of env.step rewards", 0)
    writer.add_text("benchmark/algorithm", args.algorithm, 0)
    writer.add_scalar("benchmark/action_repeat", args.action_repeat, 0)
    env = _make_env(args.env)
    observation, _ = env.reset(seed=args.seed + agent.steps)
    stack = ActivityStack(4)
    state = stack.reset(connectome.reset(observation))
    episode_return = 0.0
    episode_length = 0
    episode_max_x = 0
    action_counts: Counter[int] = Counter()
    stop_requested = False
    starting_step = agent.steps
    last_save = agent.steps
    started = time.monotonic()
    rollout = {
        "states": [],
        "actions": [],
        "log_probabilities": [],
        "rewards": [],
        "dones": [],
        "values": [],
    }

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    try:
        while agent.steps < args.steps and not stop_requested:
            if args.algorithm == "ppo":
                action, log_probability, value = agent.act(state)
            else:
                action = agent.act(state)
            next_state, reward, terminated, truncated, info = _collect_interval(
                env,
                connectome,
                stack,
                action,
                args.action_repeat,
                agent.steps,
            )
            done = terminated or truncated
            action_counts[action] += 1
            episode_return += reward
            episode_length += 1
            episode_max_x = max(episode_max_x, int(info.get("x_pos", 0)))

            if args.algorithm == "ppo":
                rollout["states"].append(state.copy())
                rollout["actions"].append(action)
                rollout["log_probabilities"].append(log_probability)
                rollout["rewards"].append(reward)
                rollout["dones"].append(done)
                rollout["values"].append(value)
                agent.steps += 1
                if len(rollout["states"]) >= agent.config.rollout_steps:
                    metrics = _flush_ppo(agent, rollout, next_state, done)
                    assert metrics is not None
                    for name, metric in metrics.items():
                        writer.add_scalar(f"loss/{name}", metric, agent.steps)
            else:
                assert replay is not None
                replay.add(state, action, reward, next_state, done)
                loss = agent.learn(replay)
                if loss is not None:
                    writer.add_scalar("loss/q", loss, agent.steps)
                writer.add_scalar("exploration/epsilon", agent.epsilon(), agent.steps)
            state = next_state

            if done:
                episode_count += 1
                recent_returns.append(episode_return)
                average_return = float(np.mean(recent_returns))
                writer.add_scalar("reward/episode_return", episode_return, agent.steps)
                writer.add_scalar(
                    "reward/average_100_episodes", average_return, agent.steps
                )
                writer.add_scalar(
                    "episode/length_decisions", episode_length, agent.steps
                )
                writer.add_scalar("episode/max_x", episode_max_x, agent.steps)
                print(
                    f"algorithm={args.algorithm} episode={episode_count} "
                    f"step={agent.steps} decisions={episode_length} "
                    f"raw_return={episode_return:+.1f} avg100={average_return:+.1f} "
                    f"max_x={episode_max_x}",
                    flush=True,
                )
                observation, _ = env.reset()
                state = stack.reset(connectome.reset(observation))
                episode_return = 0.0
                episode_length = 0
                episode_max_x = 0

            if agent.steps - last_save >= args.save_every:
                if args.algorithm == "ppo":
                    _save_ppo(agent, checkpoint)
                else:
                    assert replay is not None
                    _save_q(agent, replay, checkpoint, replay_path)
                _save_progress(progress_path, episode_count, recent_returns)
                writer.flush()
                last_save = agent.steps
                rate = (agent.steps - starting_step) / max(
                    time.monotonic() - started, 1.0e-6
                )
                print(
                    f"saved={checkpoint} step={agent.steps} decisions_per_s={rate:.2f}",
                    flush=True,
                )
    finally:
        if args.algorithm == "ppo":
            metrics = _flush_ppo(agent, rollout, state, False)
            if metrics is not None:
                for name, metric in metrics.items():
                    writer.add_scalar(f"loss/{name}", metric, agent.steps)
            _save_ppo(agent, checkpoint)
        else:
            assert replay is not None
            _save_q(agent, replay, checkpoint, replay_path)
        for action, count in action_counts.items():
            writer.add_scalar(f"actions/count_{action}", count, agent.steps)
        _save_progress(progress_path, episode_count, recent_returns)
        writer.flush()
        writer.close()
        env.close()
    return agent, agent.steps


def evaluate(args: argparse.Namespace, agent: object, step: int) -> dict:
    """Run a frozen deterministic evaluation and add its aggregate to TensorBoard."""

    connectome = MaleCNS(
        data=args.male_cns_data,
        device=args.male_cns_device,
        seed=args.seed,
    )
    env = _make_env(args.env)
    returns = []
    max_positions = []
    completions = 0
    deaths = 0
    try:
        for episode in range(args.eval_episodes):
            observation, _ = env.reset(seed=args.seed + 10_000 + episode)
            stack = ActivityStack(4)
            state = stack.reset(connectome.reset(observation))
            episode_return = 0.0
            max_x = 0
            terminated = truncated = False
            last_info: dict = {}
            for _ in range(args.eval_max_decisions):
                if args.algorithm == "ppo":
                    action, _, _ = agent.act(state, deterministic=True)
                else:
                    action = agent.act(state, epsilon=0.0)
                state, reward, terminated, truncated, last_info = _collect_interval(
                    env,
                    connectome,
                    stack,
                    action,
                    args.action_repeat,
                    step,
                )
                episode_return += reward
                max_x = max(max_x, int(last_info.get("x_pos", 0)))
                if terminated or truncated:
                    break
            returns.append(episode_return)
            max_positions.append(max_x)
            reward_components = last_info.get("reward_components", {})
            completed = bool(
                last_info.get("flag_get", False)
                or last_info.get("clear", False)
                or reward_components.get("completion", 0.0) > 0.0
            )
            died = bool(
                last_info.get("death", False)
                or last_info.get("is_dead", False)
                or last_info.get("is_dying", False)
                or reward_components.get("death", 0.0) < 0.0
            )
            completions += int(completed)
            deaths += int(died)
    finally:
        env.close()

    report = {
        "algorithm": args.algorithm,
        "seed": args.seed,
        "training_steps": step,
        "reward_contract": "raw sum of env.step rewards",
        "episodes": args.eval_episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_max_x": float(np.mean(max_positions)),
        "completion_rate": completions / args.eval_episodes,
        "death_rate": deaths / args.eval_episodes,
        "returns": returns,
        "max_x": max_positions,
    }
    report_path = args.output_dir / f"{args.algorithm}-evaluation.json"
    _atomic_json(report_path, report)
    writer = SummaryWriter(
        log_dir=str(args.tensorboard_dir / args.algorithm / f"seed_{args.seed}")
    )
    writer.add_scalar("evaluation/mean_episode_return", report["mean_return"], step)
    writer.add_scalar("evaluation/mean_max_x", report["mean_max_x"], step)
    writer.add_scalar("evaluation/completion_rate", report["completion_rate"], step)
    writer.add_scalar("evaluation/death_rate", report["death_rate"], step)
    writer.add_histogram("evaluation/episode_returns", np.asarray(returns), step)
    writer.close()
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--env", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--action-repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--replay-capacity", type=int, default=20_000)
    parser.add_argument("--replay-start", type=int, default=500)
    parser.add_argument("--ppo-rollout-steps", type=int, default=256)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-max-decisions", type=int, default=300)
    parser.add_argument("--male-cns-data", type=Path)
    parser.add_argument(
        "--male-cns-device", choices=("auto", "cpu", "cuda"), default="cpu"
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    args = parser.parse_args()
    if min(
        args.steps,
        args.action_repeat,
        args.save_every,
        args.eval_episodes,
        args.eval_max_decisions,
    ) <= 0:
        parser.error("step, repeat, save, and evaluation counts must be positive")
    agent, step = train(args)
    if step >= args.steps:
        evaluate(args, agent, step)


if __name__ == "__main__":
    main()
