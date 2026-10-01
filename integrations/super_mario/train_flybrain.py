"""Train a DQN readout over the real MaleCNS connectome in Mario."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import time

import torch

from male_cns import MaleCNS
from mario_dqn import (
    ActivityStack,
    FlybrainAgent,
    FlybrainConfig,
    PrioritizedReplay,
)
from mario_sidecar import nes_actions
from rollouts import REWARD_COMPONENTS, training_reward


def _device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _save(
    agent: FlybrainAgent,
    connectome: MaleCNS,
    output: Path,
    *,
    replay: PrioritizedReplay | None = None,
    replay_state: Path | None = None,
) -> None:
    """Atomically save both learned layers so Slurm termination is recoverable."""

    temporary = output.with_name(f".{output.name}.tmp")
    agent.save(temporary)
    temporary.replace(output)
    connectome.save_plasticity()
    if replay is not None and replay_state is not None:
        replay.save(replay_state)


def run(args: argparse.Namespace) -> None:
    import gymnasium as gym
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros  # noqa: F401 -- registers environments

    connectome = MaleCNS(
        data=args.male_cns_data,
        device=args.male_cns_device,
        seed=args.seed,
        spike_file=args.spike_file,
        dopamine_state=args.dopamine_state,
        dopamine_learning_rate=args.dopamine_learning_rate,
    )
    if args.resume:
        agent = FlybrainAgent.load(args.resume, device=_device(args.device))
        config = agent.config
    else:
        config = FlybrainConfig(
            feature_dim=connectome.feature_dim,
            replay_capacity=args.replay_capacity,
            replay_start=args.replay_start,
        )
        agent = FlybrainAgent(config, device=_device(args.device), seed=args.seed)
    replay = PrioritizedReplay(
        config.replay_capacity,
        (config.stack_depth, config.feature_dim),
        alpha=config.per_alpha,
        seed=args.seed,
    )
    if args.replay_state is not None and args.replay_state.exists():
        replay.load(args.replay_state)
        print(
            f"restored replay={args.replay_state} transitions={len(replay)}",
            flush=True,
        )

    if config.num_actions != len(nes_actions()):
        raise ValueError(
            f"checkpoint has {config.num_actions} actions, but separate walk/run "
            f"training requires {len(nes_actions())}; "
            "start a fresh flybrain"
        )
    env = gym.make(args.env, render_mode="rgb_array")
    env = JoypadSpace(env, nes_actions())
    observation, _ = env.reset(seed=args.seed)
    stack = ActivityStack(config.stack_depth)
    state = stack.reset(connectome.reset(observation))
    episode_reward = 0.0
    episode = 0
    last_loss = float("nan")
    started = time.monotonic()
    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    try:
        first_step = agent.steps + 1
        for environment_step in range(first_step, args.steps + 1):
            action = agent.act(state)
            reward_sum = 0.0
            reward_components = {key: 0.0 for key in REWARD_COMPONENTS}
            terminated = truncated = False
            for _ in range(args.action_repeat):
                observation, reward, terminated, truncated, info = env.step(action)
                reward_sum += float(reward)
                for key, value in info.get("reward_components", {}).items():
                    if key in reward_components:
                        reward_components[key] += float(value)
                stack.append(connectome.observe(observation, action_sequence=environment_step))
                if terminated or truncated:
                    break
            next_state = stack.state
            done = terminated or truncated
            shaped_reward = training_reward(
                reward_components, raw_reward=reward_sum
            )
            prediction_error = agent.td_error(
                state, action, shaped_reward, next_state, done
            )
            connectome.reinforce(prediction_error)
            replay.add(state, action, shaped_reward, next_state, done)
            loss = agent.learn(replay)
            if loss is not None:
                last_loss = loss
            state = next_state
            episode_reward += reward_sum

            if done:
                episode += 1
                dopamine = connectome.dopamine_stats()
                dopamine_text = (
                    ""
                    if dopamine is None
                    else f" dopamine_rpe={dopamine['signal']:+.3f}"
                    f" kc_mbon={dopamine['mean_kc_mbon_scale']:.4f}"
                )
                print(
                    f"episode={episode} step={environment_step} reward={episode_reward:.1f} "
                    f"epsilon={agent.epsilon():.3f} loss={last_loss:.4f}"
                    f"{dopamine_text}",
                    flush=True,
                )
                observation, _ = env.reset()
                state = stack.reset(connectome.reset(observation))
                episode_reward = 0.0

            if environment_step % args.save_every == 0:
                _save(agent, connectome, args.output)
                rate = environment_step / max(time.monotonic() - started, 1.0e-6)
                print(
                    f"saved={args.output} step={environment_step} "
                    f"env_steps_per_s={rate:.1f}",
                    flush=True,
                )
            if (
                args.replay_state is not None
                and environment_step % args.replay_save_every == 0
            ):
                replay.save(args.replay_state)
                print(
                    f"saved_replay={args.replay_state} step={environment_step} "
                    f"transitions={len(replay)}",
                    flush=True,
                )
            if stop_requested:
                print(
                    f"received shutdown signal at step={environment_step}; "
                    "saving resumable state",
                    flush=True,
                )
                break
    finally:
        _save(
            agent,
            connectome,
            args.output,
            replay=replay,
            replay_state=args.replay_state,
        )
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a dueling DQN from MaleCNS descending-neuron traces"
    )
    parser.add_argument("--env", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--replay-capacity", type=int, default=20_000)
    parser.add_argument("--replay-start", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=25_000)
    parser.add_argument("--output", type=Path, default=Path("flybrain.pt"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--replay-state", type=Path)
    parser.add_argument("--replay-save-every", type=int, default=5_000)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--male-cns-data", type=Path)
    parser.add_argument(
        "--male-cns-device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--spike-file", type=Path)
    parser.add_argument("--dopamine-state", type=Path)
    parser.add_argument("--dopamine-learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    if args.steps <= 0 or args.action_repeat <= 0:
        parser.error("--steps and --action-repeat must be positive")
    if args.save_every <= 0 or args.replay_save_every <= 0:
        parser.error("--save-every and --replay-save-every must be positive")
    if args.dopamine_learning_rate <= 0:
        parser.error("--dopamine-learning-rate must be positive")
    if args.dopamine_state is not None and args.male_cns_device != "cpu":
        parser.error("--dopamine-state requires --male-cns-device cpu")
    run(args)


if __name__ == "__main__":
    main()
