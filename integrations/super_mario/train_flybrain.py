"""Train the pixel-based flybrain directly in the Mario emulator."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from flybrain import (
    FlybrainAgent,
    FlybrainConfig,
    FrameStack,
    PrioritizedReplay,
    preprocess_frame,
)
from mario_sidecar import nes_actions


def _device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run(args: argparse.Namespace) -> None:
    import gymnasium as gym
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros  # noqa: F401 -- registers environments

    if args.resume:
        agent = FlybrainAgent.load(args.resume, device=_device(args.device))
        config = agent.config
    else:
        config = FlybrainConfig(
            replay_capacity=args.replay_capacity,
            replay_start=args.replay_start,
        )
        agent = FlybrainAgent(config, device=_device(args.device), seed=args.seed)
    replay = PrioritizedReplay(
        config.replay_capacity,
        (config.stack_depth, config.frame_size, config.frame_size),
        alpha=config.per_alpha,
        seed=args.seed,
    )

    if config.num_actions != len(nes_actions()):
        raise ValueError(
            f"checkpoint has {config.num_actions} actions, but dynamic walk/run "
            f"training requires {len(nes_actions())}; start a fresh flybrain"
        )
    env = gym.make(args.env, render_mode="rgb_array")
    env = JoypadSpace(env, nes_actions())
    observation, _ = env.reset(seed=args.seed)
    stack = FrameStack(config.stack_depth)
    state = stack.reset(preprocess_frame(observation, config.frame_size))
    episode_reward = 0.0
    episode = 0
    last_loss = float("nan")
    started = time.monotonic()

    try:
        for environment_step in range(1, args.steps + 1):
            action = agent.act(state)
            reward_sum = 0.0
            terminated = truncated = False
            frames = []
            for _ in range(args.action_repeat):
                observation, reward, terminated, truncated, _ = env.step(action)
                reward_sum += float(reward)
                frames.append(observation)
                if terminated or truncated:
                    break
            # Max over the last two frames suppresses sprite flicker.
            visible = (
                np.maximum(frames[-1], frames[-2])
                if len(frames) > 1
                else frames[-1]
            )
            next_state = stack.append(preprocess_frame(visible, config.frame_size))
            done = terminated or truncated
            replay.add(state, action, np.sign(reward_sum), next_state, done)
            loss = agent.learn(replay)
            if loss is not None:
                last_loss = loss
            state = next_state
            episode_reward += reward_sum

            if done:
                episode += 1
                print(
                    f"episode={episode} step={environment_step} reward={episode_reward:.1f} "
                    f"epsilon={agent.epsilon():.3f} loss={last_loss:.4f}"
                )
                observation, _ = env.reset()
                state = stack.reset(preprocess_frame(observation, config.frame_size))
                episode_reward = 0.0

            if environment_step % args.save_every == 0:
                agent.save(args.output)
                rate = environment_step / max(time.monotonic() - started, 1.0e-6)
                print(
                    f"saved={args.output} step={environment_step} "
                    f"env_steps_per_s={rate:.1f}"
                )
        agent.save(args.output)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Double/Dueling DQN flybrain from stacked Mario frames"
    )
    parser.add_argument("--env", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--replay-capacity", type=int, default=20_000)
    parser.add_argument("--replay-start", type=int, default=2_000)
    parser.add_argument("--save-every", type=int, default=25_000)
    parser.add_argument("--output", type=Path, default=Path("flybrain.pt"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.steps <= 0 or args.action_repeat <= 0:
        parser.error("--steps and --action-repeat must be positive")
    if args.save_every <= 0:
        parser.error("--save-every must be positive")
    run(args)


if __name__ == "__main__":
    main()
