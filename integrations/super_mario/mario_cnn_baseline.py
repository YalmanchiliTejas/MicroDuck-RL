"""Independent SB3 CNN-PPO control; matched NES buttons, reward and episode limit.

Recipe: https://github.com/imenei/mario-ppo-agent (CnnPolicy + four-frame stack).
This is an adaptation, NOT a reproduction of its reported completion rate:
we retain this project's ten buttons, reward_contract and episode limit.
"""
from __future__ import annotations

import argparse
from collections import deque
from importlib.metadata import version
import json
from pathlib import Path
import shutil

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from mario_sidecar import nes_actions, action_name
from reward_contract import REWARD_CONTRACT
from rollouts import REWARD_COMPONENTS, training_reward

CONTRACT = "direct-buttons-4frames-pixel-stack-v1"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


class MarioPixels(gym.Wrapper):
    """Repeat actual buttons four frames; never insert releases or auto-reset.

    Max-pool the last two RGB frames, grayscale/resize, stack four observations.
    Reward is computed ONCE per action interval by the SAME function as MaleCNS.
    Frame records are diagnostic only and are never policy observations.
    """
    def __init__(self, env, max_decisions=3000):
        super().__init__(env)
        self.max_decisions = max_decisions
        self.observation_space = gym.spaces.Box(0, 255, (4, 84, 84), np.uint8)
        self.history = deque(maxlen=4)
        self.total_emulator_frames = 0
        self.capture = False
        self.frame_records = []

    @staticmethod
    def pixels(rgb):
        return np.asarray(Image.fromarray(rgb).convert("L").resize(
            (84, 84), Image.Resampling.BILINEAR), dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.history.clear()
        self.history.extend([self.pixels(obs)] * 4)
        self.decisions = self.jump_hold = self.action_hold = 0
        self.previous_action = None
        self.previous_jump = False
        self.frame_records = []
        return np.stack(self.history), info

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError("invalid NES action")
        frames, components, raw = [], {}, 0.0
        self.frame_records = []
        presses = releases = max_hold = max_action_hold = 0
        completed = died = False
        max_x = 0
        for index in range(4):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.total_emulator_frames += 1
            frames.append(obs.copy())
            raw += float(reward)
            for key, value in info.get("reward_components", {}).items():
                if key in REWARD_COMPONENTS:
                    components[key] = components.get(key, 0.0) + float(value)
            completed |= bool(info.get("clear", False) or info.get("flag_get", False))
            died |= bool(info.get("death", False) or info.get("is_dead", False)
                         or info.get("is_dying", False))
            max_x = max(max_x, int(info.get("x_pos", 0)))
            jump = "A" in nes_actions()[action]
            pressed, released = jump and not self.previous_jump, self.previous_jump and not jump
            presses += int(pressed)
            releases += int(released)
            self.jump_hold = self.jump_hold + 1 if jump else 0
            self.action_hold = self.action_hold + 1 if action == self.previous_action else 1
            max_hold = max(max_hold, self.jump_hold)
            max_action_hold = max(max_action_hold, self.action_hold)
            self.previous_jump, self.previous_action = jump, action
            if self.capture:
                self.frame_records.append((obs.copy(), dict(
                    decision=self.decisions + 1, frame_in_decision=index + 1,
                    executed_action=action_name(action), x=int(info.get("x_pos", 0)),
                    y=int(info.get("y_pos", 0)), jump_pressed_edge=bool(pressed),
                    jump_released_edge=bool(released), jump_hold_frames=self.jump_hold,
                    terminated=bool(terminated), truncated=bool(truncated))))
            if terminated or truncated:
                break
        self.decisions += 1
        self.history.append(self.pixels(np.maximum(frames[-1], frames[-2] if len(frames) > 1 else frames[-1])))
        limited = self.decisions >= self.max_decisions and not (terminated or truncated)
        result = dict(info)
        result.update(raw_reward=raw, reward_components=components, interval_max_x=max_x,
                      completion=completed or components.get("completion", 0) > 0,
                      death=died or components.get("death", 0) < 0,
                      emulator_frames=len(frames), jump_press_edges=presses,
                      jump_release_edges=releases, max_jump_hold_frames=max_hold,
                      max_action_hold_frames=max_action_hold, evaluator_truncated=limited)
        return (np.stack(self.history), training_reward(components, raw_reward=raw),
                bool(terminated), bool(truncated or limited), result)


def make_env(env_id, max_decisions):
    import gym_super_mario_bros  # noqa: F401: register environments
    from nes_py.wrappers import JoypadSpace
    return MarioPixels(JoypadSpace(gym.make(env_id, render_mode="rgb_array"), nes_actions()), max_decisions)


def contract(args):
    return dict(contract=CONTRACT, env=args.env, actions=nes_actions(),
                reward_contract=REWARD_CONTRACT, frames_per_decision=4,
                max_decisions=args.max_decisions, n_envs=args.n_envs,
                seed=args.seed, rollout_steps_per_env=1024 // args.n_envs,
                stable_baselines3=version("stable-baselines3"),
                gym_super_mario_bros=version("gym-super-mario-bros"))


class CheckpointPPO(PPO):
    """Snapshot only AFTER updates, so snapshots include the latest learning."""
    def train(self):
        super().train()
        if self.num_timesteps - self.last_saved >= self.save_every:
            self.save(self.snapshot_dir / f"cnn-ppo-step-{self.num_timesteps:09d}.zip")
            self.last_saved = self.num_timesteps


def train(args):
    run = args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    (run / "checkpoints").mkdir(exist_ok=True)
    config_path, latest = run / "cnn-config.json", run / "cnn-ppo.zip"
    config = contract(args)
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("CNN contract changed; choose a fresh run directory")
    write_json(config_path, config)
    if latest.exists() and not args.resume:
        raise ValueError("checkpoint exists; use --resume or a fresh run")
    if args.resume and not latest.exists():
        raise ValueError("--resume requires cnn-ppo.zip")
    env = DummyVecEnv([lambda: Monitor(make_env(args.env, args.max_decisions)) for _ in range(args.n_envs)])
    try:
        if args.resume:
            model = CheckpointPPO.load(latest, env=env, device=args.device)
        else:
            model = CheckpointPPO("CnnPolicy", env, learning_rate=2.5e-4,
                n_steps=1024 // args.n_envs, batch_size=64, n_epochs=4,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                vf_coef=0.5, max_grad_norm=0.5, target_kl=0.02,
                tensorboard_log=str(run / "tensorboard"), seed=args.seed,
                device=args.device, verbose=1)
        model.snapshot_dir = run / "checkpoints"
        model.save_every = args.save_every
        model.last_saved = model.num_timesteps
        remaining = args.steps - model.num_timesteps
        if remaining <= 0:
            raise ValueError("target must exceed saved decision count")
        model.learn(remaining, reset_num_timesteps=False)
        model.save(latest)
        model.save(model.snapshot_dir / f"cnn-ppo-step-{model.num_timesteps:09d}.zip")
        metrics = {k: float(v) for k, v in model.logger.name_to_value.items()
                   if isinstance(v, (int, float, np.number))}
        if not metrics or not all(np.isfinite(v) for v in metrics.values()):
            raise RuntimeError("missing/nonfinite CNN training metrics")
        write_json(run / "cnn-training.json", dict(requested_decisions=args.steps,
            actual_decisions=model.num_timesteps, metrics=metrics,
            note="SB3 rounds up to complete 1024-transition rollouts; steps count all environments."))
    finally:
        env.close()


def evaluate(checkpoint, output, *, seed, episodes, record_episodes, device):
    config = json.loads((checkpoint.parent.parent / "cnn-config.json"
                        if checkpoint.parent.name == "checkpoints" else checkpoint.parent / "cnn-config.json").read_text())
    if config["contract"] != CONTRACT or config["reward_contract"] != REWARD_CONTRACT:
        raise ValueError("unsupported CNN contract")
    model = PPO.load(checkpoint, device=device)
    output.mkdir(parents=True, exist_ok=False)
    env = make_env(config["env"], config["max_decisions"])
    reports = []
    try:
        for mode in ("mean", "sampled"):
            model.set_random_seed(seed)
            rows = []
            for episode in range(episodes):
                observation, _ = env.reset(seed=seed + episode)
                env.capture = episode < record_episodes
                directory = output / f"{mode}-episode-{episode:03d}"
                if env.capture:
                    directory.mkdir()
                images, durations, traces = [], [], []
                raw = shaped = 0.0
                max_x = presses = releases = hold = action_hold = decisions = 0
                death = completion = False
                for decision in range(config["max_decisions"]):
                    action, _ = model.predict(observation, deterministic=mode == "mean")
                    if env.capture:
                        import torch
                        with torch.no_grad():
                            tensor, _ = model.policy.obs_to_tensor(observation)
                            probabilities = model.policy.get_distribution(tensor).distribution.probs[0].cpu().tolist()
                    observation, reward, terminated, truncated, info = env.step(action)
                    raw += info["raw_reward"]
                    shaped += reward
                    max_x = max(max_x, info["interval_max_x"])
                    presses += info["jump_press_edges"]
                    releases += info["jump_release_edges"]
                    hold = max(hold, info["max_jump_hold_frames"])
                    action_hold = max(action_hold, info["max_action_hold_frames"])
                    death |= info["death"]
                    completion |= info["completion"]
                    decisions = decision + 1
                    for rgb, row in env.frame_records:
                        row["action_probabilities"] = dict(zip(map(action_name, range(10)), probabilities))
                        traces.append(row)
                        if row["frame_in_decision"] % 2 == 0 or row["terminated"] or row["truncated"]:
                            image = Image.new("RGB", (max(360, rgb.shape[1]), rgb.shape[0] + 28))
                            image.paste(Image.fromarray(rgb), (0, 0))
                            ImageDraw.Draw(image).text((3, rgb.shape[0] + 3),
                                f"{mode} x={row['x']} {row['executed_action']} hold={row['jump_hold_frames']}", fill="white")
                            images.append(image)
                            durations.append(40 if len(images) % 3 == 0 else 30)
                    if terminated or truncated:
                        break
                if env.capture:
                    with (directory / "frames.jsonl").open("w") as stream:
                        for row in traces:
                            stream.write(json.dumps(row) + "\n")
                    if images:
                        images[0].save(directory / "rollout.gif", save_all=True,
                            append_images=images[1:], duration=durations, loop=0)
                rows.append(dict(episode=episode, completion=completion, death=death,
                    max_x=max_x, raw_reward=raw, environment_training_reward=shaped,
                    decisions=decisions, jump_press_edges=presses, jump_release_edges=releases,
                    max_jump_hold_decisions=hold, max_action_hold_decisions=action_hold,
                    environment_terminated=terminated, environment_truncated=truncated,
                    evaluator_truncated=info["evaluator_truncated"]))
            summary = {"episode_count": len(rows)}
            for name, field in {"completion_rate": "completion", "death_rate": "death",
                "mean_max_x": "max_x", "mean_raw_reward": "raw_reward",
                "mean_environment_training_reward": "environment_training_reward",
                "mean_decisions": "decisions", "mean_jump_press_edges": "jump_press_edges",
                "mean_jump_release_edges": "jump_release_edges",
                "mean_max_jump_hold_decisions": "max_jump_hold_decisions"}.items():
                summary[name] = float(np.mean([row[field] for row in rows]))
            report = dict(mode=mode, checkpoint=str(checkpoint.resolve()), seed=seed,
                visual_encoder="sb3_nature_cnn", temporal_encoder="four_frame_stack",
                learning_enabled=False, visual_ablation="normal", direct_four_frame=True,
                bounded_jump_actions=False, hold_counter_units="frames", frames_per_decision=4,
                training_decisions=int(model.num_timesteps), reward_contract=REWARD_CONTRACT,
                environment=config["env"], actions=config["actions"], episodes=rows, summary=summary)
            write_json(output / f"frozen-{mode}.json", report)
            reports.append(report)
    finally:
        env.close()
    return reports


def rank(args):
    paths = sorted((args.run_dir / "checkpoints").glob("cnn-ppo-step-*.zip"))
    if not paths:
        raise ValueError("no CNN snapshots")
    if len(paths) > args.max_checkpoints:
        paths = [paths[i] for i in np.linspace(0, len(paths) - 1, args.max_checkpoints, dtype=int)]
    rows = []
    for path in paths:
        reports = evaluate(path, args.output / path.stem, seed=args.seed,
            episodes=args.episodes, record_episodes=0, device=args.device)
        mean, sampled = [r["summary"] for r in reports]
        # Same completion-first ordering as rank_ppo_checkpoints.py.
        key = ((mean["completion_rate"] + sampled["completion_rate"]) / 2,
            mean["completion_rate"], min(mean["mean_max_x"], sampled["mean_max_x"]),
            (mean["mean_max_x"] + sampled["mean_max_x"]) / 2,
            (mean["mean_raw_reward"] + sampled["mean_raw_reward"]) / 2,
            reports[0]["training_decisions"])
        rows.append(dict(checkpoint=str(path), key=key, modes={r["mode"]: r["summary"] for r in reports}))
    best = max(rows, key=lambda row: row["key"])
    shutil.copy2(best["checkpoint"], args.run_dir / "best-cnn-ppo.zip")
    write_json(args.output / "ranking.json", dict(recommended_checkpoint=best["checkpoint"], rows=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate", "rank"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--env", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--max-decisions", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--record-episodes", type=int, default=3)
    parser.add_argument("--max-checkpoints", type=int, default=6)
    args = parser.parse_args()
    if args.n_envs <= 0 or args.n_envs > 512 or 1024 % args.n_envs:
        parser.error("n-envs must divide 1024 and be at most 512")
    if min(args.steps, args.max_decisions, args.save_every, args.episodes, args.max_checkpoints) <= 0:
        parser.error("counts must be positive")
    if args.record_episodes < 0:
        parser.error("record-episodes cannot be negative")
    if args.command == "train":
        train(args)
    elif args.output is None:
        parser.error("evaluation/ranking requires a new --output directory")
    elif args.command == "rank":
        rank(args)
    else:
        evaluate(args.checkpoint or args.run_dir / "best-cnn-ppo.zip", args.output,
            seed=args.seed, episodes=args.episodes, record_episodes=args.record_episodes, device=args.device)


if __name__ == "__main__":
    main()
