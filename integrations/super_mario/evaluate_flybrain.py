"""Evaluate a frozen MaleCNS Mario policy without any learning."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from male_cns import MaleCNS
from mario_dqn import ActivityStack, FlybrainAction, FlybrainAgent
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


def _legacy_training_reward(components: dict[str, float], raw_reward: float) -> float:
    """Reproduce the reward used by the compromised schema-4 run."""

    if not components:
        return float(np.clip(raw_reward / 10.0, -5.0, 5.0))
    value = (
        components.get("progress", 0.0) / 10.0
        + components.get("time", 0.0) / 20.0
        + components.get("score", 0.0) / 5.0
        + components.get("coins", 0.0) / 5.0
        + components.get("powerup", 0.0) / 5.0
        + components.get("completion", 0.0) / 10.0
        + components.get("death", 0.0) / 10.0
    )
    return float(np.clip(value, -5.0, 5.0))


def _atomic_json(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    temporary.replace(path)


class ForensicRecorder:
    """Record one emulator episode as synchronized GIF and JSONL traces."""

    def __init__(self, directory: Path, episode: int, fps: int) -> None:
        self.directory = directory
        self.episode = episode
        self.fps = fps
        self.frames = []
        self.frame_rows: list[dict] = []
        self.decision_rows: list[dict] = []

    def add_frame(self, frame: np.ndarray, row: dict, overlay: list[str]) -> None:
        from PIL import Image, ImageDraw

        image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(image)
        line_height = 11
        overlay_height = line_height * len(overlay) + 4
        draw.rectangle((0, 0, image.width, overlay_height), fill=(0, 0, 0))
        for line, text in enumerate(overlay):
            draw.text((3, 2 + line * line_height), text, fill=(255, 255, 255))
        self.frames.append(image)
        self.frame_rows.append(row)

    def add_decision(self, row: dict) -> None:
        self.decision_rows.append(row)

    def close(self) -> dict[str, str]:
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = f"episode-{self.episode:03d}"
        frame_trace = self.directory / f"{stem}-frames.jsonl"
        decision_trace = self.directory / f"{stem}-decisions.jsonl"
        animation = self.directory / f"{stem}.gif"
        _atomic_jsonl(frame_trace, self.frame_rows)
        _atomic_jsonl(decision_trace, self.decision_rows)
        if self.frames:
            temporary = animation.with_name(f".{animation.name}.tmp.gif")
            self.frames[0].save(
                temporary,
                save_all=True,
                append_images=self.frames[1:],
                duration=max(1, round(1000 / self.fps)),
                loop=0,
                optimize=False,
            )
            temporary.replace(animation)
        return {
            "video": str(animation.resolve()),
            "frame_trace": str(frame_trace.resolve()),
            "decision_trace": str(decision_trace.resolve()),
        }


def run(args: argparse.Namespace) -> dict:
    import gym_super_mario_bros  # noqa: F401 -- registers environments
    import gymnasium as gym
    from nes_py.wrappers import JoypadSpace

    device = _device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    algorithm = str(payload.get("algorithm", "dqn"))
    if algorithm == "ppo":
        from mario_ppo import PPOAgent

        agent = PPOAgent.load(args.checkpoint, device=device)
        agent.network.eval()
    else:
        agent = FlybrainAgent.load(
            args.checkpoint,
            device=device,
            allow_legacy_reward_contract=args.allow_legacy_reward_contract,
        )
        agent.online.eval()
    connectome = MaleCNS(
        data=args.male_cns_data,
        device=args.male_cns_device,
        seed=args.seed,
        dopamine_state=args.dopamine_state,
        dopamine_learning_rate=args.dopamine_learning_rate,
    )
    if connectome.feature_dim != agent.config.feature_dim:
        raise ValueError(
            f"checkpoint expects {agent.config.feature_dim} features, "
            f"MaleCNS provides {connectome.feature_dim}"
        )

    action_names = [action.name.lower() for action in FlybrainAction]
    # mario_sidecar.nes_actions and the checkpoint share the FlybrainAction order.
    if len(action_names) != agent.config.num_actions:
        raise ValueError("checkpoint action count does not match the Mario action space")

    env = JoypadSpace(gym.make(args.env, render_mode="rgb_array"), nes_actions())
    action_counts: Counter[int] = Counter()
    greedy_counts: Counter[int] = Counter()
    initial_action_counts: Counter[int] = Counter()
    fatal_terminal_action_counts: Counter[int] = Counter()
    q_sums = np.zeros(agent.config.num_actions, dtype=np.float64)
    q_observations = 0
    q_margin_sum = 0.0
    state_rms_sum = 0.0
    state_delta_rms_sum = 0.0
    episodes: list[dict] = []
    first_actions: list[str] = []
    forensic_artifacts: list[dict] = []
    record_dir = getattr(args, "record_dir", None)
    record_episodes = int(getattr(args, "record_episodes", 0))
    record_fps = int(getattr(args, "record_fps", 60))

    try:
        for episode_index in range(args.episodes):
            observation, reset_info = env.reset(seed=args.seed + episode_index)
            stack = ActivityStack(agent.config.stack_depth)
            state = stack.reset(connectome.reset(observation))
            current_x = int(reset_info.get("x_pos", 0))
            components = {key: 0.0 for key in REWARD_COMPONENTS}
            raw_total = environment_training_total = legacy_total = 0.0
            max_x = 0
            terminated = truncated = False
            death = completion = False
            initial_action: int | None = None
            terminal_action: int | None = None
            recorder = (
                ForensicRecorder(record_dir, episode_index, record_fps)
                if record_dir is not None and episode_index < record_episodes
                else None
            )

            for decision in range(1, args.max_decisions_per_episode + 1):
                policy_diagnostics: dict[str, object] = {}
                with torch.no_grad():
                    state_t = torch.as_tensor(state, device=agent.device).unsqueeze(0)
                    if algorithm == "ppo":
                        policy_diagnostics = agent.policy_diagnostics(state)
                        action_scores = policy_diagnostics["action_probabilities"]
                    else:
                        action_scores = agent.online(state_t)[0].detach().cpu().numpy()
                greedy_action = int(np.argmax(action_scores))
                if algorithm == "ppo":
                    action, log_probability, value = agent.act(
                        state, deterministic=not args.sample_actions
                    )
                else:
                    action = agent.act(state, epsilon=args.epsilon)
                    log_probability = value = None
                q_sums += action_scores
                q_margin_sum += float(
                    np.partition(action_scores, -2)[-1]
                    - np.partition(action_scores, -2)[-2]
                )
                state_rms_sum += float(np.sqrt(np.mean(np.square(state))))
                q_observations += 1
                greedy_counts[greedy_action] += 1
                action_counts[action] += 1
                if initial_action is None:
                    initial_action = action
                    initial_action_counts[action] += 1
                terminal_action = action
                if len(first_actions) < args.first_actions:
                    first_actions.append(action_names[action])

                interval_raw = 0.0
                interval_components: dict[str, float] = {}
                x_before = current_x
                state_rms = float(np.sqrt(np.mean(np.square(state))))
                for frame_in_decision in range(1, args.action_repeat + 1):
                    observation, reward, terminated, truncated, info = env.step(action)
                    interval_raw += float(reward)
                    current_x = int(info.get("x_pos", current_x))
                    max_x = max(max_x, current_x)
                    for key, component_value in info.get(
                        "reward_components", {}
                    ).items():
                        if key in REWARD_COMPONENTS:
                            interval_components[key] = (
                                interval_components.get(key, 0.0)
                                + float(component_value)
                            )
                    stack.append(
                        connectome.observe(
                            observation,
                            action_sequence=q_observations - 1,
                        )
                    )
                    death |= bool(
                        info.get("death", False)
                        or info.get("is_dead", False)
                        or info.get("is_dying", False)
                    )
                    completion |= bool(
                        info.get("clear", False) or info.get("flag_get", False)
                    )
                    if recorder is not None:
                        frame_row = {
                            "episode": episode_index,
                            "decision": decision,
                            "frame_in_decision": frame_in_decision,
                            "action": action_names[action],
                            "x": current_x,
                            "y": int(info.get("y_pos", 0)),
                            "reward": float(reward),
                            "interval_reward": interval_raw,
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            "death": bool(death),
                            "completion": bool(completion),
                        }
                        recorder.add_frame(
                            observation,
                            frame_row,
                            [
                                (
                                    f"episode={episode_index} decision={decision} "
                                    f"frame={frame_in_decision}/{args.action_repeat}"
                                ),
                                (
                                    f"action={action_names[action]} x={current_x} "
                                    f"y={frame_row['y']} reward={interval_raw:+.1f}"
                                ),
                            ],
                        )
                    if terminated or truncated:
                        break

                for key, component_value in interval_components.items():
                    components[key] += component_value
                raw_total += interval_raw
                environment_training_total += training_reward(
                    interval_components, raw_reward=interval_raw
                )
                legacy_total += _legacy_training_reward(
                    interval_components, interval_raw
                )
                next_state = stack.state
                state_delta_rms_sum += float(
                    np.sqrt(np.mean(np.square(next_state - state)))
                )
                if recorder is not None:
                    action_probabilities = [float(value) for value in action_scores]
                    probability_array = np.asarray(action_probabilities)
                    decision_row = {
                        "episode": episode_index,
                        "decision": decision,
                        "x_before": x_before,
                        "x_after": current_x,
                        "max_x": max_x,
                        "action": action_names[action],
                        "greedy_action": action_names[greedy_action],
                        "selected_probability": (
                            float(action_scores[action]) if algorithm == "ppo" else None
                        ),
                        "log_probability": log_probability,
                        "critic_value": value,
                        "action_entropy": (
                            float(
                                -np.sum(
                                    probability_array
                                    * np.log(np.maximum(probability_array, 1.0e-12))
                                )
                            )
                            if algorithm == "ppo"
                            else None
                        ),
                        "action_probabilities": dict(
                            zip(action_names, action_probabilities, strict=True)
                        ),
                        "direction_probabilities": (
                            dict(
                                zip(
                                    ("neutral", "left", "right"),
                                    (
                                        float(item)
                                        for item in policy_diagnostics.get(
                                            "direction_probabilities", []
                                        )
                                    ),
                                    strict=True,
                                )
                            )
                            if "direction_probabilities" in policy_diagnostics
                            else None
                        ),
                        "jump_probabilities": (
                            dict(
                                zip(
                                    ("released", "pressed"),
                                    (
                                        float(item)
                                        for item in policy_diagnostics.get(
                                            "jump_probabilities", []
                                        )
                                    ),
                                    strict=True,
                                )
                            )
                            if "jump_probabilities" in policy_diagnostics
                            else None
                        ),
                        "run_probabilities": (
                            dict(
                                zip(
                                    ("off", "on"),
                                    (
                                        float(item)
                                        for item in policy_diagnostics.get(
                                            "run_probabilities", []
                                        )
                                    ),
                                    strict=True,
                                )
                            )
                            if "run_probabilities" in policy_diagnostics
                            else None
                        ),
                        "state_rms": state_rms,
                        "state_delta_rms": float(
                            np.sqrt(np.mean(np.square(next_state - state)))
                        ),
                        "interval_reward": interval_raw,
                        "reward_components": interval_components,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "death": bool(death),
                        "completion": bool(completion),
                    }
                    recorder.add_decision(decision_row)
                state = next_state
                if terminated or truncated:
                    break

            death |= components["death"] < 0.0
            completion |= components["completion"] > 0.0
            assert initial_action is not None and terminal_action is not None
            if death:
                fatal_terminal_action_counts[terminal_action] += 1
            row = {
                "episode": episode_index,
                "decisions": decision,
                "raw_reward": raw_total,
                "legacy_training_reward": legacy_total,
                "environment_training_reward": environment_training_total,
                "max_x": max_x,
                "death": death,
                "completion": completion,
                "initial_action": action_names[initial_action],
                "terminal_action": action_names[terminal_action],
                "environment_terminated": bool(terminated),
                "environment_truncated": bool(truncated),
                "evaluator_truncated": not (terminated or truncated),
                "components": components,
            }
            episodes.append(row)
            if recorder is not None:
                forensic_artifacts.append(
                    {"episode": episode_index, **recorder.close()}
                )
            print(
                f"episode={episode_index} decisions={decision} max_x={max_x} "
                f"death={int(death)} completion={int(completion)} "
                f"raw={raw_total:+.1f} legacy={legacy_total:+.3f} "
                f"environment={environment_training_total:+.3f}",
                flush=True,
            )
    finally:
        env.close()

    decision_count = max(1, sum(action_counts.values()))
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dopamine_state": (
            None if args.dopamine_state is None else str(args.dopamine_state.resolve())
        ),
        "learning_enabled": False,
        "dopamine_updates_enabled": False,
        "algorithm": algorithm,
        "mode": "sampled" if args.sample_actions else "mean",
        "epsilon": args.epsilon,
        "seed": args.seed,
        "episodes": episodes,
        "forensic_artifacts": forensic_artifacts,
        "summary": {
            "episode_count": len(episodes),
            "death_rate": float(np.mean([row["death"] for row in episodes])),
            "completion_rate": float(
                np.mean([row["completion"] for row in episodes])
            ),
            "mean_decisions": float(
                np.mean([row["decisions"] for row in episodes])
            ),
            "mean_max_x": float(np.mean([row["max_x"] for row in episodes])),
            "mean_raw_reward": float(
                np.mean([row["raw_reward"] for row in episodes])
            ),
            "mean_legacy_training_reward": float(
                np.mean([row["legacy_training_reward"] for row in episodes])
            ),
            "mean_environment_training_reward": float(
                np.mean([row["environment_training_reward"] for row in episodes])
            ),
            "mean_greedy_action_score_margin": q_margin_sum / max(1, q_observations),
            "mean_state_rms": state_rms_sum / max(1, q_observations),
            "mean_temporal_state_delta_rms": (
                state_delta_rms_sum / max(1, q_observations)
            ),
            "actions": {
                action_names[index]: {
                    "selected": action_counts[index],
                    "selected_fraction": action_counts[index] / decision_count,
                    "greedy_argmax": greedy_counts[index],
                    "mean_action_score": float(
                        q_sums[index] / max(1, q_observations)
                    ),
                }
                for index in range(agent.config.num_actions)
            },
            "initial_action_counts": {
                action_names[index]: initial_action_counts[index]
                for index in range(agent.config.num_actions)
            },
            "fatal_terminal_action_counts": {
                action_names[index]: fatal_terminal_action_counts[index]
                for index in range(agent.config.num_actions)
            },
            "first_actions": first_actions,
        },
    }
    if args.output is not None:
        _atomic_json(args.output, report)
        print(f"report={args.output}", flush=True)
    print(json.dumps(report["summary"], indent=2, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dopamine-state", type=Path)
    parser.add_argument("--male-cns-data", type=Path)
    parser.add_argument(
        "--male-cns-device", choices=("auto", "cpu", "cuda"), default="cpu"
    )
    parser.add_argument("--dopamine-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--env", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-decisions-per-episode", type=int, default=500)
    parser.add_argument("--action-repeat", type=int, default=30)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument(
        "--sample-actions",
        action="store_true",
        help="sample a PPO policy instead of using its deterministic argmax",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--first-actions", type=int, default=50)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--record-dir",
        type=Path,
        help="write annotated GIFs plus synchronized frame/decision JSONL traces",
    )
    parser.add_argument(
        "--record-episodes",
        type=int,
        default=0,
        help="record this many leading episodes (requires --record-dir)",
    )
    parser.add_argument("--record-fps", type=int, default=60)
    parser.add_argument(
        "--allow-legacy-reward-contract",
        action="store_true",
        help="evaluation only: load a schema-4 checkpoint without permitting training resume",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    args = parser.parse_args()
    if (
        args.episodes <= 0
        or args.max_decisions_per_episode <= 0
        or args.action_repeat <= 0
        or args.first_actions < 0
        or args.record_episodes < 0
        or args.record_fps <= 0
        or not 0.0 <= args.epsilon <= 1.0
    ):
        parser.error("episode/decision/action counts must be positive and epsilon in [0, 1]")
    if args.record_episodes and args.record_dir is None:
        parser.error("--record-episodes requires --record-dir")
    run(args)


if __name__ == "__main__":
    main()
