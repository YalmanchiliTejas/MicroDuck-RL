#!/usr/bin/env python3
"""Headless, command-conditioned checkpoint evaluation (no training or video).

Example on a GPU node:
    uv run python scripts/evaluate_mario_controller.py /path/model_500.pt

Reports clean-success fractions over ready command timesteps, not command
completion probabilities. Sampled actions are the default; --mode mean checks
the same actor and normalizer used by deployment without a consolidation run.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch


COMMANDS = {
    "neutral": (0., 0., 0.),
    "left": (-1., 0., 0.),
    "right": (1., 0., 0.),
    "jump": (0., 0., 1.),
    "left_jump": (-1., 0., 1.),
    "right_jump": (1., 0., 1.),
}


def command_counts(commands, ready, success, fallen):
    """Keep neutral, transitions and terminal failures out of active success."""
    counts = []
    for target in COMMANDS.values():
        eligible = (commands == commands.new_tensor(target)).all(-1) & ready.bool()
        clean = eligible & success.bool() & ~fallen.bool()
        counts.append(torch.stack((eligible.sum(), clean.sum())))
    return torch.stack(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("sampled", "mean"), default="sampled")
    parser.add_argument("--include-combinations", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.num_envs <= 0 or args.steps <= 0:
        parser.error("--num-envs and --steps must be positive")

    # Import the registry plugin before loading the task.
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from rsl_rl.runners import OnPolicyRunner

    task = "Mjlab-MarioController-Flat-MicroDuck"
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    weights = [0.] * 14
    for index in ((0, 1, 2, 5, 8, 11) if args.include_combinations else (0, 1, 2, 5)):
        weights[index] = 1.
    cfg.commands["twist"].category_weights = tuple(weights)
    agent_cfg = load_rl_cfg(task)
    raw_env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    try:
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(task) or OnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=args.device)
        runner.load(str(args.checkpoint), map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        obs, _ = env.reset()
        metrics = raw_env.metrics_manager
        ready_idx = metrics.active_terms.index("command_ready")
        success_idx = metrics.active_terms.index("requested_button_success")
        counts = torch.zeros(len(COMMANDS), 2, dtype=torch.long, device=args.device)
        falls = torch.zeros((), dtype=torch.long, device=args.device)
        endings = torch.zeros_like(falls)
        with torch.inference_mode():
            for _ in range(args.steps):
                # Commands may change/reset in step(); label with the request
                # which the action actually answered, not the next request.
                commands = raw_env.command_manager.get_command("twist").clone()
                actions = policy(obs, stochastic_output=args.mode == "sampled")
                obs, _, done, _ = env.step(actions)
                fallen = raw_env.termination_manager.get_term("fell_over")
                counts += command_counts(
                    commands, metrics._step_values[:, ready_idx],
                    metrics._step_values[:, success_idx], fallen,
                )
                falls += fallen.sum()
                endings += done.sum()
        report = {
            "checkpoint": str(args.checkpoint.resolve()),
            "mode": args.mode, "seed": args.seed,
            "num_envs": args.num_envs, "steps": args.steps,
            "environment": "play configuration, balanced command sampling",
            "falls": falls.item(), "completed_episodes": endings.item(),
            "commands": {},
        }
        for name, (eligible, successful) in zip(COMMANDS, counts.cpu().tolist(), strict=True):
            report["commands"][name] = {
                "ready_timesteps": eligible,
                "clean_timesteps": successful,
                "clean_fraction": successful / eligible if eligible else None,
            }
        print(json.dumps(report, indent=2))
    finally:
        raw_env.close()


if __name__ == "__main__":
    main()
