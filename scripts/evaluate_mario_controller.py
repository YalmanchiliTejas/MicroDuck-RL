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

FAILURE_COMPONENTS = (
    "requested_pressed", "wrong_buttons_released", "support", "camera",
    "leg_pose", "foot_pose", "left_released_if_unrequested",
    "right_released_if_unrequested", "jump_released_if_unrequested",
    "b_released_if_unrequested",
)
CAMERA_COMPONENTS = ("trunk_height", "camera_height", "trunk_tilt", "view_alignment")


class TransitionDiagnostics:
    """Attribute falls to the last actual command switch; resets start anew."""

    def __init__(self, num_envs, device):
        self.names = list(COMMANDS) + ["episode_start"]
        self.previous = torch.full((num_envs,), 6, device=device, dtype=torch.long)
        self.source = self.previous.clone()
        self.previous_age = torch.full((num_envs,), -1., device=device)
        self.reset = torch.ones(num_envs, device=device, dtype=torch.bool)
        # start count, exposure steps, fall count, falls within first 0.5 s
        self.counts = torch.zeros(7, 6, 4, device=device, dtype=torch.long)

    def update(self, commands, ages, fallen, done):
        targets = commands.new_tensor(list(COMMANDS.values()))
        matches = (commands[:, None, :] == targets[None, :, :]).all(-1)
        if not matches.any(-1).all():
            raise ValueError("Unexpected command in transition diagnostics")
        current = matches.long().argmax(-1)
        changed = self.reset | (current != self.previous) | (ages < self.previous_age)
        self.source = torch.where(changed, self.previous, self.source)
        self.source = torch.where(self.reset, 6, self.source)
        indices = self.source * 6 + current
        values = torch.stack((changed.long(), torch.ones_like(current), fallen.long(),
                              (fallen.bool() & (ages <= 0.5)).long()), dim=-1)
        self.counts.view(-1, 4).index_add_(0, indices, values)
        self.previous = current.clone()
        self.previous_age = ages.clone()
        self.reset = done.bool().clone()

    def report(self, step_dt):
        result = {}
        for source, rows in zip(self.names, self.counts.cpu().tolist(), strict=True):
            for target, (starts, steps, falls, early) in zip(COMMANDS, rows, strict=True):
                if steps:
                    result[f"{source}->{target}"] = {
                        "command_windows_started": starts,
                        "exposure_seconds": steps * step_dt,
                        "falls": falls, "falls_within_0_5s": early,
                        "falls_per_100_seconds": falls * 100 / (steps * step_dt),
                    }
        return result


class PreFallDiagnostics:
    """Sample failed checks before a fall, never from a preceding episode."""

    def __init__(self, num_envs, num_checks, step_dt, device):
        self.lags = {str(seconds): max(1, round(seconds / step_dt)) for seconds in (.2, .5)}
        self.history = torch.zeros(max(self.lags.values()) + 1, num_envs, num_checks,
                                   dtype=torch.bool, device=device)
        self.age = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.t = 0
        self.totals = {key: torch.zeros(num_checks, dtype=torch.long, device=device) for key in self.lags}
        self.samples = {key: torch.zeros((), dtype=torch.long, device=device) for key in self.lags}

    def update(self, failed, fallen, done):
        self.history[self.t % len(self.history)] = failed
        for key, lag in self.lags.items():
            valid = fallen.bool() & (self.age >= lag)
            past = self.history[(self.t - lag) % len(self.history)]
            self.totals[key] += (past & valid[:, None]).sum(0)
            self.samples[key] += valid.sum()
        self.age = torch.where(done.bool(), 0, self.age + 1)
        self.t += 1

    def report(self, names):
        return {
            key: {
                "eligible_falls": self.samples[key].item(),
                "failure_fractions": {
                    name.removeprefix("diagnostic_"): n / self.samples[key].item()
                    if self.samples[key].item() else None
                    for name, n in zip(names, self.totals[key].cpu().tolist(), strict=True)
                },
            }
            for key in self.lags
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
    parser.add_argument("--output", type=Path, help="Also save the JSON report to this file")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.num_envs <= 0 or args.steps <= 0:
        parser.error("--num-envs and --steps must be positive")

    # Import the registry plugin before loading the task.
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers import MetricsTermCfg
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from rsl_rl.runners import OnPolicyRunner
    from mjlab_microduck.tasks import mdp

    task = "Mjlab-MarioController-Flat-MicroDuck"
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    weights = [0.] * 14
    for index in ((0, 1, 2, 5, 8, 11) if args.include_combinations else (0, 1, 2, 5)):
        weights[index] = 1.
    cfg.commands["twist"].category_weights = tuple(weights)
    success_params = cfg.metrics["requested_button_success"].params
    diagnostic_names = []
    for component in FAILURE_COMPONENTS:
        name = f"diagnostic_{component}"
        cfg.metrics[name] = MetricsTermCfg(
            func=mdp.mario_clean_button_success,
            params={**success_params, "component": component},
        )
        diagnostic_names.append(name)
    for component in CAMERA_COMPONENTS:
        name = f"diagnostic_camera_{component}"
        cfg.metrics[name] = MetricsTermCfg(
            func=mdp.mario_camera_ready,
            params={**cfg.metrics["camera_ready"].params, "component": component},
        )
        diagnostic_names.append(name)
    # Supplemental balance diagnostic, NOT a new clean-success requirement.
    cfg.metrics["diagnostic_com_balance"] = MetricsTermCfg(
        func=cfg.rewards["commanded_trunk_offset"].func,
        params=dict(cfg.rewards["commanded_trunk_offset"].params),
    )
    diagnostic_names.append("diagnostic_com_balance")
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
        diagnostic_indices = [metrics.active_terms.index(n) for n in diagnostic_names]
        failures = torch.zeros(len(COMMANDS), len(diagnostic_names),
                               dtype=torch.long, device=args.device)
        fall_failures = torch.zeros(len(diagnostic_names), dtype=torch.long, device=args.device)
        transitions = TransitionDiagnostics(args.num_envs, args.device)
        prefall = PreFallDiagnostics(args.num_envs, len(diagnostic_names), raw_env.step_dt, args.device)
        readiness_threshold = success_params.get("readiness_threshold", 0.5)
        with torch.inference_mode():
            for _ in range(args.steps):
                # Commands may change/reset in step(); label with the request
                # which the action actually answered, not the next request.
                commands = raw_env.command_manager.get_command("twist").clone()
                ages = raw_env.command_manager.get_term("twist").command_age.clone()
                actions = policy(obs, stochastic_output=args.mode == "sampled")
                obs, _, done, _ = env.step(actions)
                fallen = raw_env.termination_manager.get_term("fell_over")
                counts += command_counts(
                    commands, metrics._step_values[:, ready_idx],
                    metrics._step_values[:, success_idx], fallen,
                )
                falls += fallen.sum()
                endings += done.sum()
                # All values are stored by MetricsManager before auto-reset.
                failed = metrics._step_values[:, diagnostic_indices] < readiness_threshold
                ready = metrics._step_values[:, ready_idx].bool()
                for i, target in enumerate(COMMANDS.values()):
                    eligible = (commands == commands.new_tensor(target)).all(-1) & ready
                    failures[i] += (failed & eligible[:, None]).sum(0)
                fall_failures += (failed & fallen[:, None]).sum(0)
                transitions.update(commands, ages, fallen, done)
                prefall.update(failed, fallen, done)
        report = {
            "checkpoint": str(args.checkpoint.resolve()),
            "mode": args.mode, "seed": args.seed,
            "num_envs": args.num_envs, "steps": args.steps,
            "environment": "play configuration, balanced command sampling",
            "falls": falls.item(), "completed_episodes": endings.item(),
            "commands": {},
            "diagnostic_notes": [
                "Failure fractions use ready timesteps for each command; causes overlap.",
                "Camera subgate failures use the same readiness threshold; several mildly reduced subgates can also fail the product.",
                "Transition attribution is association, not proof of cause. Windows still active at evaluation end are censored.",
                "Failure measurements are captured before automatic episode reset.",
                "com_balance is supplemental: existing CoM support-region score below 0.5; it is not part of clean success.",
            ],
            "transitions": transitions.report(raw_env.step_dt),
            "seconds_before_fall": prefall.report(diagnostic_names),
        }
        failure_rows = failures.cpu().tolist()
        for i, (name, (eligible, successful)) in enumerate(zip(COMMANDS, counts.cpu().tolist(), strict=True)):
            report["commands"][name] = {
                "ready_timesteps": eligible,
                "clean_timesteps": successful,
                "clean_fraction": successful / eligible if eligible else None,
                "failure_fractions": {
                    key.removeprefix("diagnostic_"): count / eligible if eligible else None
                    for key, count in zip(diagnostic_names, failure_rows[i], strict=True)
                },
            }
        report["failure_fractions_at_fall"] = {
            key.removeprefix("diagnostic_"): count / falls.item() if falls.item() else None
            for key, count in zip(diagnostic_names, fall_failures.cpu().tolist(), strict=True)
        }
        print(json.dumps(report, indent=2))
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        raw_env.close()


if __name__ == "__main__":
    main()
