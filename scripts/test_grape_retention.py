#!/usr/bin/env python3
"""Test whether the simulated mouth can retain a grape during a fixed rise.

This is deliberately not an RL evaluation.  It first records one four-second
action cycle from a trained GrapePick checkpoint.  It then resets the scene,
places the grape in the open mouth exactly once just before scripted closure,
and replays the recorded actions unchanged across all trials.  If the grape falls, the
failure is in the contact model / trajectory rather than policy exploration.

Example:
    uv run scripts/test_grape_retention.py \
      --checkpoint-file /path/to/model_1250.pt \
      --trials 128 --placement-noise-mm 2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply
from mjlab.utils.torch import configure_torch_backends

import mjlab_microduck.tasks  # noqa: F401 -- populate the task registry
from mjlab_microduck.tasks.microduck_grape_pick_env_cfg import (
    DESCENT_END,
    GP_PERIOD,
    GRAPE_HALF_HEIGHT,
    GRAPE_LIFT_HEIGHT,
    HOLD_END,
    RISE_END,
)


TASK_ID = "Mjlab-GrapePick-Flat-MicroDuck"
_BASELINE_RESET_EVENTS = {
    "expand_bam_friction_fields",
    "reset_action_history",
    "reset_base",
    "reset_grape",
    "reset_robot_joints",
}


def _configure_env(num_envs: int, enable_dr: bool):
    cfg = load_env_cfg(TASK_ID, play=True)
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = GP_PERIOD + 1.0
    cfg.auto_reset = False
    cfg.terminations = {}
    cfg.curriculum = {}

    # Every replay starts at the deployment phase origin instead of a random
    # point in the cycle.
    cfg.commands["twist"].randomize_phase = False

    if not enable_dr:
        cfg.events = {
            name: term
            for name, term in cfg.events.items()
            if name in _BASELINE_RESET_EVENTS
        }
        reset_base = cfg.events["reset_base"]
        reset_base.params["pose_range"].update(
            {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.125, 0.125), "yaw": (0.0, 0.0)}
        )
        cfg.events["reset_grape"].params["noise_xy"] = 0.0

        # Observation corruption is irrelevant during fixed replay, but also
        # makes the recorded reference trajectory deterministic.
        for group in cfg.observations.values():
            group.enable_corruption = False
        for name in ("base_ang_vel", "projected_gravity"):
            term = cfg.observations["actor"].terms.get(name)
            if term is not None and "max_angle_deg" in term.params:
                term.params["max_angle_deg"] = 0.0

    return cfg


def _seat_grape_at_mouth(
    env: ManagerBasedRlEnv,
    local_offset_m: tuple[float, float, float],
    placement_noise_m: float,
) -> np.ndarray:
    """Place the grape once at the mouth and return applied local offsets."""
    robot = env.scene["robot"]
    grape = env.scene["grape"]
    mouth_ids, _ = robot.find_sites(["mouth_tip"])
    mouth_id = int(mouth_ids[0])
    env_ids = torch.arange(env.num_envs, device=env.device)

    dtype = robot.data.site_pos_w.dtype
    base_offset = torch.tensor(local_offset_m, device=env.device, dtype=dtype)
    offsets = base_offset.repeat(env.num_envs, 1)
    if placement_noise_m > 0.0:
        offsets += (
            torch.rand(env.num_envs, 3, device=env.device, dtype=dtype) * 2.0 - 1.0
        ) * placement_noise_m

    mouth_pos = robot.data.site_pos_w[:, mouth_id, :]
    mouth_quat = robot.data.site_quat_w[:, mouth_id, :]
    grape_pos = mouth_pos + quat_apply(mouth_quat, offsets)
    # The grape starts supported by the terrain; never initialize its center
    # below its physical half-height merely to satisfy the requested offset.
    terrain_z = env.scene.terrain.env_origins[:, 2]
    grape_pos[:, 2] = torch.maximum(
        grape_pos[:, 2], terrain_z + GRAPE_HALF_HEIGHT
    )

    pose = torch.zeros(env.num_envs, 7, device=env.device, dtype=dtype)
    pose[:, :3] = grape_pos
    pose[:, 3] = 1.0
    grape.write_root_link_pose_to_sim(pose, env_ids)
    grape.write_root_link_velocity_to_sim(
        torch.zeros(env.num_envs, 6, device=env.device, dtype=dtype), env_ids
    )
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=0.0)
    return offsets.detach().cpu().numpy()


def summarize_trace(
    phase: np.ndarray,
    grape_height: np.ndarray,
    mouth_distance: np.ndarray,
    final_height_threshold: float,
    distance_threshold: float,
) -> dict[str, np.ndarray | float]:
    """Summarize trial columns from arrays shaped ``(steps, trials)``."""
    seated = phase >= HOLD_END
    final_hold = phase >= RISE_END
    if not np.any(seated) or not np.any(final_hold):
        raise ValueError("Trace does not contain the rise and final-hold phases")

    seated_height = np.where(seated, grape_height, np.nan)
    hold_height = np.where(final_hold, grape_height, np.nan)
    hold_distance = np.where(final_hold, mouth_distance, np.nan)
    max_height = np.nanmax(seated_height, axis=0)
    min_hold_height = np.nanmin(hold_height, axis=0)
    max_hold_distance = np.nanmax(hold_distance, axis=0)
    final_height = grape_height[-1]
    final_distance = mouth_distance[-1]
    success = (min_hold_height >= final_height_threshold) & (
        max_hold_distance <= distance_threshold
    )
    return {
        "success": success,
        "success_rate": float(np.mean(success)),
        "max_height": max_height,
        "min_hold_height": min_hold_height,
        "max_hold_distance": max_hold_distance,
        "final_height": final_height,
        "final_distance": final_distance,
    }


def _write_trace_csv(
    path: Path,
    time_s: np.ndarray,
    phase: np.ndarray,
    phase_gate: np.ndarray,
    grape_height: np.ndarray,
    target_height: np.ndarray,
    mouth_distance: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            (
                "trial",
                "time_s",
                "phase",
                "phase_gate",
                "grape_height_m",
                "target_grape_height_m",
                "mouth_grape_distance_m",
            )
        )
        for step in range(len(time_s)):
            for trial in range(grape_height.shape[1]):
                writer.writerow(
                    (
                        trial,
                        float(time_s[step]),
                        float(phase[step, trial]),
                        float(phase_gate[step, trial]),
                        float(grape_height[step, trial]),
                        float(target_height[step, trial]),
                        float(mouth_distance[step, trial]),
                    )
                )


def _write_plot(
    path: Path,
    time_s: np.ndarray,
    grape_height: np.ndarray,
    target_height: np.ndarray,
    mouth_distance: np.ndarray,
    distance_threshold: float,
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, values, label, ylabel in (
        (axes[0], grape_height, "grape", "Height (m)"),
        (axes[1], mouth_distance, "mouth distance", "Distance (m)"),
    ):
        mean = np.mean(values, axis=1)
        low, high = np.percentile(values, (10, 90), axis=1)
        ax.plot(time_s, mean, label=f"mean {label}")
        ax.fill_between(time_s, low, high, alpha=0.2, label="p10–p90")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()

    axes[0].plot(time_s, np.mean(target_height, axis=1), "--", label="target")
    axes[0].legend()
    axes[1].axhline(distance_threshold, color="red", linestyle="--", label="limit")
    axes[1].legend()
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint_file).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_torch_backends()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = _configure_env(args.trials, args.enable_dr)
    agent_cfg = load_rl_cfg(TASK_ID)
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    step_dt = raw_env.step_dt
    num_steps = int(math.ceil(GP_PERIOD / step_dt))

    # Pass 1: record a single deterministic policy action sequence. Broadcasting
    # env 0's action makes the later physical trials share exactly one script.
    obs = env.get_observations()
    action_trace: list[torch.Tensor] = []
    for _ in range(num_steps):
        # Restrict inference mode to the network forward pass. mjlab/BAM creates
        # and mutates stateful delay-buffer tensors inside env.step/reset; running
        # those operations under inference mode makes the tensors immutable on
        # the following reset.
        with torch.inference_mode():
            policy_action = policy(obs)
        scripted_action = policy_action[0].detach().clone()
        action_trace.append(scripted_action)
        actions = scripted_action.unsqueeze(0).expand(args.trials, -1)
        obs, _, _, _ = env.step(actions)

    # Pass 2: reset, seat once while the jaw is still open at DESCENT_END,
    # then let the scripted mouth close over [DESCENT_END, HOLD_END].
    env.reset()
    robot = raw_env.scene["robot"]
    grape = raw_env.scene["grape"]
    mouth_ids, _ = robot.find_sites(["mouth_tip"])
    mouth_id = int(mouth_ids[0])
    command = raw_env.command_manager.get_term("twist")
    seat_step = int(math.ceil(DESCENT_END * GP_PERIOD / step_dt))
    applied_offsets: np.ndarray | None = None

    phase_rows: list[np.ndarray] = []
    gate_rows: list[np.ndarray] = []
    grape_height_rows: list[np.ndarray] = []
    target_height_rows: list[np.ndarray] = []
    mouth_distance_rows: list[np.ndarray] = []

    for step, scripted_action in enumerate(action_trace):
        if step == seat_step:
            applied_offsets = _seat_grape_at_mouth(
                raw_env,
                local_offset_m=(
                    args.offset_x_mm / 1000.0,
                    args.offset_y_mm / 1000.0,
                    args.offset_z_mm / 1000.0,
                ),
                placement_noise_m=args.placement_noise_mm / 1000.0,
            )

        actions = scripted_action.unsqueeze(0).expand(args.trials, -1)
        env.step(actions)

        phase_t = command._gp_phase.detach()
        gate_t = torch.where(
            phase_t < HOLD_END,
            torch.zeros_like(phase_t),
            torch.where(
                phase_t < RISE_END,
                (phase_t - HOLD_END) / (RISE_END - HOLD_END),
                torch.ones_like(phase_t),
            ),
        )
        terrain_z = raw_env.scene.terrain.env_origins[:, 2]
        grape_height_t = grape.data.root_link_pos_w[:, 2] - terrain_z
        target_t = GRAPE_HALF_HEIGHT + gate_t * (
            GRAPE_LIFT_HEIGHT - GRAPE_HALF_HEIGHT
        )
        distance_t = torch.linalg.vector_norm(
            robot.data.site_pos_w[:, mouth_id, :] - grape.data.root_link_pos_w,
            dim=-1,
        )

        phase_rows.append(phase_t.cpu().numpy().copy())
        gate_rows.append(gate_t.cpu().numpy().copy())
        grape_height_rows.append(grape_height_t.cpu().numpy().copy())
        target_height_rows.append(target_t.cpu().numpy().copy())
        mouth_distance_rows.append(distance_t.cpu().numpy().copy())

    raw_env.close()
    if applied_offsets is None:
        raise RuntimeError("Grape was never seated; cycle ended before DESCENT_END")

    phase = np.stack(phase_rows)
    phase_gate = np.stack(gate_rows)
    grape_height = np.stack(grape_height_rows)
    target_height = np.stack(target_height_rows)
    mouth_distance = np.stack(mouth_distance_rows)
    time_s = (np.arange(num_steps, dtype=np.float32) + 1.0) * step_dt
    summary = summarize_trace(
        phase,
        grape_height,
        mouth_distance,
        final_height_threshold=args.final_height_threshold,
        distance_threshold=args.distance_threshold,
    )

    _write_trace_csv(
        output_dir / "trace.csv",
        time_s,
        phase,
        phase_gate,
        grape_height,
        target_height,
        mouth_distance,
    )
    _write_plot(
        output_dir / "retention.png",
        time_s,
        grape_height,
        target_height,
        mouth_distance,
        args.distance_threshold,
    )

    success = np.asarray(summary["success"])
    trial_rows = []
    for trial in range(args.trials):
        trial_rows.append(
            {
                "trial": trial,
                "offset_x_m": float(applied_offsets[trial, 0]),
                "offset_y_m": float(applied_offsets[trial, 1]),
                "offset_z_m": float(applied_offsets[trial, 2]),
                "success": bool(success[trial]),
                "max_height_m": float(np.asarray(summary["max_height"])[trial]),
                "min_hold_height_m": float(
                    np.asarray(summary["min_hold_height"])[trial]
                ),
                "max_hold_distance_m": float(
                    np.asarray(summary["max_hold_distance"])[trial]
                ),
                "final_height_m": float(np.asarray(summary["final_height"])[trial]),
                "final_distance_m": float(
                    np.asarray(summary["final_distance"])[trial]
                ),
            }
        )
    with (output_dir / "trials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trial_rows[0]))
        writer.writeheader()
        writer.writerows(trial_rows)

    report = {
        "checkpoint": str(checkpoint),
        "trials": args.trials,
        "domain_randomization": args.enable_dr,
        "placement_noise_mm": args.placement_noise_mm,
        "requested_local_offset_mm": [
            args.offset_x_mm,
            args.offset_y_mm,
            args.offset_z_mm,
        ],
        "final_height_threshold_m": args.final_height_threshold,
        "distance_threshold_m": args.distance_threshold,
        "success_rate": summary["success_rate"],
        "mean_max_height_m": float(np.mean(summary["max_height"])),
        "mean_min_hold_height_m": float(np.mean(summary["min_hold_height"])),
        "mean_final_distance_m": float(np.mean(summary["final_distance"])),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(report, indent=2))
    print(f"Trace:   {output_dir / 'trace.csv'}")
    print(f"Trials:  {output_dir / 'trials.csv'}")
    print(f"Plot:    {output_dir / 'retention.png'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-file", required=True)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--placement-noise-mm", type=float, default=0.0)
    parser.add_argument("--offset-x-mm", type=float, default=10.0)
    parser.add_argument("--offset-y-mm", type=float, default=0.0)
    parser.add_argument("--offset-z-mm", type=float, default=0.0)
    parser.add_argument("--enable-dr", action="store_true")
    parser.add_argument("--final-height-threshold", type=float, default=0.10)
    parser.add_argument("--distance-threshold", type=float, default=0.03)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/grape-retention")
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be positive")
    if args.placement_noise_mm < 0.0:
        parser.error("--placement-noise-mm must be non-negative")
    return args


if __name__ == "__main__":
    run(_parse_args())
