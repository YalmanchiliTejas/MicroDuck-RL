#!/usr/bin/env python3
"""Diagnostic test for MicroDuck grape retention.

This is NOT an RL evaluation.

Sequence:
1. Record one deterministic GrapePick trajectory.
2. Reset.
3. Replay descent into pickup pose.
4. Place grape relative to mouth_tip.
5. Hold pickup pose while jaw should close.
6. Replay learned rise.
7. Hold standing pose.
8. Measure whether grape remained captured.

Diagnostics:
- grape height vs target
- mouth-grape distance
- grape vertical velocity
- jaw joint position, if an explicit jaw/mouth/beak joint exists
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

import mjlab_microduck.tasks  # noqa: F401
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


# ============================================================
# Helpers
# ============================================================

def _find_jaw_joint(robot):
    """Try to find an explicit jaw/mouth/beak joint.

    Returns:
        (joint_id, joint_name)

    If no suitable joint exists:
        (None, None)
    """

    for pattern in (
        r".*jaw.*",
        r".*mouth.*",
        r".*beak.*",
    ):
        try:
            joint_ids, joint_names = robot.find_joints(pattern)

            if len(joint_ids) > 0:
                return int(joint_ids[0]), str(joint_names[0])

        except Exception:
            pass

    return None, None


def _configure_env(num_envs: int, enable_dr: bool):
    cfg = load_env_cfg(TASK_ID, play=True)

    cfg.scene.num_envs = num_envs

    # Normal cycle = 4 s.
    # Give ourselves extra room for the final standing hold.
    cfg.episode_length_s = GP_PERIOD + 1.0

    cfg.auto_reset = False
    cfg.terminations = {}
    cfg.curriculum = {}

    # Start every test at phase zero.
    cfg.commands["twist"].randomize_phase = False

    if not enable_dr:
        cfg.events = {
            name: term
            for name, term in cfg.events.items()
            if name in _BASELINE_RESET_EVENTS
        }

        reset_base = cfg.events["reset_base"]

        reset_base.params["pose_range"].update(
            {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.125, 0.125),
                "yaw": (0.0, 0.0),
            }
        )

        cfg.events["reset_grape"].params["noise_xy"] = 0.0

        # Remove observation corruption for deterministic replay.
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
    """Teleport the grape relative to mouth_tip.

    Default:
        (0.010, 0, 0)

    means approximately 10 mm below mouth_tip in the current
    MicroDuck mouth coordinate frame.
    """

    robot = env.scene["robot"]
    grape = env.scene["grape"]

    mouth_ids, _ = robot.find_sites(["mouth_tip"])
    mouth_id = int(mouth_ids[0])

    env_ids = torch.arange(
        env.num_envs,
        device=env.device,
    )

    dtype = robot.data.site_pos_w.dtype

    base_offset = torch.tensor(
        local_offset_m,
        device=env.device,
        dtype=dtype,
    )

    offsets = base_offset.repeat(
        env.num_envs,
        1,
    )

    if placement_noise_m > 0:
        offsets += (
            torch.rand(
                env.num_envs,
                3,
                device=env.device,
                dtype=dtype,
            )
            * 2.0
            - 1.0
        ) * placement_noise_m

    mouth_pos = robot.data.site_pos_w[:, mouth_id, :]
    mouth_quat = robot.data.site_quat_w[:, mouth_id, :]

    # Convert local mouth offset -> world offset.
    grape_pos = mouth_pos + quat_apply(
        mouth_quat,
        offsets,
    )

    # Don't initialize grape inside the ground.
    terrain_z = env.scene.terrain.env_origins[:, 2]

    grape_pos[:, 2] = (terrain_z + GRAPE_HALF_HEIGHT)

    pose = torch.zeros(
        env.num_envs,
        7,
        device=env.device,
        dtype=dtype,
    )

    pose[:, :3] = grape_pos

    # Identity quaternion
    pose[:, 3] = 1.0

    grape.write_root_link_pose_to_sim(
        pose,
        env_ids,
    )

    grape.write_root_link_velocity_to_sim(
        torch.zeros(
            env.num_envs,
            6,
            device=env.device,
            dtype=dtype,
        ),
        env_ids,
    )

    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=0.0)

    return offsets.detach().cpu().numpy()


# ============================================================
# Metrics
# ============================================================

def summarize_trace(
    phase: np.ndarray,
    grape_height: np.ndarray,
    mouth_distance: np.ndarray,
    final_height_threshold: float,
    distance_threshold: float,
):
    """Summarize whether grape survived the lift + hold."""

    seated = phase >= HOLD_END
    final_hold = phase >= RISE_END

    if not np.any(seated):
        raise ValueError(
            "Trace does not contain the rise phase."
        )

    if not np.any(final_hold):
        raise ValueError(
            "Trace does not contain the final hold phase."
        )

    seated_height = np.where(
        seated,
        grape_height,
        np.nan,
    )

    hold_height = np.where(
        final_hold,
        grape_height,
        np.nan,
    )

    hold_distance = np.where(
        final_hold,
        mouth_distance,
        np.nan,
    )

    max_height = np.nanmax(
        seated_height,
        axis=0,
    )

    min_hold_height = np.nanmin(
        hold_height,
        axis=0,
    )

    max_hold_distance = np.nanmax(
        hold_distance,
        axis=0,
    )

    final_height = grape_height[-1]
    final_distance = mouth_distance[-1]

    success = (
        min_hold_height >= final_height_threshold
    ) & (
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


# ============================================================
# CSV
# ============================================================

def _write_trace_csv(
    path: Path,
    time_s: np.ndarray,
    phase: np.ndarray,
    phase_gate: np.ndarray,
    grape_height: np.ndarray,
    target_height: np.ndarray,
    mouth_distance: np.ndarray,
    grape_vertical_velocity: np.ndarray,
    jaw_position: np.ndarray,
):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

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
                "grape_vertical_velocity_mps",
                "jaw_position_rad",
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
                        float(grape_vertical_velocity[step, trial]),
                        float(jaw_position[step, trial]),
                    )
                )


# ============================================================
# Plot
# ============================================================

def _write_plot(
    path: Path,
    time_s: np.ndarray,
    grape_height: np.ndarray,
    target_height: np.ndarray,
    mouth_distance: np.ndarray,
    grape_vertical_velocity: np.ndarray,
    jaw_position: np.ndarray,
    phase_gate: np.ndarray,
    distance_threshold: float,
    jaw_joint_name: str | None,
):
    os.environ.setdefault(
        "MPLBACKEND",
        "Agg",
    )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11, 12),
        sharex=True,
    )

    def plot_band(ax, values, label):
        mean = np.mean(values, axis=1)

        low, high = np.percentile(
            values,
            (10, 90),
            axis=1,
        )

        ax.plot(
            time_s,
            mean,
            label=f"mean {label}",
        )

        ax.fill_between(
            time_s,
            low,
            high,
            alpha=0.2,
            label="p10-p90",
        )

        ax.grid(alpha=0.25)

    # --------------------------------------------------------
    # Plot 1: grape height
    # --------------------------------------------------------

    plot_band(
        axes[0],
        grape_height,
        "grape",
    )

    axes[0].plot(
        time_s,
        np.mean(target_height, axis=1),
        "--",
        label="target",
    )

    axes[0].set_ylabel("Height (m)")
    axes[0].set_title("Grape height")
    axes[0].legend()

    # --------------------------------------------------------
    # Plot 2: mouth distance
    # --------------------------------------------------------

    plot_band(
        axes[1],
        mouth_distance,
        "mouth distance",
    )

    axes[1].axhline(
        distance_threshold,
        linestyle="--",
        label="distance limit",
    )

    axes[1].set_ylabel("Distance (m)")
    axes[1].set_title("Mouth-grape distance")
    axes[1].legend()

    # --------------------------------------------------------
    # Plot 3: grape vertical velocity
    # --------------------------------------------------------

    plot_band(
        axes[2],
        grape_vertical_velocity,
        "grape vertical velocity",
    )

    axes[2].axhline(
        0.0,
        linewidth=1,
    )

    axes[2].set_ylabel("Velocity (m/s)")
    axes[2].set_title("Grape vertical velocity")
    axes[2].legend()

    # --------------------------------------------------------
    # Plot 4: actual jaw position, if available
    # --------------------------------------------------------

    if np.isfinite(jaw_position).any():

        plot_band(
            axes[3],
            jaw_position,
            "jaw position",
        )

        axes[3].set_ylabel("Joint position (rad)")
        axes[3].set_title(
            f"Jaw motion ({jaw_joint_name})"
        )

    else:
        # Fallback if no articulated jaw joint is exposed.
        plot_band(
            axes[3],
            phase_gate,
            "phase gate",
        )

        axes[3].set_ylabel("Gate")
        axes[3].set_title(
            "No jaw joint found - showing phase gate"
        )

    axes[3].legend()
    axes[3].set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ============================================================
# Main test
# ============================================================

def run(args: argparse.Namespace) -> None:

    checkpoint = (
        Path(args.checkpoint_file)
        .expanduser()
        .resolve()
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    output_dir = (
        Path(args.output)
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    configure_torch_backends()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device or (
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    env_cfg = _configure_env(
        args.trials,
        args.enable_dr,
    )

    agent_cfg = load_rl_cfg(TASK_ID)

    raw_env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=device,
    )

    env = RslRlVecEnvWrapper(
        raw_env,
        clip_actions=agent_cfg.clip_actions,
    )

    runner_cls = (
        load_runner_cls(TASK_ID)
        or MjlabOnPolicyRunner
    )

    runner = runner_cls(
        env,
        asdict(agent_cfg),
        device=device,
    )

    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )

    policy = runner.get_inference_policy(
        device=device,
    )

    step_dt = raw_env.step_dt

    num_steps = int(
        math.ceil(
            GP_PERIOD / step_dt
        )
    )

    # ========================================================
    # PASS 1:
    # record a deterministic learned trajectory
    # ========================================================

    obs = env.get_observations()

    action_trace: list[torch.Tensor] = []

    for _ in range(num_steps):

        with torch.inference_mode():
            policy_action = policy(obs)

        scripted_action = (
            policy_action[0]
            .detach()
            .clone()
        )

        action_trace.append(
            scripted_action
        )

        actions = (
            scripted_action
            .unsqueeze(0)
            .expand(args.trials, -1)
        )

        obs, _, _, _ = env.step(actions)

    # ========================================================
    # Convert phase locations to trajectory indices
    # ========================================================

    descent_step = int(
        math.ceil(
            DESCENT_END
            * GP_PERIOD
            / step_dt
        )
    )

    rise_start_step = int(
        math.ceil(
            HOLD_END
            * GP_PERIOD
            / step_dt
        )
    )

    rise_end_step = int(
        math.ceil(
            RISE_END
            * GP_PERIOD
            / step_dt
        )
    )

    pickup_action = action_trace[
        min(
            descent_step,
            len(action_trace) - 1,
        )
    ]

    standing_action = action_trace[
        min(
            rise_end_step,
            len(action_trace) - 1,
        )
    ]

    print()
    print("Recorded reference trajectory")
    print(f"  step_dt:         {step_dt:.6f} s")
    print(f"  total steps:     {len(action_trace)}")
    print(f"  descent step:    {descent_step}")
    print(f"  rise start step: {rise_start_step}")
    print(f"  rise end step:   {rise_end_step}")
    print()

    # ========================================================
    # PASS 2:
    # controlled physical retention test
    # ========================================================

    env.reset()

    robot = raw_env.scene["robot"]
    grape = raw_env.scene["grape"]

    mouth_ids, _ = robot.find_sites(
        ["mouth_tip"]
    )

    mouth_id = int(
        mouth_ids[0]
    )

    # Try to locate actual jaw joint.
    jaw_joint_id, jaw_joint_name = _find_jaw_joint(
        robot
    )

    if jaw_joint_id is None:
        print(
            "WARNING: no jaw/mouth/beak joint found."
        )

        try:
            print(
                "Robot joints:",
                robot.joint_names,
            )
        except Exception:
            pass

        print(
            "Fourth plot will show phase_gate instead."
        )

    else:
        print(
            f"Tracking jaw joint: "
            f"{jaw_joint_name} "
            f"(id={jaw_joint_id})"
        )

    command = (
        raw_env.command_manager
        .get_term("twist")
    )

    # ========================================================
    # Metric storage
    # ========================================================

    phase_rows = []
    gate_rows = []

    grape_height_rows = []
    target_height_rows = []
    mouth_distance_rows = []

    grape_vertical_velocity_rows = []
    jaw_position_rows = []

    applied_offsets = None

    # ========================================================
    # Metric recorder
    # ========================================================

    def record_metrics():

        phase_t = command._gp_phase.detach()

        gate_t = torch.where(
            phase_t < HOLD_END,
            torch.zeros_like(phase_t),
            torch.where(
                phase_t < RISE_END,
                (
                    phase_t - HOLD_END
                )
                / (
                    RISE_END - HOLD_END
                ),
                torch.ones_like(phase_t),
            ),
        )

        terrain_z = (
            raw_env.scene
            .terrain
            .env_origins[:, 2]
        )

        grape_height_t = (
            grape.data.root_link_pos_w[:, 2]
            - terrain_z
        )

        target_height_t = (
            GRAPE_HALF_HEIGHT
            + gate_t
            * (
                GRAPE_LIFT_HEIGHT
                - GRAPE_HALF_HEIGHT
            )
        )

        mouth_distance_t = (
            torch.linalg.vector_norm(
                robot.data.site_pos_w[
                    :, mouth_id, :
                ]
                - grape.data.root_link_pos_w,
                dim=-1,
            )
        )

        # ----------------------------------------------------
        # Grape vertical velocity
        # ----------------------------------------------------
        #
        # Root linear velocity should occupy the first 3 values:
        # vx, vy, vz.
        #
        # We care about vz.
        # ----------------------------------------------------

        grape_vertical_velocity_t = (
            grape.data.root_link_vel_w[:, 2]
        )

        # ----------------------------------------------------
        # Jaw position
        # ----------------------------------------------------

        if jaw_joint_id is not None:

            try:
                jaw_position_t = (
                    robot.data.joint_pos[
                        :, jaw_joint_id
                    ]
                    .detach()
                )

            except Exception:
                jaw_position_t = torch.full_like(
                    phase_t,
                    float("nan"),
                )

        else:
            jaw_position_t = torch.full_like(
                phase_t,
                float("nan"),
            )

        phase_rows.append(
            phase_t.cpu().numpy().copy()
        )

        gate_rows.append(
            gate_t.cpu().numpy().copy()
        )

        grape_height_rows.append(
            grape_height_t.cpu().numpy().copy()
        )

        target_height_rows.append(
            target_height_t.cpu().numpy().copy()
        )

        mouth_distance_rows.append(
            mouth_distance_t.cpu().numpy().copy()
        )

        grape_vertical_velocity_rows.append(
            grape_vertical_velocity_t
            .cpu()
            .numpy()
            .copy()
        )

        jaw_position_rows.append(
            jaw_position_t
            .cpu()
            .numpy()
            .copy()
        )

    # ========================================================
    # STEP A:
    # replay descent only
    # ========================================================

    obs = env.get_observations()

    for scripted_action in action_trace[
        :descent_step
    ]:

        actions = (
            scripted_action
            .unsqueeze(0)
            .expand(args.trials, -1)
        )

        obs, _, _, _ = env.step(actions)

        record_metrics()

    # ========================================================
    # STEP B:
    # place grape near mouth
    # ========================================================

    print(
        "Placing grape with local mouth offset:",
        (
            args.offset_x_mm,
            args.offset_y_mm,
            args.offset_z_mm,
        ),
        "mm",
    )

    applied_offsets = _seat_grape_at_mouth(
        raw_env,
        local_offset_m=(
            args.offset_x_mm / 1000.0,
            args.offset_y_mm / 1000.0,
            args.offset_z_mm / 1000.0,
        ),
        placement_noise_m=(
            args.placement_noise_mm
            / 1000.0
        ),
    )

    # Log immediately after teleport.
    record_metrics()

    # ========================================================
    # STEP C:
    # HOLD pickup pose while jaw should close
    # ========================================================

    close_steps = int(
        math.ceil(
            (
                HOLD_END
                - DESCENT_END
            )
            * GP_PERIOD
            / step_dt
        )
    )

    print(
        f"Holding pickup pose for jaw closure: "
        f"{close_steps} steps"
    )

    for _ in range(close_steps):

        actions = (
            pickup_action
            .unsqueeze(0)
            .expand(args.trials, -1)
        )

        obs, _, _, _ = env.step(actions)

        record_metrics()

    # ========================================================
    # STEP D:
    # replay learned rise
    # ========================================================

    print("Replaying learned rise...")

    for scripted_action in action_trace[
        rise_start_step:rise_end_step
    ]:

        actions = (
            scripted_action
            .unsqueeze(0)
            .expand(args.trials, -1)
        )

        obs, _, _, _ = env.step(actions)

        record_metrics()

    # ========================================================
    # STEP E:
    # hold standing pose for 0.5 s
    # ========================================================

    stand_hold_duration_s = 0.5

    stand_hold_steps = int(
        math.ceil(
            stand_hold_duration_s
            / step_dt
        )
    )

    print(
        f"Holding standing pose for "
        f"{stand_hold_duration_s:.2f}s "
        f"({stand_hold_steps} steps)"
    )

    for _ in range(stand_hold_steps):

        actions = (
            standing_action
            .unsqueeze(0)
            .expand(args.trials, -1)
        )

        obs, _, _, _ = env.step(actions)

        record_metrics()

    # ========================================================
    # Convert recordings -> numpy arrays
    # ========================================================

    if applied_offsets is None:
        raw_env.close()

        raise RuntimeError(
            "Grape was never placed."
        )

    phase = np.stack(phase_rows)

    phase_gate = np.stack(gate_rows)

    grape_height = np.stack(
        grape_height_rows
    )

    target_height = np.stack(
        target_height_rows
    )

    mouth_distance = np.stack(
        mouth_distance_rows
    )

    grape_vertical_velocity = np.stack(
        grape_vertical_velocity_rows
    )

    jaw_position = np.stack(
        jaw_position_rows
    )

    time_s = (
        np.arange(
            len(grape_height_rows),
            dtype=np.float32,
        )
        + 1.0
    ) * step_dt

    # ========================================================
    # Summary
    # ========================================================

    summary = summarize_trace(
        phase,
        grape_height,
        mouth_distance,
        final_height_threshold=(
            args.final_height_threshold
        ),
        distance_threshold=(
            args.distance_threshold
        ),
    )

    # ========================================================
    # Write CSV
    # ========================================================

    _write_trace_csv(
        output_dir / "trace.csv",
        time_s,
        phase,
        phase_gate,
        grape_height,
        target_height,
        mouth_distance,
        grape_vertical_velocity,
        jaw_position,
    )

    # ========================================================
    # Write diagnostic plot
    # ========================================================

    _write_plot(
        output_dir / "retention.png",
        time_s,
        grape_height,
        target_height,
        mouth_distance,
        grape_vertical_velocity,
        jaw_position,
        phase_gate,
        args.distance_threshold,
        jaw_joint_name,
    )

    # ========================================================
    # Trial-level outputs
    # ========================================================

    success = np.asarray(
        summary["success"]
    )

    trial_rows = []

    for trial in range(args.trials):

        trial_rows.append(
            {
                "trial": trial,

                "offset_x_m": float(
                    applied_offsets[trial, 0]
                ),

                "offset_y_m": float(
                    applied_offsets[trial, 1]
                ),

                "offset_z_m": float(
                    applied_offsets[trial, 2]
                ),

                "success": bool(
                    success[trial]
                ),

                "max_height_m": float(
                    np.asarray(
                        summary["max_height"]
                    )[trial]
                ),

                "min_hold_height_m": float(
                    np.asarray(
                        summary[
                            "min_hold_height"
                        ]
                    )[trial]
                ),

                "max_hold_distance_m": float(
                    np.asarray(
                        summary[
                            "max_hold_distance"
                        ]
                    )[trial]
                ),

                "final_height_m": float(
                    np.asarray(
                        summary["final_height"]
                    )[trial]
                ),

                "final_distance_m": float(
                    np.asarray(
                        summary[
                            "final_distance"
                        ]
                    )[trial]
                ),
            }
        )

    with (
        output_dir / "trials.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                trial_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(trial_rows)

    # ========================================================
    # Summary JSON
    # ========================================================

    report = {
        "checkpoint": str(checkpoint),

        "trials": args.trials,

        "domain_randomization": (
            args.enable_dr
        ),

        "placement_noise_mm": (
            args.placement_noise_mm
        ),

        "requested_local_offset_mm": [
            args.offset_x_mm,
            args.offset_y_mm,
            args.offset_z_mm,
        ],

        "descent_step": descent_step,

        "rise_start_step": (
            rise_start_step
        ),

        "rise_end_step": (
            rise_end_step
        ),

        "close_steps": close_steps,

        "jaw_joint_name": jaw_joint_name,

        "standing_hold_seconds": (
            stand_hold_duration_s
        ),

        "final_height_threshold_m": (
            args.final_height_threshold
        ),

        "distance_threshold_m": (
            args.distance_threshold
        ),

        "success_rate": float(
            summary["success_rate"]
        ),

        "mean_max_height_m": float(
            np.mean(
                summary["max_height"]
            )
        ),

        "mean_min_hold_height_m": float(
            np.mean(
                summary[
                    "min_hold_height"
                ]
            )
        ),

        "mean_final_height_m": float(
            np.mean(
                summary["final_height"]
            )
        ),

        "mean_final_distance_m": float(
            np.mean(
                summary[
                    "final_distance"
                ]
            )
        ),

        "mean_max_hold_distance_m": float(
            np.mean(
                summary[
                    "max_hold_distance"
                ]
            )
        ),

        "mean_peak_abs_grape_vertical_velocity_mps": float(
            np.mean(
                np.max(
                    np.abs(
                        grape_vertical_velocity
                    ),
                    axis=0,
                )
            )
        ),
    }

    (
        output_dir / "summary.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # ========================================================
    # Print result
    # ========================================================

    print()
    print("=" * 70)
    print("GRAPE RETENTION TEST")
    print("=" * 70)

    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print(
        f"Trace:  "
        f"{output_dir / 'trace.csv'}"
    )

    print(
        f"Trials: "
        f"{output_dir / 'trials.csv'}"
    )

    print(
        f"Plot:   "
        f"{output_dir / 'retention.png'}"
    )

    raw_env.close()


# ============================================================
# CLI
# ============================================================

def _parse_args():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--checkpoint-file",
        required=True,
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--placement-noise-mm",
        type=float,
        default=0.0,
    )

    # Default placement:
    # mouth_tip local +X ≈ downward.
    parser.add_argument(
        "--offset-x-mm",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--offset-y-mm",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--offset-z-mm",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--enable-dr",
        action="store_true",
    )

    parser.add_argument(
        "--final-height-threshold",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        default="artifacts/grape-retention",
    )

    args = parser.parse_args()

    if args.trials <= 0:
        parser.error(
            "--trials must be positive"
        )

    if args.placement_noise_mm < 0:
        parser.error(
            "--placement-noise-mm "
            "must be non-negative"
        )

    return args


if __name__ == "__main__":
    run(_parse_args())