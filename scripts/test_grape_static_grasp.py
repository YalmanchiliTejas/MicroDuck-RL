#!/usr/bin/env python3
"""Static MicroDuck grape grasp test.

Purpose
-------
Answer one question only:

    Can the simulated closed mouth physically hold the grape?

Test sequence
-------------
1. Load a trained GrapePick checkpoint.
2. Record one deterministic policy trajectory.
3. Reset the environment.
4. Replay only the descent, so the robot reaches its learned pickup pose.
5. Place the grape on the ground near mouth_tip.
6. Hold the exact pickup pose while the jaw closes.
7. Continue holding the same pickup pose for a configurable amount of time.
8. Measure whether the grape remains near the mouth and off / above its
   original resting position.

There is NO lift in this test.

Example
-------
uv run scripts/test_grape_static_grasp.py \
    --checkpoint-file /path/to/model.pt \
    --trials 64 \
    --offset-x-mm 0 \
    --offset-y-mm 0 \
    --offset-z-mm 0 \
    --hold-seconds 1.5
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
    HOLD_END,
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
# Environment
# ============================================================

def _configure_env(
    num_envs: int,
    enable_dr: bool,
    hold_seconds: float,
):
    cfg = load_env_cfg(
        TASK_ID,
        play=True,
    )

    cfg.scene.num_envs = num_envs

    # Enough time for:
    # descent + jaw closure + static hold.
    cfg.episode_length_s = (
        GP_PERIOD
        + hold_seconds
        + 1.0
    )

    cfg.auto_reset = False
    cfg.terminations = {}
    cfg.curriculum = {}

    # Always start at phase zero.
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

        for group in cfg.observations.values():
            group.enable_corruption = False

        for name in (
            "base_ang_vel",
            "projected_gravity",
        ):
            term = cfg.observations["actor"].terms.get(name)

            if (
                term is not None
                and "max_angle_deg" in term.params
            ):
                term.params["max_angle_deg"] = 0.0

    return cfg


# ============================================================
# Jaw lookup
# ============================================================

def _find_jaw_joint(robot):
    """Find the passive jaw / mouth joint if available."""

    for pattern in (
        r".*passive_mouth.*",
        r".*jaw.*",
        r".*mouth.*",
        r".*beak.*",
    ):
        try:
            joint_ids, joint_names = robot.find_joints(
                pattern
            )

            if len(joint_ids) > 0:
                return (
                    int(joint_ids[0]),
                    str(joint_names[0]),
                )

        except Exception:
            pass

    return None, None


# ============================================================
# Grape placement
# ============================================================

def _place_grape_on_ground_near_mouth(
    env: ManagerBasedRlEnv,
    local_offset_m: tuple[float, float, float],
    placement_noise_m: float,
) -> np.ndarray:
    """Place grape using mouth-relative X/Y but force it onto ground.

    Important:

    We deliberately force the grape center to:

        terrain_z + GRAPE_HALF_HEIGHT

    so it starts resting on the floor instead of floating and falling.

    The local mouth offset is still useful for choosing horizontal placement.
    """

    robot = env.scene["robot"]
    grape = env.scene["grape"]

    mouth_ids, _ = robot.find_sites(
        ["mouth_tip"]
    )

    mouth_id = int(
        mouth_ids[0]
    )

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

    mouth_pos = robot.data.site_pos_w[
        :, mouth_id, :
    ]

    mouth_quat = robot.data.site_quat_w[
        :, mouth_id, :
    ]

    grape_pos = mouth_pos + quat_apply(
        mouth_quat,
        offsets,
    )

    terrain_z = (
        env.scene
        .terrain
        .env_origins[:, 2]
    )

    # Force grape to rest on ground.
    grape_pos[:, 2] = (
        terrain_z
        + GRAPE_HALF_HEIGHT
    )

    pose = torch.zeros(
        env.num_envs,
        7,
        device=env.device,
        dtype=dtype,
    )

    pose[:, :3] = grape_pos

    # identity quaternion
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
# Plot
# ============================================================

def _write_plot(
    path: Path,
    time_s: np.ndarray,
    grape_height: np.ndarray,
    mouth_distance: np.ndarray,
    grape_vertical_velocity: np.ndarray,
    jaw_position: np.ndarray,
    jaw_joint_name: str | None,
    ground_height: float,
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

    def plot_band(
        ax,
        values,
        label,
    ):
        mean = np.mean(
            values,
            axis=1,
        )

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
        ax.legend()

    # --------------------------------------------------------
    # 1. Grape height
    # --------------------------------------------------------

    plot_band(
        axes[0],
        grape_height,
        "grape height",
    )

    axes[0].axhline(
        ground_height,
        linestyle="--",
        label="ground-rest height",
    )

    axes[0].set_ylabel(
        "Height (m)"
    )

    axes[0].set_title(
        "Grape height"
    )

    axes[0].legend()

    # --------------------------------------------------------
    # 2. Mouth distance
    # --------------------------------------------------------

    plot_band(
        axes[1],
        mouth_distance,
        "mouth distance",
    )

    axes[1].set_ylabel(
        "Distance (m)"
    )

    axes[1].set_title(
        "Mouth-grape distance"
    )

    # --------------------------------------------------------
    # 3. Vertical velocity
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

    axes[2].set_ylabel(
        "Velocity (m/s)"
    )

    axes[2].set_title(
        "Grape vertical velocity"
    )

    # --------------------------------------------------------
    # 4. Jaw position
    # --------------------------------------------------------

    if np.isfinite(
        jaw_position
    ).any():

        plot_band(
            axes[3],
            jaw_position,
            "jaw position",
        )

        axes[3].set_ylabel(
            "Joint position (rad)"
        )

        axes[3].set_title(
            f"Jaw motion ({jaw_joint_name})"
        )

    else:
        axes[3].text(
            0.5,
            0.5,
            "No jaw joint detected",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes[3].transAxes,
        )

        axes[3].set_title(
            "Jaw motion unavailable"
        )

    axes[3].set_xlabel(
        "Time (s)"
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=160,
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def run(
    args: argparse.Namespace,
) -> None:

    checkpoint = (
        Path(
            args.checkpoint_file
        )
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

    torch.manual_seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    device = args.device or (
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    env_cfg = _configure_env(
        args.trials,
        args.enable_dr,
        args.hold_seconds,
    )

    agent_cfg = load_rl_cfg(
        TASK_ID
    )

    raw_env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=device,
    )

    env = RslRlVecEnvWrapper(
        raw_env,
        clip_actions=agent_cfg.clip_actions,
    )

    runner_cls = (
        load_runner_cls(
            TASK_ID
        )
        or MjlabOnPolicyRunner
    )

    runner = runner_cls(
        env,
        asdict(agent_cfg),
        device=device,
    )

    runner.load(
        str(checkpoint),
        load_cfg={
            "actor": True,
        },
        strict=True,
        map_location=device,
    )

    policy = runner.get_inference_policy(
        device=device,
    )

    step_dt = raw_env.step_dt

    num_steps = int(
        math.ceil(
            GP_PERIOD
            / step_dt
        )
    )

    # ========================================================
    # PASS 1
    #
    # Record deterministic reference trajectory.
    # ========================================================

    obs = env.get_observations()

    action_trace: list[
        torch.Tensor
    ] = []

    for _ in range(
        num_steps
    ):

        with torch.inference_mode():
            policy_action = policy(
                obs
            )

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
            .expand(
                args.trials,
                -1,
            )
        )

        obs, _, _, _ = env.step(
            actions
        )

    # ========================================================
    # Find pickup point and jaw-close duration.
    # ========================================================

    descent_step = int(
        math.ceil(
            DESCENT_END
            * GP_PERIOD
            / step_dt
        )
    )

    close_end_step = int(
        math.ceil(
            HOLD_END
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

    close_steps = (
        close_end_step
        - descent_step
    )

    print()
    print(
        "STATIC GRASP TEST"
    )

    print(
        f"step_dt:       "
        f"{step_dt:.6f}"
    )

    print(
        f"descent_step:  "
        f"{descent_step}"
    )

    print(
        f"close_steps:   "
        f"{close_steps}"
    )

    print(
        f"static hold:   "
        f"{args.hold_seconds:.2f}s"
    )

    # ========================================================
    # PASS 2
    #
    # Controlled static grasp test.
    # ========================================================

    env.reset()

    robot = raw_env.scene[
        "robot"
    ]

    grape = raw_env.scene[
        "grape"
    ]

    mouth_ids, _ = robot.find_sites(
        ["mouth_tip"]
    )

    mouth_id = int(
        mouth_ids[0]
    )

    jaw_joint_id, jaw_joint_name = (
        _find_jaw_joint(
            robot
        )
    )

    if jaw_joint_id is None:
        print(
            "WARNING: no jaw joint found."
        )
    else:
        print(
            f"Jaw joint: "
            f"{jaw_joint_name} "
            f"(id={jaw_joint_id})"
        )

    # ========================================================
    # Metric storage
    # ========================================================

    grape_height_rows = []
    mouth_distance_rows = []
    grape_vertical_velocity_rows = []
    jaw_position_rows = []

    stage_rows = []

    # Stage meanings:
    #
    # 0 = descent
    # 1 = jaw closing
    # 2 = static hold

    def record_metrics(
        stage: int,
    ):
        terrain_z = (
            raw_env.scene
            .terrain
            .env_origins[:, 2]
        )

        grape_height_t = (
            grape.data.root_link_pos_w[:, 2]
            - terrain_z
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

        grape_vz_t = (
            grape.data.root_link_vel_w[
                :, 2
            ]
        )

        if jaw_joint_id is not None:
            try:
                jaw_t = (
                    robot.data.joint_pos[
                        :, jaw_joint_id
                    ]
                    .detach()
                )
            except Exception:
                jaw_t = torch.full_like(
                    grape_height_t,
                    float("nan"),
                )
        else:
            jaw_t = torch.full_like(
                grape_height_t,
                float("nan"),
            )

        grape_height_rows.append(
            grape_height_t
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        mouth_distance_rows.append(
            mouth_distance_t
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        grape_vertical_velocity_rows.append(
            grape_vz_t
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        jaw_position_rows.append(
            jaw_t
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        stage_rows.append(
            np.full(
                args.trials,
                stage,
                dtype=np.int32,
            )
        )

    # ========================================================
    # STEP A
    #
    # Replay descent into pickup pose.
    # ========================================================

    obs = env.get_observations()

    print(
        "Moving robot into pickup pose..."
    )

    for scripted_action in action_trace[
        :descent_step
    ]:

        actions = (
            scripted_action
            .unsqueeze(0)
            .expand(
                args.trials,
                -1,
            )
        )

        obs, _, _, _ = env.step(
            actions
        )

        record_metrics(
            stage=0
        )

    # ========================================================
    # STEP B
    #
    # Put grape on ground near mouth.
    # ========================================================

    print(
        "Placing grape..."
    )

    applied_offsets = (
        _place_grape_on_ground_near_mouth(
            raw_env,
            local_offset_m=(
                args.offset_x_mm
                / 1000.0,

                args.offset_y_mm
                / 1000.0,

                args.offset_z_mm
                / 1000.0,
            ),
            placement_noise_m=(
                args.placement_noise_mm
                / 1000.0
            ),
        )
    )

    # Record placement before another physics step.
    record_metrics(
        stage=1
    )

    # ========================================================
    # STEP C
    #
    # Freeze pickup pose while jaw closes.
    # ========================================================

    print(
        "Closing jaw while holding pickup pose..."
    )

    for _ in range(
        close_steps
    ):

        actions = (
            pickup_action
            .unsqueeze(0)
            .expand(
                args.trials,
                -1,
            )
        )

        obs, _, _, _ = env.step(
            actions
        )

        record_metrics(
            stage=1
        )

    # ========================================================
    # STEP D
    #
    # STATIC HOLD.
    #
    # DO NOT RISE.
    # Keep feeding the exact same pickup_action.
    # ========================================================

    hold_steps = int(
        math.ceil(
            args.hold_seconds
            / step_dt
        )
    )

    print(
        f"Holding closed-mouth pickup pose "
        f"for {args.hold_seconds:.2f}s..."
    )

    for _ in range(
        hold_steps
    ):

        actions = (
            pickup_action
            .unsqueeze(0)
            .expand(
                args.trials,
                -1,
            )
        )

        obs, _, _, _ = env.step(
            actions
        )

        record_metrics(
            stage=2
        )

    # ========================================================
    # Convert metrics
    # ========================================================

    grape_height = np.stack(
        grape_height_rows
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

    stages = np.stack(
        stage_rows
    )

    time_s = (
        np.arange(
            len(grape_height_rows),
            dtype=np.float32,
        )
        + 1.0
    ) * step_dt

    # ========================================================
    # Analyze ONLY static hold period.
    # ========================================================

    static_mask = (
        stages[:, 0]
        == 2
    )

    static_height = (
        grape_height[
            static_mask
        ]
    )

    static_distance = (
        mouth_distance[
            static_mask
        ]
    )

    static_velocity = (
        grape_vertical_velocity[
            static_mask
        ]
    )

    terrain_rest_height = (
        GRAPE_HALF_HEIGHT
    )

    final_height = (
        grape_height[-1]
    )

    final_distance = (
        mouth_distance[-1]
    )

    max_static_height = (
        np.max(
            static_height,
            axis=0,
        )
    )

    min_static_height = (
        np.min(
            static_height,
            axis=0,
        )
    )

    max_static_distance = (
        np.max(
            static_distance,
            axis=0,
        )
    )

    max_static_abs_vz = (
        np.max(
            np.abs(
                static_velocity
            ),
            axis=0,
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # This is only a heuristic grasp success metric.
    #
    # The strongest proof would eventually be:
    #
    # upper_pad_contact AND lower_pad_contact
    #
    # For now we ask:
    #
    # - does grape remain close to mouth?
    # - does it remain mechanically stable?
    #
    # We DO NOT require a high absolute grape height because
    # this is a ground-level static clamp test.
    # --------------------------------------------------------

    success = (
        max_static_distance
        <= args.distance_threshold
    ) & (
        max_static_abs_vz
        <= args.velocity_threshold
    )

    success_rate = float(
        np.mean(
            success
        )
    )

    # ========================================================
    # Plot
    # ========================================================

    _write_plot(
        output_dir / "static_grasp.png",
        time_s,
        grape_height,
        mouth_distance,
        grape_vertical_velocity,
        jaw_position,
        jaw_joint_name,
        terrain_rest_height,
    )

    # ========================================================
    # Trace CSV
    # ========================================================

    with (
        output_dir / "trace.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            (
                "trial",
                "time_s",
                "stage",
                "grape_height_m",
                "mouth_distance_m",
                "grape_vertical_velocity_mps",
                "jaw_position_rad",
            )
        )

        for step in range(
            len(time_s)
        ):
            for trial in range(
                args.trials
            ):

                writer.writerow(
                    (
                        trial,
                        float(
                            time_s[step]
                        ),
                        int(
                            stages[
                                step,
                                trial,
                            ]
                        ),
                        float(
                            grape_height[
                                step,
                                trial,
                            ]
                        ),
                        float(
                            mouth_distance[
                                step,
                                trial,
                            ]
                        ),
                        float(
                            grape_vertical_velocity[
                                step,
                                trial,
                            ]
                        ),
                        float(
                            jaw_position[
                                step,
                                trial,
                            ]
                        ),
                    )
                )

    # ========================================================
    # Per-trial results
    # ========================================================

    trial_rows = []

    for trial in range(
        args.trials
    ):
        trial_rows.append(
            {
                "trial": trial,

                "offset_x_m": float(
                    applied_offsets[
                        trial,
                        0,
                    ]
                ),

                "offset_y_m": float(
                    applied_offsets[
                        trial,
                        1,
                    ]
                ),

                "offset_z_m": float(
                    applied_offsets[
                        trial,
                        2,
                    ]
                ),

                "success": bool(
                    success[
                        trial
                    ]
                ),

                "final_height_m": float(
                    final_height[
                        trial
                    ]
                ),

                "final_distance_m": float(
                    final_distance[
                        trial
                    ]
                ),

                "max_static_height_m": float(
                    max_static_height[
                        trial
                    ]
                ),

                "min_static_height_m": float(
                    min_static_height[
                        trial
                    ]
                ),

                "max_static_distance_m": float(
                    max_static_distance[
                        trial
                    ]
                ),

                "max_static_abs_vz_mps": float(
                    max_static_abs_vz[
                        trial
                    ]
                ),
            }
        )

    with (
        output_dir
        / "trials.csv"
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
        writer.writerows(
            trial_rows
        )

    # ========================================================
    # Summary
    # ========================================================

    report = {
        "checkpoint": str(
            checkpoint
        ),

        "trials": args.trials,

        "jaw_joint_name": (
            jaw_joint_name
        ),

        "requested_offset_mm": [
            args.offset_x_mm,
            args.offset_y_mm,
            args.offset_z_mm,
        ],

        "hold_seconds": (
            args.hold_seconds
        ),

        "distance_threshold_m": (
            args.distance_threshold
        ),

        "velocity_threshold_mps": (
            args.velocity_threshold
        ),

        "success_rate": (
            success_rate
        ),

        "mean_final_height_m": float(
            np.mean(
                final_height
            )
        ),

        "mean_final_distance_m": float(
            np.mean(
                final_distance
            )
        ),

        "mean_max_static_distance_m": float(
            np.mean(
                max_static_distance
            )
        ),

        "mean_max_static_abs_vz_mps": float(
            np.mean(
                max_static_abs_vz
            )
        ),
    }

    (
        output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "=" * 70
    )

    print(
        "STATIC GRAPE GRASP RESULT"
    )

    print(
        "=" * 70
    )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print(
        f"Plot:    "
        f"{output_dir / 'static_grasp.png'}"
    )

    print(
        f"Trace:   "
        f"{output_dir / 'trace.csv'}"
    )

    print(
        f"Trials:  "
        f"{output_dir / 'trials.csv'}"
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

    # Start with the grape directly underneath the mouth-tip
    # projection rather than adding the previous +10 mm offset.
    parser.add_argument(
        "--offset-x-mm",
        type=float,
        default=0.0,
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
        "--hold-seconds",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--velocity-threshold",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--enable-dr",
        action="store_true",
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
        default=(
            "artifacts/"
            "grape-static-grasp"
        ),
    )

    args = parser.parse_args()

    if args.trials <= 0:
        parser.error(
            "--trials must be positive"
        )

    if args.hold_seconds <= 0:
        parser.error(
            "--hold-seconds must be positive"
        )

    if args.placement_noise_mm < 0:
        parser.error(
            "--placement-noise-mm "
            "must be non-negative"
        )

    return args


if __name__ == "__main__":
    run(
        _parse_args()
    )