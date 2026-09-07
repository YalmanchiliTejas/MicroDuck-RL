#!/usr/bin/env python3

"""End-to-end diagnostic for MicroDuck grape grasping.

This test answers ONE question:

    Can the mouth physically latch onto a grape on the floor
    and carry it upward?

Sequence
--------
1. Load trained GrapePick policy.
2. Record one deterministic policy trajectory.
3. Reset.
4. Replay only the learned descent.
5. Robot is now frozen in its pickup pose.
6. Place grape ON THE FLOOR at the XY midpoint between the
   upper and lower grip geoms.
7. Keep robot frozen until passive_mouth is ACTUALLY closed.
8. Record grip/grape geometry at closure.
9. Replay ONLY the learned rising motion.
10. Check whether grape rises with mouth.

NO reward function is used to decide success.

Success requires:
- jaw actually closed
- mouth actually rose
- grape actually rose
- grape followed a substantial fraction of mouth motion
- grape remained near mouth at the end

Diagnostics logged:
- grape height
- mouth_tip height
- grape vertical velocity
- mouth_tip -> grape distance
- upper grip center -> grape distance
- lower grip center -> grape distance
- upper/lower grip center gap
- jaw position

NOTE:
Geom distances are CENTER-to-CENTER distances.
They are useful for geometry debugging but are not surface-contact distances.

Example
-------
uv run scripts/test_grape_grasp_diagnostic.py \
    --checkpoint-file /path/to/model.pt \
    --trials 1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import (
    load_env_cfg,
    load_rl_cfg,
    load_runner_cls,
)
from mjlab.utils.torch import configure_torch_backends

import mjlab_microduck.tasks  # noqa: F401

from mjlab_microduck.tasks.microduck_grape_pick_env_cfg import (
    DESCENT_END,
    GP_PERIOD,
    GRAPE_HALF_HEIGHT,
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
# Environment
# ============================================================


def _configure_env(
    num_envs: int,
    enable_dr: bool,
    max_close_seconds: float,
):
    cfg = load_env_cfg(
        TASK_ID,
        play=True,
    )

    cfg.scene.num_envs = num_envs

    # Make sure the episode cannot auto-reset while we are
    # diagnosing the grasp.
    cfg.episode_length_s = (
        GP_PERIOD
        + max_close_seconds
        + 2.0
    )

    cfg.auto_reset = False

    # Nothing should terminate the diagnostic halfway through.
    cfg.terminations = {}
    cfg.curriculum = {}

    # Start choreography at phase zero.
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

        cfg.events["reset_grape"].params[
            "noise_xy"
        ] = 0.0

        # Remove observation noise.
        for group in cfg.observations.values():
            group.enable_corruption = False

        for name in (
            "base_ang_vel",
            "projected_gravity",
        ):
            term = (
                cfg.observations["actor"]
                .terms
                .get(name)
            )

            if (
                term is not None
                and "max_angle_deg" in term.params
            ):
                term.params[
                    "max_angle_deg"
                ] = 0.0

    return cfg


# ============================================================
# Robot element lookup
# ============================================================


def _find_jaw_joint(robot):
    """Find passive mouth joint."""

    candidates = (
        r"^passive_mouth$",
        r".*passive.*mouth.*",
        r".*jaw.*",
        r".*mouth.*",
        r".*beak.*",
    )

    for pattern in candidates:
        try:
            ids, names = robot.find_joints(
                pattern
            )

            if len(ids) > 0:
                return (
                    int(ids[0]),
                    str(names[0]),
                )

        except Exception:
            continue

    return None, None


def _relevant_geom_names(robot):
    """Return mouth-related geom names for debugging."""

    result = []

    for name in robot.geom_names:
        lower = str(name).lower()

        if any(
            word in lower
            for word in (
                "mouth",
                "jaw",
                "soft",
                "grip",
                "beak",
            )
        ):
            result.append(str(name))

    return result


def _resolve_geom(
    robot,
    requested_name: str | None,
    kind: str,
):
    """Resolve upper/lower grip geom.

    First uses explicitly requested name.

    Otherwise searches likely names.
    """

    if requested_name:
        try:
            ids, names = robot.find_geoms(
                f"^{re.escape(requested_name)}$"
            )

            if len(ids) > 0:
                return (
                    int(ids[0]),
                    str(names[0]),
                )

        except Exception:
            pass

        raise RuntimeError(
            f"Could not find requested {kind} grip geom "
            f"{requested_name!r}.\n"
            f"Relevant available geoms:\n"
            + "\n".join(
                f"  - {x}"
                for x in _relevant_geom_names(
                    robot
                )
            )
        )

    if kind == "upper":

        patterns = (
            r"^upper_mouth_grip$",
            r".*upper.*mouth.*grip.*",
            r".*upper.*grip.*",
            r".*soft_mouth_top.*",
            r".*mouth.*top.*",
        )

    else:

        patterns = (
            r"^lower_mouth_grip$",
            r".*lower.*mouth.*grip.*",
            r".*lower.*grip.*",
            r".*jaw_soft.*",
        )

    for pattern in patterns:

        try:
            ids, names = robot.find_geoms(
                pattern
            )

            if len(ids) > 0:
                return (
                    int(ids[0]),
                    str(names[0]),
                )

        except Exception:
            continue

    raise RuntimeError(
        f"Could not automatically find the "
        f"{kind} grip geom.\n\n"
        f"Relevant available geoms:\n"
        + "\n".join(
            f"  - {x}"
            for x in _relevant_geom_names(
                robot
            )
        )
        + "\n\n"
        + f"Rerun with:\n"
        + f"  --{kind}-grip-geom YOUR_GEOM_NAME"
    )


# ============================================================
# Grape placement
# ============================================================


def _place_grape_between_grips_on_floor(
    env: ManagerBasedRlEnv,
    upper_geom_id: int,
    lower_geom_id: int,
    offset_x_m: float,
    offset_y_m: float,
):
    """Place grape on floor at midpoint between grip centers.

    X/Y come from midpoint between upper and lower grip geoms.
    Z is forced to grape resting height on the terrain.

    Optional offsets are WORLD-frame X/Y adjustments.
    """

    robot = env.scene["robot"]
    grape = env.scene["grape"]

    env_ids = torch.arange(
        env.num_envs,
        device=env.device,
    )

    upper_pos = (
        robot.data.geom_pos_w[
            :, upper_geom_id, :
        ]
    )

    lower_pos = (
        robot.data.geom_pos_w[
            :, lower_geom_id, :
        ]
    )

    midpoint = (
        upper_pos + lower_pos
    ) * 0.5

    grape_pos = midpoint.clone()

    grape_pos[:, 0] += offset_x_m
    grape_pos[:, 1] += offset_y_m

    terrain_z = (
        env.scene
        .terrain
        .env_origins[:, 2]
    )

    # Most important part:
    #
    # grape starts resting ON THE FLOOR.
    grape_pos[:, 2] = (
        terrain_z
        + GRAPE_HALF_HEIGHT
    )

    dtype = grape_pos.dtype

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

    # Required before reading geom/site positions again.
    env.sim.forward()

    env.scene.update(
        dt=0.0
    )

    return (
        grape_pos.detach()
        .cpu()
        .numpy()
    )


# ============================================================
# Plot
# ============================================================


def _write_plot(
    path: Path,
    time_s: np.ndarray,
    grape_height: np.ndarray,
    mouth_height: np.ndarray,
    grape_vz: np.ndarray,
    mouth_distance: np.ndarray,
    upper_distance: np.ndarray,
    lower_distance: np.ndarray,
    pad_gap: np.ndarray,
    jaw_position: np.ndarray,
    event_times: dict[str, float],
):
    os.environ.setdefault(
        "MPLBACKEND",
        "Agg",
    )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(12, 15),
        sharex=True,
    )

    def mean_band(
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
            label=label,
        )

        if values.shape[1] > 1:
            ax.fill_between(
                time_s,
                low,
                high,
                alpha=0.15,
            )

    def events(ax):
        for name, t in event_times.items():

            ax.axvline(
                t,
                linestyle=":",
                linewidth=1,
                alpha=0.7,
            )

        ax.grid(
            alpha=0.25
        )

    # --------------------------------------------------------
    # 1. Did grape rise with mouth?
    # --------------------------------------------------------

    mean_band(
        axes[0],
        grape_height,
        "grape height",
    )

    mean_band(
        axes[0],
        mouth_height,
        "mouth-tip height",
    )

    axes[0].set_ylabel(
        "Height (m)"
    )

    axes[0].set_title(
        "Does the grape rise with the mouth?"
    )

    axes[0].legend()

    events(
        axes[0]
    )

    # --------------------------------------------------------
    # 2. Where is grape relative to mouth/grips?
    # --------------------------------------------------------

    mean_band(
        axes[1],
        mouth_distance,
        "mouth tip -> grape",
    )

    mean_band(
        axes[1],
        upper_distance,
        "upper grip center -> grape",
    )

    mean_band(
        axes[1],
        lower_distance,
        "lower grip center -> grape",
    )

    axes[1].set_ylabel(
        "Distance (m)"
    )

    axes[1].set_title(
        "Grape position relative to mouth geometry"
    )

    axes[1].legend()

    events(
        axes[1]
    )

    # --------------------------------------------------------
    # 3. Jaw motion
    # --------------------------------------------------------

    mean_band(
        axes[2],
        jaw_position,
        "passive_mouth",
    )

    axes[2].set_ylabel(
        "Jaw position (rad)"
    )

    axes[2].set_title(
        "Jaw closing"
    )

    axes[2].legend()

    events(
        axes[2]
    )

    # --------------------------------------------------------
    # 4. Pad center gap
    # --------------------------------------------------------

    mean_band(
        axes[3],
        pad_gap,
        "upper-lower grip center gap",
    )

    axes[3].set_ylabel(
        "Distance (m)"
    )

    axes[3].set_title(
        "Distance between grip centers"
    )

    axes[3].legend()

    events(
        axes[3]
    )

    # --------------------------------------------------------
    # 5. Grape velocity
    # --------------------------------------------------------

    mean_band(
        axes[4],
        grape_vz,
        "grape vertical velocity",
    )

    axes[4].axhline(
        0.0,
        linewidth=1,
    )

    axes[4].set_ylabel(
        "Velocity (m/s)"
    )

    axes[4].set_xlabel(
        "Time since grape placement (s)"
    )

    axes[4].set_title(
        "Grape vertical velocity"
    )

    axes[4].legend()

    events(
        axes[4]
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=170,
    )

    plt.close(
        fig
    )


# ============================================================
# Main diagnostic
# ============================================================


def run(
    args: argparse.Namespace,
):

    checkpoint = (
        Path(args.checkpoint_file)
        .expanduser()
        .resolve()
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            checkpoint
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
        args.max_close_seconds,
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

    policy = (
        runner.get_inference_policy(
            device=device
        )
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
    # Record one complete learned GrapePick motion.
    # ========================================================

    obs = env.get_observations()

    action_trace = []

    for _ in range(
        num_steps
    ):

        with torch.inference_mode():

            policy_action = policy(
                obs
            )

        reference_action = (
            policy_action[0]
            .detach()
            .clone()
        )

        action_trace.append(
            reference_action
        )

        actions = (
            reference_action
            .unsqueeze(0)
            .expand(
                args.trials,
                -1,
            )
        )

        obs, _, _, _ = env.step(
            actions
        )

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

    rise_actions = action_trace[
        rise_start_step:
        rise_end_step
    ]

    # ========================================================
    # PASS 2
    #
    # Actual grasp test.
    # ========================================================

    env.reset()

    robot = raw_env.scene[
        "robot"
    ]

    grape = raw_env.scene[
        "grape"
    ]

    # --------------------------------------------------------
    # Resolve mouth components
    # --------------------------------------------------------

    mouth_ids, mouth_names = (
        robot.find_sites(
            r"^mouth_tip$"
        )
    )

    if not mouth_ids:
        raise RuntimeError(
            "Could not find mouth_tip site."
        )

    mouth_id = int(
        mouth_ids[0]
    )

    jaw_id, jaw_name = (
        _find_jaw_joint(
            robot
        )
    )

    if jaw_id is None:
        raise RuntimeError(
            "Could not find passive_mouth jaw joint."
        )

    upper_id, upper_name = (
        _resolve_geom(
            robot,
            args.upper_grip_geom,
            "upper",
        )
    )

    lower_id, lower_name = (
        _resolve_geom(
            robot,
            args.lower_grip_geom,
            "lower",
        )
    )

    print()
    print("=" * 70)
    print("MOUTH COMPONENTS")
    print("=" * 70)

    print(
        f"mouth_tip:       {mouth_names[0]}"
    )

    print(
        f"jaw joint:       {jaw_name}"
    )

    print(
        f"upper grip geom: {upper_name}"
    )

    print(
        f"lower grip geom: {lower_name}"
    )

    # ========================================================
    # Step 1
    #
    # Bend down.
    #
    # We DO NOT record the grape during this period.
    #
    # The actual diagnostic clock begins only after placement.
    # ========================================================

    obs = env.get_observations()

    print()
    print(
        "1. Moving duck into pickup pose..."
    )

    for action in action_trace[
        :descent_step
    ]:

        actions = (
            action
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
    # Step 2
    #
    # Place grape on floor exactly underneath midpoint of pads.
    # ========================================================

    print(
        "2. Placing grape on floor between grip centers..."
    )

    placed_positions = (
        _place_grape_between_grips_on_floor(
            raw_env,
            upper_geom_id=upper_id,
            lower_geom_id=lower_id,
            offset_x_m=(
                args.offset_x_mm
                / 1000.0
            ),
            offset_y_m=(
                args.offset_y_mm
                / 1000.0
            ),
        )
    )

    # ========================================================
    # Metric recording begins NOW.
    # ========================================================

    grape_height_rows = []
    mouth_height_rows = []
    grape_vz_rows = []

    mouth_distance_rows = []
    upper_distance_rows = []
    lower_distance_rows = []

    pad_gap_rows = []
    jaw_position_rows = []

    stage_rows = []

    event_times = {
        "grape placed": 0.0,
    }

    current_step = 0

    terrain_z = (
        raw_env.scene
        .terrain
        .env_origins[:, 2]
    )

    ground_rest_height = (
        GRAPE_HALF_HEIGHT
    )

    def record(
        stage: int,
    ):
        nonlocal current_step

        grape_pos = (
            grape.data
            .root_link_pos_w
        )

        mouth_pos = (
            robot.data
            .site_pos_w[
                :, mouth_id, :
            ]
        )

        upper_pos = (
            robot.data
            .geom_pos_w[
                :, upper_id, :
            ]
        )

        lower_pos = (
            robot.data
            .geom_pos_w[
                :, lower_id, :
            ]
        )

        grape_height_t = (
            grape_pos[:, 2]
            - terrain_z
        )

        mouth_height_t = (
            mouth_pos[:, 2]
            - terrain_z
        )

        grape_vz_t = (
            grape.data
            .root_link_lin_vel_w[
                :, 2
            ]
        )

        mouth_distance_t = (
            torch.linalg.vector_norm(
                mouth_pos
                - grape_pos,
                dim=-1,
            )
        )

        upper_distance_t = (
            torch.linalg.vector_norm(
                upper_pos
                - grape_pos,
                dim=-1,
            )
        )

        lower_distance_t = (
            torch.linalg.vector_norm(
                lower_pos
                - grape_pos,
                dim=-1,
            )
        )

        pad_gap_t = (
            torch.linalg.vector_norm(
                upper_pos
                - lower_pos,
                dim=-1,
            )
        )

        jaw_t = (
            robot.data
            .joint_pos[
                :, jaw_id
            ]
        )

        grape_height_rows.append(
            grape_height_t
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        mouth_height_rows.append(
            mouth_height_t
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        grape_vz_rows.append(
            grape_vz_t
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

        upper_distance_rows.append(
            upper_distance_t
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        lower_distance_rows.append(
            lower_distance_t
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        pad_gap_rows.append(
            pad_gap_t
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

        current_step += 1

    # Record exact placement state.
    record(
        stage=0
    )

    # ========================================================
    # Step 3
    #
    # HOLD pickup pose until jaw is PHYSICALLY CLOSED.
    #
    # We no longer assume HOLD_END means jaw has actually
    # reached its closed position.
    # ========================================================

    print(
        "3. Holding pickup pose until jaw closes..."
    )

    max_close_steps = int(
        math.ceil(
            args.max_close_seconds
            / step_dt
        )
    )

    closed_streak = 0
    jaw_closed = False

    close_steps_used = 0

    for _ in range(
        max_close_steps
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

        record(
            stage=1
        )

        close_steps_used += 1

        jaw_value = float(
            torch.mean(
                robot.data
                .joint_pos[
                    :, jaw_id
                ]
            ).item()
        )

        if (
            jaw_value
            <= args.jaw_closed_threshold
        ):
            closed_streak += 1
        else:
            closed_streak = 0

        if (
            closed_streak
            >= args.jaw_closed_stable_steps
        ):
            jaw_closed = True
            break

    jaw_closed_time = (
        current_step
        * step_dt
    )

    event_times[
        "jaw closed"
    ] = jaw_closed_time

    print(
        f"   jaw closed: {jaw_closed}"
    )

    print(
        f"   close time: "
        f"{close_steps_used * step_dt:.3f}s"
    )

    print(
        f"   jaw position: "
        f"{float(torch.mean(robot.data.joint_pos[:, jaw_id])):.4f} rad"
    )

    # ========================================================
    # Save state AT closure.
    # ========================================================

    latch_grape_height = (
        grape_height_rows[-1]
    )

    latch_mouth_distance = (
        mouth_distance_rows[-1]
    )

    latch_upper_distance = (
        upper_distance_rows[-1]
    )

    latch_lower_distance = (
        lower_distance_rows[-1]
    )

    latch_pad_gap = (
        pad_gap_rows[-1]
    )

    # ========================================================
    # Step 4
    #
    # Start lift.
    #
    # This is the decisive test.
    # ========================================================

    lift_start_time = (
        current_step
        * step_dt
    )

    event_times[
        "lift start"
    ] = lift_start_time

    print(
        "4. Replaying learned rising motion..."
    )

    lift_start_grape_height = (
        grape_height_rows[-1]
        .copy()
    )

    lift_start_mouth_height = (
        mouth_height_rows[-1]
        .copy()
    )

    for action in rise_actions:

        actions = (
            action
            .unsqueeze(0)
            .expand(
                args.trials,
                -1,
            )
        )

        obs, _, _, _ = env.step(
            actions
        )

        record(
            stage=2
        )

    lift_end_time = (
        current_step
        * step_dt
    )

    event_times[
        "lift end"
    ] = lift_end_time

    # ========================================================
    # Convert arrays
    # ========================================================

    grape_height = np.stack(
        grape_height_rows
    )

    mouth_height = np.stack(
        mouth_height_rows
    )

    grape_vz = np.stack(
        grape_vz_rows
    )

    mouth_distance = np.stack(
        mouth_distance_rows
    )

    upper_distance = np.stack(
        upper_distance_rows
    )

    lower_distance = np.stack(
        lower_distance_rows
    )

    pad_gap = np.stack(
        pad_gap_rows
    )

    jaw_position = np.stack(
        jaw_position_rows
    )

    stages = np.stack(
        stage_rows
    )

    time_s = (
        np.arange(
            len(
                grape_height_rows
            ),
            dtype=np.float32,
        )
        * step_dt
    )

    # ========================================================
    # Real physical success metrics
    # ========================================================

    final_grape_height = (
        grape_height[-1]
    )

    final_mouth_height = (
        mouth_height[-1]
    )

    final_mouth_distance = (
        mouth_distance[-1]
    )

    final_upper_distance = (
        upper_distance[-1]
    )

    final_lower_distance = (
        lower_distance[-1]
    )

    grape_lift_during_rise = (
        final_grape_height
        - lift_start_grape_height
    )

    mouth_lift_during_rise = (
        final_mouth_height
        - lift_start_mouth_height
    )

    grape_height_above_ground = (
        final_grape_height
        - ground_rest_height
    )

    # If grape is attached, it should rise by roughly
    # the same amount as the mouth.
    follow_ratio = (
        grape_lift_during_rise
        / np.maximum(
            mouth_lift_during_rise,
            1e-6,
        )
    )

    mouth_actually_rose = (
        mouth_lift_during_rise
        >= args.min_mouth_lift
    )

    grape_actually_rose = (
        grape_height_above_ground
        >= args.min_grape_lift
    )

    grape_followed_mouth = (
        follow_ratio
        >= args.min_follow_ratio
    )

    grape_still_near_mouth = (
        final_mouth_distance
        <= args.max_final_mouth_distance
    )

    # THIS is the actual success definition.
    #
    # A grape sitting on the floor can no longer count.
    success = (
        jaw_closed
        & mouth_actually_rose
        & grape_actually_rose
        & grape_followed_mouth
        & grape_still_near_mouth
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
        output_dir
        / "grasp_diagnostic.png",
        time_s,
        grape_height,
        mouth_height,
        grape_vz,
        mouth_distance,
        upper_distance,
        lower_distance,
        pad_gap,
        jaw_position,
        event_times,
    )

    # ========================================================
    # Full trace CSV
    # ========================================================

    with (
        output_dir
        / "trace.csv"
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
                "mouth_height_m",
                "grape_vz_mps",
                "mouth_grape_distance_m",
                "upper_grip_grape_center_distance_m",
                "lower_grip_grape_center_distance_m",
                "upper_lower_grip_center_gap_m",
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
                            mouth_height[
                                step,
                                trial,
                            ]
                        ),
                        float(
                            grape_vz[
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
                            upper_distance[
                                step,
                                trial,
                            ]
                        ),
                        float(
                            lower_distance[
                                step,
                                trial,
                            ]
                        ),
                        float(
                            pad_gap[
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
    # Per-trial report
    # ========================================================

    trial_rows = []

    for trial in range(
        args.trials
    ):

        trial_rows.append(
            {
                "trial": trial,

                "success": bool(
                    success[trial]
                ),

                "placed_grape_x_m": float(
                    placed_positions[
                        trial, 0
                    ]
                ),

                "placed_grape_y_m": float(
                    placed_positions[
                        trial, 1
                    ]
                ),

                "placed_grape_z_m": float(
                    placed_positions[
                        trial, 2
                    ]
                ),

                "latch_grape_height_m": float(
                    latch_grape_height[
                        trial
                    ]
                ),

                "latch_mouth_distance_m": float(
                    latch_mouth_distance[
                        trial
                    ]
                ),

                "latch_upper_grip_distance_m": float(
                    latch_upper_distance[
                        trial
                    ]
                ),

                "latch_lower_grip_distance_m": float(
                    latch_lower_distance[
                        trial
                    ]
                ),

                "latch_pad_center_gap_m": float(
                    latch_pad_gap[
                        trial
                    ]
                ),

                "mouth_lift_m": float(
                    mouth_lift_during_rise[
                        trial
                    ]
                ),

                "grape_lift_during_rise_m": float(
                    grape_lift_during_rise[
                        trial
                    ]
                ),

                "final_grape_height_m": float(
                    final_grape_height[
                        trial
                    ]
                ),

                "grape_height_above_ground_m": float(
                    grape_height_above_ground[
                        trial
                    ]
                ),

                "follow_ratio": float(
                    follow_ratio[
                        trial
                    ]
                ),

                "final_mouth_distance_m": float(
                    final_mouth_distance[
                        trial
                    ]
                ),

                "final_upper_grip_distance_m": float(
                    final_upper_distance[
                        trial
                    ]
                ),

                "final_lower_grip_distance_m": float(
                    final_lower_distance[
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
    # Human-readable summary
    # ========================================================

    report = {

        "checkpoint": str(
            checkpoint
        ),

        "trials": (
            args.trials
        ),

        "geometry": {

            "upper_grip_geom": (
                upper_name
            ),

            "lower_grip_geom": (
                lower_name
            ),

            "jaw_joint": (
                jaw_name
            ),

            "mouth_site": (
                str(
                    mouth_names[0]
                )
            ),
        },

        "placement": {

            "description": (
                "grape placed on floor at XY midpoint "
                "between upper/lower grip geom centers"
            ),

            "offset_x_mm": (
                args.offset_x_mm
            ),

            "offset_y_mm": (
                args.offset_y_mm
            ),
        },

        "jaw": {

            "closed": (
                jaw_closed
            ),

            "closed_threshold_rad": (
                args.jaw_closed_threshold
            ),

            "close_time_s": (
                close_steps_used
                * step_dt
            ),

            "final_jaw_position_rad": float(
                np.mean(
                    jaw_position[-1]
                )
            ),
        },

        "at_latch": {

            "mean_grape_height_m": float(
                np.mean(
                    latch_grape_height
                )
            ),

            "mean_mouth_grape_distance_m": float(
                np.mean(
                    latch_mouth_distance
                )
            ),

            "mean_upper_grip_grape_distance_m": float(
                np.mean(
                    latch_upper_distance
                )
            ),

            "mean_lower_grip_grape_distance_m": float(
                np.mean(
                    latch_lower_distance
                )
            ),

            "mean_pad_center_gap_m": float(
                np.mean(
                    latch_pad_gap
                )
            ),
        },

        "lift": {

            "mean_mouth_lift_m": float(
                np.mean(
                    mouth_lift_during_rise
                )
            ),

            "mean_grape_lift_during_rise_m": float(
                np.mean(
                    grape_lift_during_rise
                )
            ),

            "mean_grape_height_above_ground_m": float(
                np.mean(
                    grape_height_above_ground
                )
            ),

            "mean_follow_ratio": float(
                np.mean(
                    follow_ratio
                )
            ),

            "mean_final_mouth_distance_m": float(
                np.mean(
                    final_mouth_distance
                )
            ),
        },

        "success_checks": {

            "jaw_closed": (
                bool(
                    jaw_closed
                )
            ),

            "mouth_actually_rose_rate": float(
                np.mean(
                    mouth_actually_rose
                )
            ),

            "grape_actually_rose_rate": float(
                np.mean(
                    grape_actually_rose
                )
            ),

            "grape_followed_mouth_rate": float(
                np.mean(
                    grape_followed_mouth
                )
            ),

            "grape_still_near_mouth_rate": float(
                np.mean(
                    grape_still_near_mouth
                )
            ),
        },

        "success_rate": (
            success_rate
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
    print("=" * 70)
    print("GRAPE GRASP DIAGNOSTIC")
    print("=" * 70)

    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print(
        f"Plot:   "
        f"{output_dir / 'grasp_diagnostic.png'}"
    )

    print(
        f"Trace:  "
        f"{output_dir / 'trace.csv'}"
    )

    print(
        f"Trials: "
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
        default=1,
    )

    # If auto-detection works, leave these unset.
    #
    # Otherwise pass e.g.
    #
    # --upper-grip-geom upper_mouth_grip
    # --lower-grip-geom lower_mouth_grip

    parser.add_argument(
        "--upper-grip-geom",
        default=None,
    )

    parser.add_argument(
        "--lower-grip-geom",
        default=None,
    )

    # Optional WORLD XY correction from midpoint.
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

    # Based on your previous plot:
    #
    # open ~= +0.50 rad
    # closed ~= -0.08 rad
    #
    # So <= 0 is a reasonable "physically closed" criterion.
    parser.add_argument(
        "--jaw-closed-threshold",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--jaw-closed-stable-steps",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-close-seconds",
        type=float,
        default=1.2,
    )

    # --------------------------------------------------------
    # TRUE success thresholds
    # --------------------------------------------------------

    # The duck's mouth must actually rise.
    parser.add_argument(
        "--min-mouth-lift",
        type=float,
        default=0.03,
    )

    # Grape must end at least this far above ground.
    parser.add_argument(
        "--min-grape-lift",
        type=float,
        default=0.03,
    )

    # Grape must follow at least this fraction of mouth rise.
    parser.add_argument(
        "--min-follow-ratio",
        type=float,
        default=0.60,
    )

    # At end, grape should still be close to mouth.
    parser.add_argument(
        "--max-final-mouth-distance",
        type=float,
        default=0.04,
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
            "grape-grasp-diagnostic"
        ),
    )

    args = parser.parse_args()

    if args.trials <= 0:
        parser.error(
            "--trials must be positive"
        )

    if args.max_close_seconds <= 0:
        parser.error(
            "--max-close-seconds must be positive"
        )

    return args


if __name__ == "__main__":
    run(
        _parse_args()
    )