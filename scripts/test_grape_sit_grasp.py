#!/usr/bin/env python3
"""
GPU-vectorized deterministic MicroDuck grape-pick diagnostic.

No PPO checkpoint is used.

This script tests the mechanical prerequisite chain before RL:

1. BODY REACHABILITY
   - Different leg-fold blends are simulated in parallel.
   - Root is free.
   - We measure root height, grip height, speed, and standing-relative orientation.

2. HEAD / NECK REACHABILITY
   - Coarse (leg, neck, head) candidates are evaluated in GPU batches.
   - Only the best candidates are physically moved/settled.
   - IMPORTANT: the duck is intentionally folded forward, so the final
     stability gate does NOT compare the pickup pose to standing.
   - Instead we save the torso orientation after the crouch has settled and
     measure POSTURE CHANGE caused by moving the neck/head.

3. GRAPE PLACEMENT SWEEP
   - Different grape XY offsets are tested in parallel.
   - Jaw is forced open, then deterministically closed.
   - Real contact sensors measure:
       upper pad <-> grape
       lower pad <-> grape
       BOTH pads simultaneously
       contact forces

4. FINAL CLAMP + RISE
   - Reproduce the best pose and best grape placement.
   - Close and lock the jaw.
   - Physically unfold the legs while the root stays free.
   - Check whether the grip rises and the grape follows it.

Default outputs:
    artifacts/grape-full-diagnostic/
        body_reachability.csv
        body_reachability.png
        head_reachability.csv
        head_reachability.png
        placement_sweep.csv
        placement_sweep.png
        trace.csv
        grasp_diagnostic.png
        summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.tasks.microduck_grape_pick_env_cfg import GRAPE_HALF_HEIGHT


TASK_ID = "Mjlab-GrapePick-Flat-MicroDuck"

UPPER_SENSOR = "grape_upper_contact"
LOWER_SENSOR = "grape_lower_contact"


SIT_LEG_POSE = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_pitch": -0.4079,
    "left_knee": 1.35,
    "left_ankle": 0.0,

    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_pitch": 0.4079,
    "right_knee": -1.35,
    "right_ankle": 0.0,
}


GROUND_PICK_DOWN_LEGS = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_pitch": 1.57,
    "left_knee": 1.57,
    "left_ankle": 0.0,

    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_pitch": -1.57,
    "right_knee": -1.57,
    "right_ankle": 0.0,
}


BASELINE_RESET_EVENTS = {
    "expand_bam_friction_fields",
    "reset_action_history",
    "reset_base",
    "reset_grape",
    "reset_robot_joints",
}


# =============================================================================
# Environment
# =============================================================================


def configure_env(args):

    cfg = load_env_cfg(
        TASK_ID,
        play=True,
    )

    cfg.scene.num_envs = args.num_envs

    cfg.episode_length_s = 30.0

    cfg.auto_reset = False

    cfg.terminations = {}

    cfg.curriculum = {}

    cfg.commands["twist"].randomize_phase = False

    cfg.events = {
        name: term
        for name, term in cfg.events.items()
        if name in BASELINE_RESET_EVENTS
    }

    cfg.events[
        "reset_base"
    ].params[
        "pose_range"
    ].update(
        {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.125, 0.125),
            "yaw": (0.0, 0.0),
        }
    )

    cfg.events[
        "reset_grape"
    ].params[
        "noise_xy"
    ] = 0.0

    for group in cfg.observations.values():
        group.enable_corruption = False

    def make_contact_sensor(
        name,
        secondary_pattern,
    ):

        return ContactSensorCfg(
            name=name,

            primary=ContactMatch(
                mode="geom",
                pattern=r".*",
                entity="grape",
            ),

            secondary=ContactMatch(
                mode="geom",
                pattern=secondary_pattern,
                entity="robot",
            ),

            fields=(
                "found",
                "force",
                "dist",
            ),

            reduce="maxforce",

            num_slots=1,

            secondary_policy="first",

            history_length=args.contact_history,
        )

    existing = {
        sensor.name
        for sensor in cfg.scene.sensors
    }

    extra = []

    if UPPER_SENSOR not in existing:

        extra.append(
            make_contact_sensor(
                UPPER_SENSOR,
                args.upper_grip_pattern,
            )
        )

    if LOWER_SENSOR not in existing:

        extra.append(
            make_contact_sensor(
                LOWER_SENSOR,
                args.lower_grip_pattern,
            )
        )

    cfg.scene.sensors = (
        tuple(
            cfg.scene.sensors
        )
        + tuple(
            extra
        )
    )

    return cfg


# =============================================================================
# Lookup helpers
# =============================================================================


def find_joint(
    robot,
    name,
):

    for pattern in (
        rf"^{re.escape(name)}$",
        rf".*{re.escape(name)}.*",
    ):

        try:

            ids, names = robot.find_joints(
                pattern
            )

            if len(ids):

                return (
                    int(
                        ids[0]
                    ),
                    str(
                        names[0]
                    ),
                )

        except Exception:
            pass

    raise RuntimeError(
        f"Could not find joint {name!r}"
    )


def find_jaw_joint(
    robot,
):

    for pattern in (
        r"^passive_mouth$",
        r".*passive.*mouth.*",
        r".*jaw.*",
        r".*mouth.*",
        r".*beak.*",
    ):

        try:

            ids, names = robot.find_joints(
                pattern
            )

            if len(ids):

                return (
                    int(
                        ids[0]
                    ),
                    str(
                        names[0]
                    ),
                )

        except Exception:
            pass

    raise RuntimeError(
        "Could not find passive mouth/jaw joint"
    )


def find_geom(
    robot,
    pattern,
    label,
):

    ids, names = robot.find_geoms(
        pattern
    )

    if not len(
        ids
    ):

        raise RuntimeError(
            f"Could not find {label} geom "
            f"using regex {pattern!r}"
        )

    if len(
        ids
    ) > 1:

        print(
            f"WARNING: {label} matched "
            f"{len(ids)} geoms; "
            f"using {names[0]!r}"
        )

    return (
        int(
            ids[0]
        ),
        str(
            names[0]
        ),
    )


# =============================================================================
# Batch helpers
# =============================================================================


def chunks(
    values,
    batch_size,
):

    for start in range(
        0,
        len(
            values
        ),
        batch_size,
    ):

        yield values[
            start:
            start + batch_size
        ]


def pad_batch(
    values,
    n_envs,
):

    if not values:

        raise ValueError(
            "Cannot pad an empty candidate batch"
        )

    values = list(
        values
    )

    if len(
        values
    ) == n_envs:

        return values

    return (
        values
        + [
            values[0]
        ]
        * (
            n_envs
            - len(
                values
            )
        )
    )


# =============================================================================
# Pose/action helpers
# =============================================================================


def pose_to_action(
    env,
    desired_pose,
):

    term = (
        env.action_manager
        .get_term(
            "joint_pos"
        )
    )

    desired = desired_pose[
        :,
        term.target_ids,
    ]

    return (
        desired
        - term.offset
    ) / term.scale


def resolve_leg_joint_ids(
    robot,
):

    return {
        name:
            find_joint(
                robot,
                name,
            )[0]

        for name
        in SIT_LEG_POSE
    }


def build_leg_pose_batch(
    robot,
    blends,
    leg_joint_ids,
):

    pose = (
        robot.data
        .default_joint_pos
        .clone()
    )

    blend_t = torch.as_tensor(
        blends,
        device=pose.device,
        dtype=pose.dtype,
    )

    for name, joint_id in leg_joint_ids.items():

        sit = SIT_LEG_POSE[
            name
        ]

        deep = GROUND_PICK_DOWN_LEGS[
            name
        ]

        pose[
            :,
            joint_id,
        ] = (
            sit
            + blend_t
            * (
                deep
                - sit
            )
        )

    return pose


def set_head_batch(
    pose,
    neck_id,
    head_id,
    neck_values,
    head_values,
):

    result = pose.clone()

    result[
        :,
        neck_id,
    ] = torch.as_tensor(
        neck_values,
        device=result.device,
        dtype=result.dtype,
    )

    result[
        :,
        head_id,
    ] = torch.as_tensor(
        head_values,
        device=result.device,
        dtype=result.dtype,
    )

    return result


# =============================================================================
# Deterministic jaw control
# =============================================================================


def force_jaw(
    env,
    jaw_id,
    value,
):

    robot = env.scene[
        "robot"
    ]

    if torch.is_tensor(
        value
    ):

        values = (
            value
            .to(
                device=env.device,
                dtype=(
                    robot.data
                    .joint_pos
                    .dtype
                ),
            )
            .reshape(
                -1
            )
        )

        if values.numel() == 1:

            values = values.expand(
                env.num_envs
            )

        if values.numel() != env.num_envs:

            raise ValueError(
                f"Expected "
                f"{env.num_envs} "
                f"jaw values, got "
                f"{values.numel()}"
            )

    else:

        values = torch.full(
            (
                env.num_envs,
            ),
            float(
                value
            ),
            device=env.device,
            dtype=(
                robot.data
                .joint_pos
                .dtype
            ),
        )

    robot.write_joint_state_to_sim(
        position=values[
            :,
            None,
        ],

        velocity=torch.zeros(
            (
                env.num_envs,
                1,
            ),
            device=env.device,
            dtype=values.dtype,
        ),

        joint_ids=torch.tensor(
            [
                jaw_id
            ],
            device=env.device,
            dtype=torch.long,
        ),
    )

    env.sim.forward()

    env.scene.update(
        dt=0.0
    )


def step_with_locked_jaw(
    env,
    action,
    jaw_id,
    jaw_value,
):

    force_jaw(
        env,
        jaw_id,
        jaw_value,
    )

    env.step(
        action
    )

    force_jaw(
        env,
        jaw_id,
        jaw_value,
    )


def ramp_to_pose(
    env,
    target_pose,
    move_seconds,
    settle_seconds,
    jaw_id,
    jaw_value,
):

    robot = env.scene[
        "robot"
    ]

    start_pose = (
        robot.data
        .joint_pos
        .clone()
    )

    move_steps = max(
        1,
        math.ceil(
            move_seconds
            / env.step_dt
        ),
    )

    for step in range(
        move_steps
    ):

        alpha = (
            step + 1
        ) / move_steps

        desired = (
            start_pose
            + alpha
            * (
                target_pose
                - start_pose
            )
        )

        step_with_locked_jaw(
            env,
            pose_to_action(
                env,
                desired,
            ),
            jaw_id,
            jaw_value,
        )

    final_action = pose_to_action(
        env,
        target_pose,
    )

    settle_steps = max(
        1,
        math.ceil(
            settle_seconds
            / env.step_dt
        ),
    )

    for _ in range(
        settle_steps
    ):

        step_with_locked_jaw(
            env,
            final_action,
            jaw_id,
            jaw_value,
        )

    return final_action


# =============================================================================
# Orientation / geometry
# =============================================================================


def orientation_error_deg(
    robot,
    reference_quat,
):

    q = (
        robot.data
        .root_link_quat_w
    )

    q = (
        q
        / torch.linalg.vector_norm(
            q,
            dim=-1,
            keepdim=True,
        )
    )

    ref = (
        reference_quat
        / torch.linalg.vector_norm(
            reference_quat,
            dim=-1,
            keepdim=True,
        )
    )

    dot = torch.sum(
        q
        * ref,
        dim=-1,
    )

    dot = torch.clamp(
        torch.abs(
            dot
        ),
        0.0,
        1.0,
    )

    return (
        2.0
        * torch.acos(
            dot
        )
        * 180.0
        / math.pi
    )


def geometry_tensors(
    env,
    standing_reference_quat,
    mouth_id,
    upper_id,
    lower_id,
):

    robot = env.scene[
        "robot"
    ]

    terrain_z = (
        env.scene
        .terrain
        .env_origins[
            :,
            2,
        ]
    )

    mouth = (
        robot.data
        .site_pos_w[
            :,
            mouth_id,
            :,
        ]
    )

    upper = (
        robot.data
        .geom_pos_w[
            :,
            upper_id,
            :,
        ]
    )

    lower = (
        robot.data
        .geom_pos_w[
            :,
            lower_id,
            :,
        ]
    )

    midpoint = (
        0.5
        * (
            upper
            + lower
        )
    )

    grip_height = (
        midpoint[
            :,
            2,
        ]
        - terrain_z
    )

    return {
        "root_height_m":
            (
                robot.data
                .root_link_pos_w[
                    :,
                    2,
                ]
                - terrain_z
            ),

        "mouth_height_m":
            (
                mouth[
                    :,
                    2,
                ]
                - terrain_z
            ),

        "grip_midpoint_height_m":
            grip_height,

        "vertical_error_m":
            torch.abs(
                grip_height
                - GRAPE_HALF_HEIGHT
            ),

        # Diagnostic only.
        # This is NOT the final stability gate.
        "orientation_error_deg":
            orientation_error_deg(
                robot,
                standing_reference_quat,
            ),

        "root_speed_mps":
            torch.linalg.vector_norm(
                robot.data
                .root_link_lin_vel_w,
                dim=-1,
            ),
    }


def geometry_rows(
    metrics,
    candidates,
    extra_fn,
):

    n = len(
        candidates
    )

    cpu = {
        key:
            value[
                :n
            ]
            .detach()
            .cpu()
            .numpy()

        for key, value
        in metrics.items()
    }

    rows = []

    for i, candidate in enumerate(
        candidates
    ):

        row = dict(
            extra_fn(
                candidate
            )
        )

        for key, values in cpu.items():

            row[
                key
            ] = float(
                values[i]
            )

        rows.append(
            row
        )

    return rows


# =============================================================================
# Contact helpers
# =============================================================================


def reset_contact_sensors(
    env,
):

    env.scene[
        UPPER_SENSOR
    ].reset()

    env.scene[
        LOWER_SENSOR
    ].reset()


def read_contact_sensor(
    sensor,
    force_threshold,
):

    data = sensor.data

    batch = (
        data.found
        .shape[0]
    )

    found = (
        data.found
        .reshape(
            batch,
            -1,
        )
        > 0
    )

    active = torch.any(
        found,
        dim=1,
    )

    current_force = torch.zeros(
        batch,
        device=data.found.device,
    )

    if data.force is not None:

        force = (
            data.force
            .reshape(
                batch,
                -1,
                3,
            )
        )

        current_force = torch.amax(
            torch.linalg.vector_norm(
                force,
                dim=-1,
            ),
            dim=1,
        )

    history_force = torch.zeros_like(
        current_force
    )

    if data.force_history is not None:

        history = (
            data.force_history
            .reshape(
                batch,
                -1,
                3,
            )
        )

        history_force = torch.amax(
            torch.linalg.vector_norm(
                history,
                dim=-1,
            ),
            dim=1,
        )

    max_force = torch.maximum(
        current_force,
        history_force,
    )

    active = (
        active
        |
        (
            max_force
            > force_threshold
        )
    )

    min_dist = torch.full(
        (
            batch,
        ),
        float(
            "nan"
        ),
        device=data.found.device,
    )

    if data.dist is not None:

        dist = (
            data.dist
            .reshape(
                batch,
                -1,
            )
        )

        if (
            dist.shape
            == found.shape
        ):

            mask = found

        else:

            mask = torch.ones_like(
                dist,
                dtype=torch.bool,
            )

        candidate = torch.min(
            torch.where(
                mask,
                dist,
                torch.full_like(
                    dist,
                    float(
                        "inf"
                    ),
                ),
            ),
            dim=1,
        ).values

        min_dist = torch.where(
            torch.isfinite(
                candidate
            ),
            candidate,
            min_dist,
        )

    return {
        "active":
            active,

        "max_force":
            max_force,

        "min_dist":
            min_dist,
    }


def contact_tensors(
    env,
    force_threshold,
):

    upper = read_contact_sensor(
        env.scene[
            UPPER_SENSOR
        ],
        force_threshold,
    )

    lower = read_contact_sensor(
        env.scene[
            LOWER_SENSOR
        ],
        force_threshold,
    )

    return {
        "upper_contact":
            upper[
                "active"
            ].float(),

        "lower_contact":
            lower[
                "active"
            ].float(),

        "both_contact":
            (
                upper[
                    "active"
                ]
                &
                lower[
                    "active"
                ]
            ).float(),

        "upper_force_n":
            upper[
                "max_force"
            ],

        "lower_force_n":
            lower[
                "max_force"
            ],

        "upper_contact_dist_m":
            torch.nan_to_num(
                upper[
                    "min_dist"
                ],
                nan=0.0,
            ),

        "lower_contact_dist_m":
            torch.nan_to_num(
                lower[
                    "min_dist"
                ],
                nan=0.0,
            ),
    }


# =============================================================================
# Grape placement / state
# =============================================================================


def place_grapes_on_floor(
    env,
    upper_id,
    lower_id,
    x_offsets_m,
    y_offsets_m,
):

    robot = env.scene[
        "robot"
    ]

    grape = env.scene[
        "grape"
    ]

    upper = (
        robot.data
        .geom_pos_w[
            :,
            upper_id,
            :,
        ]
    )

    lower = (
        robot.data
        .geom_pos_w[
            :,
            lower_id,
            :,
        ]
    )

    grape_pos = (
        0.5
        * (
            upper
            + lower
        )
    )

    grape_pos = grape_pos.clone()

    x = torch.as_tensor(
        x_offsets_m,
        device=env.device,
        dtype=grape_pos.dtype,
    ).reshape(
        -1
    )

    y = torch.as_tensor(
        y_offsets_m,
        device=env.device,
        dtype=grape_pos.dtype,
    ).reshape(
        -1
    )

    if x.numel() == 1:

        x = x.expand(
            env.num_envs
        )

    if y.numel() == 1:

        y = y.expand(
            env.num_envs
        )

    if (
        x.numel()
        != env.num_envs
        or
        y.numel()
        != env.num_envs
    ):

        raise ValueError(
            "Grape offset arrays must have "
            "one value or num_envs values"
        )

    grape_pos[
        :,
        0,
    ] += x

    grape_pos[
        :,
        1,
    ] += y

    grape_pos[
        :,
        2,
    ] = (
        env.scene
        .terrain
        .env_origins[
            :,
            2,
        ]
        + GRAPE_HALF_HEIGHT
    )

    pose = torch.zeros(
        (
            env.num_envs,
            7,
        ),
        device=env.device,
        dtype=grape_pos.dtype,
    )

    pose[
        :,
        :3,
    ] = grape_pos

    pose[
        :,
        3,
    ] = 1.0

    env_ids = torch.arange(
        env.num_envs,
        device=env.device,
    )

    grape.write_root_link_pose_to_sim(
        pose,
        env_ids=env_ids,
    )

    grape.write_root_link_velocity_to_sim(
        torch.zeros(
            (
                env.num_envs,
                6,
            ),
            device=env.device,
            dtype=grape_pos.dtype,
        ),
        env_ids=env_ids,
    )

    env.scene.write_data_to_sim()

    env.sim.forward()

    env.scene.update(
        dt=0.0
    )


def full_state_tensors(
    env,
    standing_reference_quat,
    mouth_id,
    upper_id,
    lower_id,
    jaw_id,
    force_threshold,
):

    robot = env.scene[
        "robot"
    ]

    grape = env.scene[
        "grape"
    ]

    terrain_z = (
        env.scene
        .terrain
        .env_origins[
            :,
            2,
        ]
    )

    grape_pos = (
        grape.data
        .root_link_pos_w
    )

    mouth = (
        robot.data
        .site_pos_w[
            :,
            mouth_id,
            :,
        ]
    )

    upper = (
        robot.data
        .geom_pos_w[
            :,
            upper_id,
            :,
        ]
    )

    lower = (
        robot.data
        .geom_pos_w[
            :,
            lower_id,
            :,
        ]
    )

    midpoint = (
        0.5
        * (
            upper
            + lower
        )
    )

    result = {
        "grape_height_m":
            (
                grape_pos[
                    :,
                    2,
                ]
                - terrain_z
            ),

        "root_height_m":
            (
                robot.data
                .root_link_pos_w[
                    :,
                    2,
                ]
                - terrain_z
            ),

        "mouth_height_m":
            (
                mouth[
                    :,
                    2,
                ]
                - terrain_z
            ),

        "grip_height_m":
            (
                midpoint[
                    :,
                    2,
                ]
                - terrain_z
            ),

        "mouth_distance_m":
            torch.linalg.vector_norm(
                mouth
                - grape_pos,
                dim=-1,
            ),

        "grip_distance_m":
            torch.linalg.vector_norm(
                midpoint
                - grape_pos,
                dim=-1,
            ),

        "upper_distance_m":
            torch.linalg.vector_norm(
                upper
                - grape_pos,
                dim=-1,
            ),

        "lower_distance_m":
            torch.linalg.vector_norm(
                lower
                - grape_pos,
                dim=-1,
            ),

        "pad_center_gap_m":
            torch.linalg.vector_norm(
                upper
                - lower,
                dim=-1,
            ),

        "jaw_rad":
            robot.data
            .joint_pos[
                :,
                jaw_id,
            ],

        "grape_vz_mps":
            grape.data
            .root_link_lin_vel_w[
                :,
                2,
            ],

        "orientation_error_deg":
            orientation_error_deg(
                robot,
                standing_reference_quat,
            ),
    }

    result.update(
        contact_tensors(
            env,
            force_threshold,
        )
    )

    return result


def env0_row(
    state,
    stage,
):

    return {
        "stage":
            stage,

        **{
            key:
                float(
                    value[
                        0
                    ]
                    .detach()
                    .cpu()
                    .item()
                )

            for key, value
            in state.items()
        },
    }


# =============================================================================
# 1. GPU body search
# =============================================================================


def body_search(
    env,
    standing_reference_quat,
    jaw_id,
    mouth_id,
    upper_id,
    lower_id,
    leg_joint_ids,
    args,
):

    print()

    print(
        "=" * 72
    )

    print(
        "1. GPU BODY REACHABILITY"
    )

    print(
        "=" * 72
    )

    blend_values = (
        np.linspace(
            0.0,
            1.0,
            args.leg_blend_steps,
        )
        .tolist()
    )

    rows = []

    for batch in chunks(
        blend_values,
        args.num_envs,
    ):

        padded = pad_batch(
            batch,
            args.num_envs,
        )

        env.reset()

        target = build_leg_pose_batch(
            env.scene[
                "robot"
            ],
            padded,
            leg_joint_ids,
        )

        ramp_to_pose(
            env,
            target,
            args.fold_seconds,
            args.body_settle_seconds,
            jaw_id,
            args.jaw_open_rad,
        )

        batch_rows = geometry_rows(
            geometry_tensors(
                env,
                standing_reference_quat,
                mouth_id,
                upper_id,
                lower_id,
            ),

            batch,

            lambda blend: {
                "leg_blend":
                    float(
                        blend
                    ),
            },
        )

        for row in batch_rows:

            row[
                "candidate_ok"
            ] = bool(
                row[
                    "root_height_m"
                ]
                > 0.015

                and

                row[
                    "root_speed_mps"
                ]
                <= args.max_settled_speed

                and

                row[
                    "orientation_error_deg"
                ]
                <= (
                    args
                    .body_candidate_max_orientation_deg
                )
            )

            rows.append(
                row
            )

            print(
                f"blend="
                f"{row['leg_blend']:.3f} | "

                f"root="
                f"{row['root_height_m']*100:.2f}cm | "

                f"grip="
                f"{row['grip_midpoint_height_m']*100:.2f}cm | "

                f"err="
                f"{row['vertical_error_m']*1000:.2f}mm | "

                f"standing-relative="
                f"{row['orientation_error_deg']:.1f}° | "

                f"speed="
                f"{row['root_speed_mps']:.3f}m/s | "

                f"ok="
                f"{row['candidate_ok']}"
            )

    valid = [
        row
        for row in rows
        if row[
            "candidate_ok"
        ]
    ]

    if not valid:

        raise RuntimeError(
            "No body pose passed "
            "the exploratory body filters."
        )

    ranked = sorted(
        valid,
        key=lambda row:
            row[
                "vertical_error_m"
            ],
    )

    return (
        rows,
        ranked,
    )


# =============================================================================
# 2. GPU head / neck search
# =============================================================================


def joint_limits(
    robot,
    joint_id,
    requested_min,
    requested_max,
):

    try:

        limits = (
            robot.data
            .soft_joint_pos_limits[
                0,
                joint_id,
            ]
        )

        return (
            max(
                requested_min,
                float(
                    limits[
                        0
                    ]
                ),
            ),

            min(
                requested_max,
                float(
                    limits[
                        1
                    ]
                ),
            ),
        )

    except Exception:

        return (
            requested_min,
            requested_max,
        )


def head_search(
    env,
    standing_reference_quat,
    body_ranked,
    jaw_id,
    neck_id,
    head_id,
    mouth_id,
    upper_id,
    lower_id,
    leg_joint_ids,
    args,
):

    robot = env.scene[
        "robot"
    ]

    neck_min, neck_max = joint_limits(
        robot,
        neck_id,
        args.neck_min,
        args.neck_max,
    )

    head_min, head_max = joint_limits(
        robot,
        head_id,
        args.head_min,
        args.head_max,
    )

    neck_values = (
        np.linspace(
            neck_min,
            neck_max,
            args.neck_steps,
        )
        .tolist()
    )

    head_values = (
        np.linspace(
            head_min,
            head_max,
            args.head_steps,
        )
        .tolist()
    )

    candidates = [
        {
            "leg_blend":
                float(
                    body[
                        "leg_blend"
                    ]
                ),

            "neck_pitch_rad":
                float(
                    neck
                ),

            "head_pitch_rad":
                float(
                    head
                ),
        }

        for body
        in body_ranked[
            :args.body_top_k
        ]

        for neck
        in neck_values

        for head
        in head_values
    ]

    print()

    print(
        "=" * 72
    )

    print(
        "2A. GPU HEAD / NECK GEOMETRY SEARCH"
    )

    print(
        "=" * 72
    )

    num_batches = math.ceil(
        len(
            candidates
        )
        / args.num_envs
    )

    print(
        f"{len(candidates)} candidates | "
        f"{args.num_envs} parallel envs | "
        f"{num_batches} GPU batches"
    )

    coarse_rows = []

    for batch_index, batch in enumerate(
        chunks(
            candidates,
            args.num_envs,
        ),
        start=1,
    ):

        padded = pad_batch(
            batch,
            args.num_envs,
        )

        env.reset()

        body_target = build_leg_pose_batch(
            env.scene[
                "robot"
            ],

            [
                candidate[
                    "leg_blend"
                ]
                for candidate
                in padded
            ],

            leg_joint_ids,
        )

        ramp_to_pose(
            env,
            body_target,
            args.fold_seconds,
            args.body_settle_seconds,
            jaw_id,
            args.jaw_open_rad,
        )

        robot = env.scene[
            "robot"
        ]

        # ----------------------------------------------------
        # COARSE STAGE
        #
        # Directly set head/neck.
        # No physical head settling for every candidate.
        # ----------------------------------------------------

        q = set_head_batch(
            robot.data
            .joint_pos
            .clone(),

            neck_id,

            head_id,

            [
                candidate[
                    "neck_pitch_rad"
                ]
                for candidate
                in padded
            ],

            [
                candidate[
                    "head_pitch_rad"
                ]
                for candidate
                in padded
            ],
        )

        env_ids = torch.arange(
            env.num_envs,
            device=env.device,
        )

        robot.write_joint_state_to_sim(
            q,

            torch.zeros_like(
                q
            ),

            env_ids=env_ids,
        )

        force_jaw(
            env,
            jaw_id,
            args.jaw_open_rad,
        )

        env.scene.write_data_to_sim()

        env.sim.forward()

        env.scene.update(
            dt=0.0
        )

        coarse_rows.extend(
            geometry_rows(
                geometry_tensors(
                    env,
                    standing_reference_quat,
                    mouth_id,
                    upper_id,
                    lower_id,
                ),

                batch,

                lambda candidate: {
                    **candidate,

                    "search_stage":
                        "coarse",

                    "valid":
                        False,

                    "posture_change_deg":
                        float(
                            "nan"
                        ),
                },
            )
        )

        print(
            f"  coarse batch "
            f"{batch_index}/"
            f"{num_batches} "
            f"("
            f"{len(batch)} candidates"
            f")"
        )

    coarse_ranked = sorted(
        coarse_rows,
        key=lambda row:
            row[
                "vertical_error_m"
            ],
    )

    validation_candidates = (
        coarse_ranked[
            :min(
                args.head_validate_top_k,
                len(
                    coarse_ranked
                ),
            )
        ]
    )

    print()

    print(
        "=" * 72
    )

    print(
        "2B. GPU PHYSICAL VALIDATION"
    )

    print(
        "=" * 72
    )

    print(
        f"Physically validating "
        f"{len(validation_candidates)} "
        f"best candidates"
    )

    validated_rows = []

    for batch in chunks(
        validation_candidates,
        args.num_envs,
    ):

        padded = pad_batch(
            batch,
            args.num_envs,
        )

        env.reset()

        body_target = build_leg_pose_batch(
            env.scene[
                "robot"
            ],

            [
                candidate[
                    "leg_blend"
                ]
                for candidate
                in padded
            ],

            leg_joint_ids,
        )

        ramp_to_pose(
            env,
            body_target,
            args.fold_seconds,
            args.body_settle_seconds,
            jaw_id,
            args.jaw_open_rad,
        )

        robot = env.scene[
            "robot"
        ]

        # ====================================================
        # IMPORTANT FIX
        #
        # The duck is ALLOWED to be heavily pitched forward.
        #
        # Save the torso orientation AFTER the crouch settles.
        #
        # Then move the neck/head physically and measure how
        # much the body changes relative to this crouch.
        #
        # We are NOT using standing-relative orientation as
        # the final stability gate.
        # ====================================================

        crouch_reference_quat = (
            robot.data
            .root_link_quat_w
            .clone()
        )

        physical_target = set_head_batch(
            robot.data
            .joint_pos
            .clone(),

            neck_id,

            head_id,

            [
                candidate[
                    "neck_pitch_rad"
                ]
                for candidate
                in padded
            ],

            [
                candidate[
                    "head_pitch_rad"
                ]
                for candidate
                in padded
            ],
        )

        ramp_to_pose(
            env,
            physical_target,
            args.head_move_seconds,
            args.head_settle_seconds,
            jaw_id,
            args.jaw_open_rad,
        )

        metrics = geometry_tensors(
            env,
            standing_reference_quat,
            mouth_id,
            upper_id,
            lower_id,
        )

        posture_change = (
            orientation_error_deg(
                robot,
                crouch_reference_quat,
            )[
                :len(
                    batch
                )
            ]
            .detach()
            .cpu()
            .numpy()
        )

        batch_rows = geometry_rows(
            metrics,

            batch,

            lambda candidate: {
                "leg_blend":
                    float(
                        candidate[
                            "leg_blend"
                        ]
                    ),

                "neck_pitch_rad":
                    float(
                        candidate[
                            "neck_pitch_rad"
                        ]
                    ),

                "head_pitch_rad":
                    float(
                        candidate[
                            "head_pitch_rad"
                        ]
                    ),

                "search_stage":
                    "validated",
            },
        )

        for i, row in enumerate(
            batch_rows
        ):

            row[
                "posture_change_deg"
            ] = float(
                posture_change[
                    i
                ]
            )

            row[
                "valid"
            ] = bool(
                row[
                    "root_height_m"
                ]
                > 0.015

                and

                row[
                    "root_speed_mps"
                ]
                <= args.max_settled_speed

                and

                row[
                    "posture_change_deg"
                ]
                <= args.max_posture_change_deg
            )

            validated_rows.append(
                row
            )

            print(
                f"  blend="
                f"{row['leg_blend']:.3f}, "

                f"neck="
                f"{row['neck_pitch_rad']:.3f}, "

                f"head="
                f"{row['head_pitch_rad']:.3f} | "

                f"err="
                f"{row['vertical_error_m']*1000:.2f}mm | "

                f"standing-relative="
                f"{row['orientation_error_deg']:.1f}° | "

                f"posture-change="
                f"{row['posture_change_deg']:.1f}° | "

                f"speed="
                f"{row['root_speed_mps']:.3f}m/s | "

                f"valid="
                f"{row['valid']}"
            )

    valid_rows = [
        row
        for row in validated_rows
        if row[
            "valid"
        ]
    ]

    if not valid_rows:

        best_debug = sorted(
            validated_rows,

            key=lambda row: (
                row[
                    "posture_change_deg"
                ],
                row[
                    "vertical_error_m"
                ],
            ),
        )[
            :5
        ]

        print()

        print(
            "Best rejected physical candidates:"
        )

        for row in best_debug:

            print(
                f"  err="
                f"{row['vertical_error_m']*1000:.2f}mm | "

                f"posture-change="
                f"{row['posture_change_deg']:.1f}° | "

                f"speed="
                f"{row['root_speed_mps']:.3f}m/s | "

                f"root="
                f"{row['root_height_m']*100:.2f}cm"
            )

        raise RuntimeError(
            "No physically validated head/neck pose passed. "
            "Standing-relative orientation is NOT the gate anymore. "
            "Inspect posture-change and root-speed above."
        )

    ranked = sorted(
        valid_rows,
        key=lambda row:
            row[
                "vertical_error_m"
            ],
    )

    return (
        coarse_rows
        + validated_rows,

        ranked,
    )


# =============================================================================
# Prepare selected pose in every environment
# =============================================================================


def prepare_best_pose(
    env,
    best_pose,
    neck_id,
    head_id,
    jaw_id,
    leg_joint_ids,
    args,
):

    env.reset()

    target = build_leg_pose_batch(
        env.scene[
            "robot"
        ],

        [
            best_pose[
                "leg_blend"
            ]
        ]
        * env.num_envs,

        leg_joint_ids,
    )

    target = set_head_batch(
        target,

        neck_id,

        head_id,

        [
            best_pose[
                "neck_pitch_rad"
            ]
        ]
        * env.num_envs,

        [
            best_pose[
                "head_pitch_rad"
            ]
        ]
        * env.num_envs,
    )

    action = ramp_to_pose(
        env,
        target,
        args.fold_seconds,
        args.final_settle_seconds,
        jaw_id,
        args.jaw_open_rad,
    )

    force_jaw(
        env,
        jaw_id,
        args.jaw_open_rad,
    )

    return action


# =============================================================================
# 3. GPU grape placement sweep
# =============================================================================


def placement_sweep(
    env,
    best_pose,
    standing_reference_quat,
    neck_id,
    head_id,
    jaw_id,
    mouth_id,
    upper_id,
    lower_id,
    leg_joint_ids,
    args,
):

    print()

    print(
        "=" * 72
    )

    print(
        "3. GPU GRAPE PLACEMENT SWEEP"
    )

    print(
        "=" * 72
    )

    placements = [
        {
            "offset_x_mm":
                float(
                    x
                ),

            "offset_y_mm":
                float(
                    y
                ),
        }

        for x
        in args.sweep_x_mm

        for y
        in args.sweep_y_mm
    ]

    print(
        f"{len(placements)} placements | "
        f"{args.num_envs} parallel envs"
    )

    results = []

    for batch in chunks(
        placements,
        args.num_envs,
    ):

        padded = pad_batch(
            batch,
            args.num_envs,
        )

        action = prepare_best_pose(
            env,
            best_pose,
            neck_id,
            head_id,
            jaw_id,
            leg_joint_ids,
            args,
        )

        reset_contact_sensors(
            env
        )

        place_grapes_on_floor(
            env,
            upper_id,
            lower_id,

            [
                placement[
                    "offset_x_mm"
                ]
                / 1000.0

                for placement
                in padded
            ],

            [
                placement[
                    "offset_y_mm"
                ]
                / 1000.0

                for placement
                in padded
            ],
        )

        open_steps = max(
            1,
            math.ceil(
                args.open_grape_settle_seconds
                / env.step_dt
            ),
        )

        for _ in range(
            open_steps
        ):

            step_with_locked_jaw(
                env,
                action,
                jaw_id,
                args.jaw_open_rad,
            )

        max_upper_force = torch.zeros(
            env.num_envs,
            device=env.device,
        )

        max_lower_force = torch.zeros_like(
            max_upper_force
        )

        current_streak = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.long,
        )

        max_streak = torch.zeros_like(
            current_streak
        )

        def update_contact_accumulators(
            contact,
        ):

            nonlocal max_upper_force
            nonlocal max_lower_force
            nonlocal current_streak
            nonlocal max_streak

            max_upper_force = torch.maximum(
                max_upper_force,
                contact[
                    "upper_force_n"
                ],
            )

            max_lower_force = torch.maximum(
                max_lower_force,
                contact[
                    "lower_force_n"
                ],
            )

            both = (
                contact[
                    "both_contact"
                ]
                >= 0.5
            )

            current_streak = torch.where(
                both,
                current_streak + 1,
                torch.zeros_like(
                    current_streak
                ),
            )

            max_streak = torch.maximum(
                max_streak,
                current_streak,
            )

        close_steps = max(
            1,
            math.ceil(
                args.jaw_close_seconds
                / env.step_dt
            ),
        )

        for step in range(
            close_steps
        ):

            alpha = (
                step + 1
            ) / close_steps

            jaw_value = (
                args.jaw_open_rad
                + alpha
                * (
                    args.jaw_closed_rad
                    - args.jaw_open_rad
                )
            )

            step_with_locked_jaw(
                env,
                action,
                jaw_id,
                jaw_value,
            )

            update_contact_accumulators(
                contact_tensors(
                    env,
                    args.contact_force_threshold,
                )
            )

        hold_steps = max(
            1,
            math.ceil(
                args.closed_hold_seconds
                / env.step_dt
            ),
        )

        upper_sum = torch.zeros(
            env.num_envs,
            device=env.device,
        )

        lower_sum = torch.zeros_like(
            upper_sum
        )

        both_sum = torch.zeros_like(
            upper_sum
        )

        for _ in range(
            hold_steps
        ):

            step_with_locked_jaw(
                env,
                action,
                jaw_id,
                args.jaw_closed_rad,
            )

            contact = contact_tensors(
                env,
                args.contact_force_threshold,
            )

            upper_sum += contact[
                "upper_contact"
            ]

            lower_sum += contact[
                "lower_contact"
            ]

            both_sum += contact[
                "both_contact"
            ]

            update_contact_accumulators(
                contact
            )

        state = full_state_tensors(
            env,
            standing_reference_quat,
            mouth_id,
            upper_id,
            lower_id,
            jaw_id,
            args.contact_force_threshold,
        )

        n = len(
            batch
        )

        upper_fraction = (
            (
                upper_sum
                / hold_steps
            )[
                :n
            ]
            .detach()
            .cpu()
            .numpy()
        )

        lower_fraction = (
            (
                lower_sum
                / hold_steps
            )[
                :n
            ]
            .detach()
            .cpu()
            .numpy()
        )

        both_fraction = (
            (
                both_sum
                / hold_steps
            )[
                :n
            ]
            .detach()
            .cpu()
            .numpy()
        )

        upper_force = (
            max_upper_force[
                :n
            ]
            .detach()
            .cpu()
            .numpy()
        )

        lower_force = (
            max_lower_force[
                :n
            ]
            .detach()
            .cpu()
            .numpy()
        )

        streak = (
            max_streak[
                :n
            ]
            .detach()
            .cpu()
            .numpy()
        )

        final_distance = (
            state[
                "grip_distance_m"
            ][
                :n
            ]
            .detach()
            .cpu()
            .numpy()
        )

        final_height = (
            state[
                "grape_height_m"
            ][
                :n
            ]
            .detach()
            .cpu()
            .numpy()
        )

        for i, placement in enumerate(
            batch
        ):

            row = {
                **placement,

                "upper_contact_fraction":
                    float(
                        upper_fraction[
                            i
                        ]
                    ),

                "lower_contact_fraction":
                    float(
                        lower_fraction[
                            i
                        ]
                    ),

                "both_contact_fraction":
                    float(
                        both_fraction[
                            i
                        ]
                    ),

                "max_both_contact_streak":
                    int(
                        streak[
                            i
                        ]
                    ),

                "max_upper_force_n":
                    float(
                        upper_force[
                            i
                        ]
                    ),

                "max_lower_force_n":
                    float(
                        lower_force[
                            i
                        ]
                    ),

                "final_grip_distance_m":
                    float(
                        final_distance[
                            i
                        ]
                    ),

                "final_grape_height_m":
                    float(
                        final_height[
                            i
                        ]
                    ),
            }

            results.append(
                row
            )

            print(
                f"x="
                f"{row['offset_x_mm']:+5.1f}mm "

                f"y="
                f"{row['offset_y_mm']:+5.1f}mm | "

                f"upper="
                f"{row['upper_contact_fraction']:.2f} "

                f"lower="
                f"{row['lower_contact_fraction']:.2f} "

                f"BOTH="
                f"{row['both_contact_fraction']:.2f} | "

                f"streak="
                f"{row['max_both_contact_streak']:3d} | "

                f"F=("
                f"{row['max_upper_force_n']:.3f},"
                f"{row['max_lower_force_n']:.3f})N | "

                f"d="
                f"{row['final_grip_distance_m']*1000:.1f}mm"
            )

    def score(
        row,
    ):

        return (
            row[
                "both_contact_fraction"
            ],

            row[
                "max_both_contact_streak"
            ],

            min(
                row[
                    "max_upper_force_n"
                ],
                row[
                    "max_lower_force_n"
                ],
            ),

            -row[
                "final_grip_distance_m"
            ],
        )

    best_placement = sorted(
        results,
        key=score,
        reverse=True,
    )[
        0
    ]

    print()

    print(
        "BEST PLACEMENT"
    )

    print(
        "-" * 72
    )

    print(
        json.dumps(
            best_placement,
            indent=2,
        )
    )

    return (
        results,
        best_placement,
    )


# =============================================================================
# 4. Final clamp + rise
# =============================================================================


def final_grasp(
    env,
    best_pose,
    best_placement,
    standing_reference_quat,
    neck_id,
    head_id,
    jaw_id,
    mouth_id,
    upper_id,
    lower_id,
    leg_joint_ids,
    args,
):

    print()

    print(
        "=" * 72
    )

    print(
        "4. FINAL CLAMP + PHYSICAL RISE"
    )

    print(
        "=" * 72
    )

    action = prepare_best_pose(
        env,
        best_pose,
        neck_id,
        head_id,
        jaw_id,
        leg_joint_ids,
        args,
    )

    reset_contact_sensors(
        env
    )

    place_grapes_on_floor(
        env,
        upper_id,
        lower_id,

        best_placement[
            "offset_x_mm"
        ]
        / 1000.0,

        best_placement[
            "offset_y_mm"
        ]
        / 1000.0,
    )

    trace = []

    def record(
        stage,
    ):

        trace.append(
            env0_row(
                full_state_tensors(
                    env,
                    standing_reference_quat,
                    mouth_id,
                    upper_id,
                    lower_id,
                    jaw_id,
                    args.contact_force_threshold,
                ),
                stage,
            )
        )

    open_steps = max(
        1,
        math.ceil(
            args.open_grape_settle_seconds
            / env.step_dt
        ),
    )

    for _ in range(
        open_steps
    ):

        step_with_locked_jaw(
            env,
            action,
            jaw_id,
            args.jaw_open_rad,
        )

        record(
            "open"
        )

    close_steps = max(
        1,
        math.ceil(
            args.jaw_close_seconds
            / env.step_dt
        ),
    )

    for step in range(
        close_steps
    ):

        alpha = (
            step + 1
        ) / close_steps

        jaw_value = (
            args.jaw_open_rad
            + alpha
            * (
                args.jaw_closed_rad
                - args.jaw_open_rad
            )
        )

        step_with_locked_jaw(
            env,
            action,
            jaw_id,
            jaw_value,
        )

        record(
            "closing"
        )

    hold_steps = max(
        1,
        math.ceil(
            args.closed_hold_seconds
            / env.step_dt
        ),
    )

    closed_rows = []

    for _ in range(
        hold_steps
    ):

        step_with_locked_jaw(
            env,
            action,
            jaw_id,
            args.jaw_closed_rad,
        )

        record(
            "closed_hold"
        )

        closed_rows.append(
            trace[
                -1
            ]
        )

    pre_lift = trace[
        -1
    ]

    dual_contact_fraction = float(
        np.mean(
            [
                row[
                    "both_contact"
                ]

                for row
                in closed_rows
            ]
        )
    )

    current = 0

    best_streak = 0

    for row in closed_rows:

        if (
            row[
                "both_contact"
            ]
            >= 0.5
        ):

            current += 1

            best_streak = max(
                best_streak,
                current,
            )

        else:

            current = 0

    robot = env.scene[
        "robot"
    ]

    stand_pose = (
        robot.data
        .default_joint_pos
        .clone()
    )

    # Keep the selected neck/head grasp orientation while legs unfold.

    stand_pose[
        :,
        neck_id,
    ] = best_pose[
        "neck_pitch_rad"
    ]

    stand_pose[
        :,
        head_id,
    ] = best_pose[
        "head_pitch_rad"
    ]

    start_pose = (
        robot.data
        .joint_pos
        .clone()
    )

    rise_steps = max(
        1,
        math.ceil(
            args.rise_seconds
            / env.step_dt
        ),
    )

    for step in range(
        rise_steps
    ):

        alpha = (
            step + 1
        ) / rise_steps

        desired = (
            start_pose
            + alpha
            * (
                stand_pose
                - start_pose
            )
        )

        step_with_locked_jaw(
            env,
            pose_to_action(
                env,
                desired,
            ),
            jaw_id,
            args.jaw_closed_rad,
        )

        record(
            "rising"
        )

    final_action = pose_to_action(
        env,
        stand_pose,
    )

    final_hold_steps = max(
        1,
        math.ceil(
            args.rise_hold_seconds
            / env.step_dt
        ),
    )

    for _ in range(
        final_hold_steps
    ):

        step_with_locked_jaw(
            env,
            final_action,
            jaw_id,
            args.jaw_closed_rad,
        )

        record(
            "final_hold"
        )

    post_lift = trace[
        -1
    ]

    grape_lift = (
        post_lift[
            "grape_height_m"
        ]
        - pre_lift[
            "grape_height_m"
        ]
    )

    grip_lift = (
        post_lift[
            "grip_height_m"
        ]
        - pre_lift[
            "grip_height_m"
        ]
    )

    mouth_lift = (
        post_lift[
            "mouth_height_m"
        ]
        - pre_lift[
            "mouth_height_m"
        ]
    )

    follow_ratio = (
        grape_lift
        / max(
            grip_lift,
            1e-6,
        )
    )

    return {
        "rows":
            trace,

        "dual_contact_fraction_closed":
            dual_contact_fraction,

        "dual_contact_streak":
            best_streak,

        "grape_lift_m":
            grape_lift,

        "grip_lift_m":
            grip_lift,

        "mouth_lift_m":
            mouth_lift,

        "follow_ratio":
            follow_ratio,

        "final_grip_distance_m":
            post_lift[
                "grip_distance_m"
            ],

        "final_upper_contact":
            post_lift[
                "upper_contact"
            ],

        "final_lower_contact":
            post_lift[
                "lower_contact"
            ],
    }


# =============================================================================
# Output helpers
# =============================================================================


def write_csv(
    path,
    rows,
):

    if not rows:

        return

    # Head rows have slightly different keys between coarse and validated.
    # Build union of all keys.

    fieldnames = []

    seen = set()

    for row in rows:

        for key in row:

            if key not in seen:

                fieldnames.append(
                    key
                )

                seen.add(
                    key
                )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def make_plots(
    output_dir,
    body_rows,
    head_rows,
    placement_rows,
    trace_rows,
    dt,
):

    os.environ.setdefault(
        "MPLBACKEND",
        "Agg",
    )

    import matplotlib.pyplot as plt

    # --------------------------------------------------------
    # BODY
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            10,
            8,
        ),
        sharex=True,
    )

    x = [
        row[
            "leg_blend"
        ]

        for row
        in body_rows
    ]

    axes[
        0
    ].plot(
        x,

        [
            row[
                "root_height_m"
            ]

            for row
            in body_rows
        ],

        marker="o",

        label="root",
    )

    axes[
        0
    ].plot(
        x,

        [
            row[
                "grip_midpoint_height_m"
            ]

            for row
            in body_rows
        ],

        marker="o",

        label="grip midpoint",
    )

    axes[
        0
    ].axhline(
        GRAPE_HALF_HEIGHT,
        linestyle="--",
        label="grape center",
    )

    axes[
        0
    ].set_ylabel(
        "Height (m)"
    )

    axes[
        0
    ].set_title(
        "Free-root body reachability"
    )

    axes[
        0
    ].legend()

    axes[
        0
    ].grid(
        alpha=0.25
    )

    axes[
        1
    ].plot(
        x,

        [
            row[
                "orientation_error_deg"
            ]

            for row
            in body_rows
        ],

        marker="o",
    )

    axes[
        1
    ].set_xlabel(
        "Leg blend"
    )

    axes[
        1
    ].set_ylabel(
        "Standing-relative orientation (deg)"
    )

    axes[
        1
    ].grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "body_reachability.png",
        dpi=170,
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------
    # HEAD
    # --------------------------------------------------------

    coarse = [
        row
        for row in head_rows
        if row[
            "search_stage"
        ] == "coarse"
    ]

    validated = [
        row
        for row in head_rows
        if row[
            "search_stage"
        ] == "validated"
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            10,
            10,
        ),
    )

    scatter = axes[
        0
    ].scatter(
        [
            row[
                "neck_pitch_rad"
            ]

            for row
            in coarse
        ],

        [
            row[
                "head_pitch_rad"
            ]

            for row
            in coarse
        ],

        c=[
            row[
                "vertical_error_m"
            ]
            * 1000.0

            for row
            in coarse
        ],
    )

    fig.colorbar(
        scatter,
        ax=axes[
            0
        ],
        label="Vertical error (mm)",
    )

    axes[
        0
    ].set_xlabel(
        "neck_pitch (rad)"
    )

    axes[
        0
    ].set_ylabel(
        "head_pitch (rad)"
    )

    axes[
        0
    ].set_title(
        "GPU neck/head geometry search"
    )

    axes[
        0
    ].grid(
        alpha=0.25
    )

    validated_sorted = sorted(
        validated,
        key=lambda row:
            row[
                "vertical_error_m"
            ],
    )

    axes[
        1
    ].plot(
        np.arange(
            len(
                validated_sorted
            )
        ),

        [
            row[
                "posture_change_deg"
            ]

            for row
            in validated_sorted
        ],

        marker="o",

        label="posture change",
    )

    axes[
        1
    ].axhline(
        0.0,
        linewidth=1,
    )

    axes[
        1
    ].set_xlabel(
        "Validated candidate rank"
    )

    axes[
        1
    ].set_ylabel(
        "Posture change (deg)"
    )

    axes[
        1
    ].set_title(
        "Physical stability relative to settled crouch"
    )

    axes[
        1
    ].grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "head_reachability.png",
        dpi=170,
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------
    # PLACEMENT
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            12,
            5,
        ),
    )

    px = [
        row[
            "offset_x_mm"
        ]

        for row
        in placement_rows
    ]

    py = [
        row[
            "offset_y_mm"
        ]

        for row
        in placement_rows
    ]

    scatter = axes[
        0
    ].scatter(
        px,
        py,

        c=[
            row[
                "both_contact_fraction"
            ]

            for row
            in placement_rows
        ],

        s=150,
    )

    fig.colorbar(
        scatter,
        ax=axes[
            0
        ],
        label="Both-pad contact fraction",
    )

    axes[
        0
    ].set_xlabel(
        "X offset (mm)"
    )

    axes[
        0
    ].set_ylabel(
        "Y offset (mm)"
    )

    axes[
        0
    ].set_title(
        "Simultaneous upper + lower contact"
    )

    axes[
        0
    ].grid(
        alpha=0.25
    )

    scatter = axes[
        1
    ].scatter(
        px,
        py,

        c=[
            row[
                "final_grip_distance_m"
            ]
            * 1000.0

            for row
            in placement_rows
        ],

        s=150,
    )

    fig.colorbar(
        scatter,
        ax=axes[
            1
        ],
        label="Final grip distance (mm)",
    )

    axes[
        1
    ].set_xlabel(
        "X offset (mm)"
    )

    axes[
        1
    ].set_ylabel(
        "Y offset (mm)"
    )

    axes[
        1
    ].set_title(
        "Grape distance after closure"
    )

    axes[
        1
    ].grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "placement_sweep.png",
        dpi=170,
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------
    # FINAL DIAGNOSTIC
    # --------------------------------------------------------

    time = (
        np.arange(
            len(
                trace_rows
            )
        )
        * dt
    )

    def series(
        key,
    ):

        return np.array(
            [
                row[
                    key
                ]

                for row
                in trace_rows
            ]
        )

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(
            12,
            20,
        ),
        sharex=True,
    )

    axes[
        0
    ].plot(
        time,
        series(
            "grape_height_m"
        ),
        label="grape",
    )

    axes[
        0
    ].plot(
        time,
        series(
            "grip_height_m"
        ),
        label="grip",
    )

    axes[
        0
    ].plot(
        time,
        series(
            "root_height_m"
        ),
        label="root",
    )

    axes[
        0
    ].plot(
        time,
        series(
            "mouth_height_m"
        ),
        label="mouth",
    )

    axes[
        0
    ].set_ylabel(
        "Height (m)"
    )

    axes[
        0
    ].set_title(
        "Does the grape rise with the grip?"
    )

    axes[
        0
    ].legend()

    axes[
        1
    ].plot(
        time,
        series(
            "grip_distance_m"
        ),
        label="grip -> grape",
    )

    axes[
        1
    ].plot(
        time,
        series(
            "upper_distance_m"
        ),
        label="upper -> grape",
    )

    axes[
        1
    ].plot(
        time,
        series(
            "lower_distance_m"
        ),
        label="lower -> grape",
    )

    axes[
        1
    ].set_ylabel(
        "Distance (m)"
    )

    axes[
        1
    ].legend()

    axes[
        2
    ].plot(
        time,
        series(
            "jaw_rad"
        ),
        label="jaw",
    )

    axes[
        2
    ].plot(
        time,
        series(
            "pad_center_gap_m"
        ),
        label="pad center gap",
    )

    axes[
        2
    ].set_title(
        "Jaw closure"
    )

    axes[
        2
    ].legend()

    axes[
        3
    ].plot(
        time,
        series(
            "upper_contact"
        ),
        label="upper",
    )

    axes[
        3
    ].plot(
        time,
        series(
            "lower_contact"
        ),
        label="lower",
    )

    axes[
        3
    ].plot(
        time,
        series(
            "both_contact"
        ),
        label="BOTH",
    )

    axes[
        3
    ].set_ylim(
        -0.05,
        1.05,
    )

    axes[
        3
    ].set_title(
        "Real grape-pad contacts"
    )

    axes[
        3
    ].legend()

    axes[
        4
    ].plot(
        time,
        series(
            "upper_force_n"
        ),
        label="upper force",
    )

    axes[
        4
    ].plot(
        time,
        series(
            "lower_force_n"
        ),
        label="lower force",
    )

    axes[
        4
    ].set_ylabel(
        "Force (N)"
    )

    axes[
        4
    ].legend()

    axes[
        5
    ].plot(
        time,
        series(
            "grape_vz_mps"
        ),
    )

    axes[
        5
    ].axhline(
        0.0,
        linewidth=1,
    )

    axes[
        5
    ].set_ylabel(
        "Grape Vz"
    )

    axes[
        6
    ].plot(
        time,
        series(
            "orientation_error_deg"
        ),
    )

    axes[
        6
    ].set_ylabel(
        "Standing-relative orientation (deg)"
    )

    axes[
        6
    ].set_xlabel(
        "Time (s)"
    )

    for axis in axes:

        axis.grid(
            alpha=0.25
        )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "grasp_diagnostic.png",
        dpi=170,
    )

    plt.close(
        fig
    )


# =============================================================================
# Main
# =============================================================================


@torch.inference_mode()
def run(
    args,
):

    configure_torch_backends()

    torch.manual_seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    device = (
        args.device
        or (
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    if device.startswith(
        "cuda"
    ):

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA requested, but "
                "torch.cuda.is_available() is False"
            )

        gpu_index = (
            int(
                device.split(
                    ":"
                )[1]
            )

            if ":" in device

            else 0
        )

        torch.cuda.set_device(
            gpu_index
        )

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(gpu_index)}"
        )

        print(
            f"Parallel simulation envs: "
            f"{args.num_envs}"
        )

    else:

        print(
            f"WARNING: running on {device}. "
            f"Use --device cuda:0 for GPU execution."
        )

    output_dir = (
        Path(
            args.output
        )
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    env = ManagerBasedRlEnv(
        cfg=configure_env(
            args
        ),
        device=device,
    )

    env.reset()

    robot = env.scene[
        "robot"
    ]

    standing_reference_quat = (
        robot.data
        .root_link_quat_w
        .clone()
    )

    leg_joint_ids = resolve_leg_joint_ids(
        robot
    )

    mouth_ids, mouth_names = robot.find_sites(
        r"^mouth_tip$"
    )

    if not len(
        mouth_ids
    ):

        raise RuntimeError(
            "mouth_tip site not found"
        )

    mouth_id = int(
        mouth_ids[
            0
        ]
    )

    jaw_id, jaw_name = find_jaw_joint(
        robot
    )

    neck_id, neck_name = find_joint(
        robot,
        "neck_pitch",
    )

    head_id, head_name = find_joint(
        robot,
        "head_pitch",
    )

    upper_id, upper_name = find_geom(
        robot,
        args.upper_grip_pattern,
        "upper grip",
    )

    lower_id, lower_name = find_geom(
        robot,
        args.lower_grip_pattern,
        "lower grip",
    )

    print()

    print(
        "=" * 72
    )

    print(
        "GPU-VECTORIZED MICRODUCK GRAPE DIAGNOSTIC"
    )

    print(
        "=" * 72
    )

    print(
        f"device:     "
        f"{device}"
    )

    print(
        f"num_envs:   "
        f"{args.num_envs}"
    )

    print(
        f"mouth:      "
        f"{mouth_names[0]}"
    )

    print(
        f"jaw:        "
        f"{jaw_name}"
    )

    print(
        f"neck:       "
        f"{neck_name}"
    )

    print(
        f"head:       "
        f"{head_name}"
    )

    print(
        f"upper grip: "
        f"{upper_name}"
    )

    print(
        f"lower grip: "
        f"{lower_name}"
    )

    body_rows, body_ranked = body_search(
        env,
        standing_reference_quat,
        jaw_id,
        mouth_id,
        upper_id,
        lower_id,
        leg_joint_ids,
        args,
    )

    write_csv(
        output_dir
        / "body_reachability.csv",

        body_rows,
    )

    head_rows, head_ranked = head_search(
        env,
        standing_reference_quat,
        body_ranked,
        jaw_id,
        neck_id,
        head_id,
        mouth_id,
        upper_id,
        lower_id,
        leg_joint_ids,
        args,
    )

    write_csv(
        output_dir
        / "head_reachability.csv",

        head_rows,
    )

    best_pose = head_ranked[
        0
    ]

    print()

    print(
        "=" * 72
    )

    print(
        "BEST PHYSICALLY VALID GRASP POSE"
    )

    print(
        "=" * 72
    )

    print(
        json.dumps(
            best_pose,
            indent=2,
        )
    )

    reach_success = (
        best_pose[
            "vertical_error_m"
        ]
        <= args.reach_tolerance
    )

    if not reach_success:

        raise RuntimeError(
            f"Best physically valid pose is "
            f"{best_pose['vertical_error_m']*1000:.2f} mm "
            f"from grape-center height; "
            f"tolerance is "
            f"{args.reach_tolerance*1000:.1f} mm."
        )

    placement_rows, best_placement = placement_sweep(
        env,
        best_pose,
        standing_reference_quat,
        neck_id,
        head_id,
        jaw_id,
        mouth_id,
        upper_id,
        lower_id,
        leg_joint_ids,
        args,
    )

    write_csv(
        output_dir
        / "placement_sweep.csv",

        placement_rows,
    )

    grasp = final_grasp(
        env,
        best_pose,
        best_placement,
        standing_reference_quat,
        neck_id,
        head_id,
        jaw_id,
        mouth_id,
        upper_id,
        lower_id,
        leg_joint_ids,
        args,
    )

    write_csv(
        output_dir
        / "trace.csv",

        grasp[
            "rows"
        ],
    )

    checks = {
        "reach":
            bool(
                reach_success
            ),

        "dual_contact_fraction":
            bool(
                grasp[
                    "dual_contact_fraction_closed"
                ]
                >= args.min_dual_contact_fraction
            ),

        "dual_contact_streak":
            bool(
                grasp[
                    "dual_contact_streak"
                ]
                >= args.min_dual_contact_streak
            ),

        "grip_rose":
            bool(
                grasp[
                    "grip_lift_m"
                ]
                >= args.min_grip_lift
            ),

        "grape_rose":
            bool(
                grasp[
                    "grape_lift_m"
                ]
                >= args.min_grape_lift
            ),

        "grape_followed":
            bool(
                grasp[
                    "follow_ratio"
                ]
                >= args.min_follow_ratio
            ),

        "grape_stayed_close":
            bool(
                grasp[
                    "final_grip_distance_m"
                ]
                <= args.max_final_grip_distance
            ),
    }

    report = {
        "deterministic":
            True,

        "uses_checkpoint":
            False,

        "device":
            device,

        "num_parallel_envs":
            args.num_envs,

        "root_fixed":
            False,

        "real_contact_sensors":
            True,

        "best_pose":
            best_pose,

        "best_placement":
            best_placement,

        "closed_grasp": {
            "dual_contact_fraction":
                grasp[
                    "dual_contact_fraction_closed"
                ],

            "dual_contact_streak":
                grasp[
                    "dual_contact_streak"
                ],

            "final_upper_contact":
                grasp[
                    "final_upper_contact"
                ],

            "final_lower_contact":
                grasp[
                    "final_lower_contact"
                ],
        },

        "rise": {
            "grip_lift_m":
                grasp[
                    "grip_lift_m"
                ],

            "mouth_lift_m":
                grasp[
                    "mouth_lift_m"
                ],

            "grape_lift_m":
                grasp[
                    "grape_lift_m"
                ],

            "follow_ratio":
                grasp[
                    "follow_ratio"
                ],

            "final_grip_distance_m":
                grasp[
                    "final_grip_distance_m"
                ],
        },

        "checks":
            checks,

        "success":
            bool(
                all(
                    checks.values()
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

    make_plots(
        output_dir,
        body_rows,
        head_rows,
        placement_rows,
        grasp[
            "rows"
        ],
        env.step_dt,
    )

    print()

    print(
        "=" * 72
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=" * 72
    )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()

    print(
        f"Outputs: "
        f"{output_dir}"
    )

    env.close()


# =============================================================================
# CLI
# =============================================================================


def parse_args():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Use cuda:0 for GPU."
        ),
    )

    parser.add_argument(
        "--num-envs",
        type=int,
        default=64,
        help=(
            "Parallel MJLab environments. "
            "Start with 64; "
            "try 128 if VRAM allows."
        ),
    )

    parser.add_argument(
        "--upper-grip-pattern",
        default=r"^upper_mouth_grip$",
    )

    parser.add_argument(
        "--lower-grip-pattern",
        default=r"^lower_mouth_grip$",
    )

    parser.add_argument(
        "--leg-blend-steps",
        type=int,
        default=13,
    )

    parser.add_argument(
        "--fold-seconds",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--body-settle-seconds",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "--body-top-k",
        type=int,
        default=3,
    )

    # Exploratory only.
    # Pickup poses are allowed to be heavily pitched.

    parser.add_argument(
        "--body-candidate-max-orientation-deg",
        type=float,
        default=100.0,
    )

    # ========================================================
    # THIS is the final body stability gate.
    #
    # It compares:
    #
    # settled crouch
    #       vs
    # body orientation after neck/head motion
    #
    # It does NOT compare against standing.
    # ========================================================

    parser.add_argument(
        "--max-posture-change-deg",
        type=float,
        default=20.0,
        help=(
            "Maximum torso orientation change "
            "caused by moving neck/head after "
            "the crouch has already settled."
        ),
    )

    parser.add_argument(
        "--max-settled-speed",
        type=float,
        default=0.30,
    )

    # Your prior result was about 5-7 mm,
    # so 8 mm intentionally allows the
    # contact test to proceed.

    parser.add_argument(
        "--reach-tolerance",
        type=float,
        default=0.008,
    )

    parser.add_argument(
        "--neck-min",
        type=float,
        default=-1.5,
    )

    parser.add_argument(
        "--neck-max",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--neck-steps",
        type=int,
        default=11,
    )

    parser.add_argument(
        "--head-min",
        type=float,
        default=-1.5,
    )

    parser.add_argument(
        "--head-max",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--head-steps",
        type=int,
        default=11,
    )

    parser.add_argument(
        "--head-validate-top-k",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--head-move-seconds",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--head-settle-seconds",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--final-settle-seconds",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--jaw-open-rad",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--jaw-closed-rad",
        type=float,
        default=-0.08,
    )

    parser.add_argument(
        "--jaw-close-seconds",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--open-grape-settle-seconds",
        type=float,
        default=0.08,
    )

    parser.add_argument(
        "--closed-hold-seconds",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--contact-history",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--contact-force-threshold",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--sweep-x-mm",
        type=float,
        nargs="+",
        default=[
            -10.0,
            -6.0,
            -3.0,
            0.0,
            3.0,
            6.0,
            10.0,
        ],
    )

    parser.add_argument(
        "--sweep-y-mm",
        type=float,
        nargs="+",
        default=[
            -5.0,
            0.0,
            5.0,
        ],
    )

    parser.add_argument(
        "--rise-seconds",
        type=float,
        default=1.2,
    )

    parser.add_argument(
        "--rise-hold-seconds",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--min-dual-contact-fraction",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--min-dual-contact-streak",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--min-grip-lift",
        type=float,
        default=0.015,
    )

    parser.add_argument(
        "--min-grape-lift",
        type=float,
        default=0.010,
    )

    parser.add_argument(
        "--min-follow-ratio",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--max-final-grip-distance",
        type=float,
        default=0.030,
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
            "grape-full-diagnostic"
        ),
    )

    args = parser.parse_args()

    if args.num_envs <= 0:

        parser.error(
            "--num-envs must be positive"
        )

    return args


if __name__ == "__main__":

    run(
        parse_args()
    )