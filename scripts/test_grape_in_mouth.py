#!/usr/bin/env python3
"""
MicroDuck GrapePick crouch diagnostic.

Purpose
-------
This script isolates the exact problem we are seeing before any body/head search:

    reset
      -> move physically to STAND_POSE
      -> settle
      -> move physically to CROUCH_POSE
      -> settle

It ALWAYS saves a MuJoCo video and CSV trace, even if the duck falls face-flat.

There is no body-search filter that can abort before the video is written.

No PPO checkpoint.
No head search.
No jaw closing.
No lifting.

Outputs
-------
artifacts/grape-crouch-debug/
    00_forced_body_descent.mp4
    00_forced_body_descent.csv
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

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViewerConfig

import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.tasks.microduck_grape_pick_env_cfg import (
    GRAPE_HALF_HEIGHT,
)


TASK_ID = "Mjlab-GrapePick-Flat-MicroDuck"


# =============================================================================
# Known stand/crouch poses
# =============================================================================

STAND_POSE = {
    "left_hip_yaw": -0.0476,
    "left_hip_roll": -0.0629,
    "left_hip_pitch": -0.2869,
    "left_knee": 0.9618,
    "left_ankle": 1.1674,

    "neck_pitch": 0.6029,
    "head_pitch": 0.543,
    "head_yaw": -0.069,
    "head_roll": -0.0414,

    "right_hip_yaw": -0.0337,
    "right_hip_roll": -0.0061,
    "right_hip_pitch": 0.1534,
    "right_knee": -0.9725,
    "right_ankle": -1.0646,
}


CROUCH_POSE = {
    "left_hip_yaw": -0.0184,
    "left_hip_roll": 0.0307,
    "left_hip_pitch": 1.4082,
    "left_knee": 1.5248,
    "left_ankle": -0.0675,

    "neck_pitch": 1.0937,
    "head_pitch": 1.2149,
    "head_yaw": -0.0184,
    "head_roll": -0.0368,

    "right_hip_yaw": 0.0184,
    "right_hip_roll": -0.0169,
    "right_hip_pitch": -1.4757,
    "right_knee": -1.5907,
    "right_ankle": 0.0568,
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

    cfg.scene.num_envs = 1
    cfg.episode_length_s = 120.0
    cfg.auto_reset = False

    cfg.terminations = {}
    cfg.curriculum = {}

    if "twist" in cfg.commands:
        try:
            cfg.commands["twist"].randomize_phase = False
        except Exception:
            pass

    # Make reset deterministic.
    cfg.events = {
        name: term
        for name, term in cfg.events.items()
        if name in BASELINE_RESET_EVENTS
    }

    if "reset_base" in cfg.events:
        cfg.events[
            "reset_base"
        ].params[
            "pose_range"
        ].update(
            {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),

                # Use the ORIGINAL grape-pick reset height first.
                "z": (
                    args.base_z,
                    args.base_z,
                ),

                "yaw": (0.0, 0.0),
            }
        )

        # Make sure this test starts without an injected velocity.
        if (
            "velocity_range"
            in cfg.events["reset_base"].params
        ):
            cfg.events[
                "reset_base"
            ].params[
                "velocity_range"
            ] = {}

    if "reset_grape" in cfg.events:
        try:
            cfg.events[
                "reset_grape"
            ].params[
                "noise_xy"
            ] = 0.0
        except Exception:
            pass

    for group in cfg.observations.values():
        try:
            group.enable_corruption = False
        except Exception:
            pass

    cfg.viewer = ViewerConfig(
        origin_type=(
            ViewerConfig
            .OriginType
            .ASSET_ROOT
        ),

        entity_name="robot",
        env_idx=0,

        distance=args.camera_distance,
        elevation=args.camera_elevation,
        azimuth=args.camera_azimuth,

        width=args.video_width,
        height=args.video_height,

        max_extra_envs=0,
        enable_shadows=True,
    )

    return cfg


# =============================================================================
# Joint lookup
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
                    int(ids[0]),
                    str(names[0]),
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
                    int(ids[0]),
                    str(names[0]),
                )

        except Exception:
            pass

    return None, None


def resolve_ids(
    env,
):
    robot = env.scene[
        "robot"
    ]

    body_joint_ids = {
        name:
            find_joint(
                robot,
                name,
            )[0]

        for name
        in STAND_POSE
    }

    jaw_id, jaw_name = (
        find_jaw_joint(
            robot
        )
    )

    return {
        "body_joint_ids":
            body_joint_ids,

        "jaw_id":
            jaw_id,

        "jaw_name":
            jaw_name,
    }


# =============================================================================
# Fixed grape
# =============================================================================

def initial_grape_relative(
    env,
):
    grape = env.scene[
        "grape"
    ]

    origins = (
        env.scene
        .terrain
        .env_origins
    )

    return (
        grape.data
        .root_link_pos_w[
            0
        ]
        - origins[
            0
        ]
    ).clone()


def set_fixed_grape(
    env,
    grape_relative,
):
    grape = env.scene[
        "grape"
    ]

    origins = (
        env.scene
        .terrain
        .env_origins
    )

    relative = (
        grape_relative
        .clone()
    )

    relative[
        2
    ] = GRAPE_HALF_HEIGHT

    position = (
        origins
        + relative[
            None,
            :
        ]
    )

    pose = torch.zeros(
        (
            env.num_envs,
            7,
        ),
        device=env.device,
        dtype=position.dtype,
    )

    pose[
        :,
        :3,
    ] = position

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
            dtype=position.dtype,
        ),
        env_ids=env_ids,
    )

    env.scene.write_data_to_sim()

    env.sim.forward()

    env.scene.update(
        dt=0.0
    )


# =============================================================================
# Pose generation
# =============================================================================

def build_crouch_pose_batch(
    robot,
    blend,
    body_joint_ids,
):
    """
    blend = 0.0 -> STAND_POSE
    blend = 1.0 -> CROUCH_POSE
    """

    pose = (
        robot.data
        .default_joint_pos
        .clone()
    )

    blend_tensor = torch.full(
        (
            pose.shape[0],
        ),
        float(blend),
        device=pose.device,
        dtype=pose.dtype,
    )

    for name, joint_id in (
        body_joint_ids.items()
    ):

        stand_value = (
            STAND_POSE[
                name
            ]
        )

        crouch_value = (
            CROUCH_POSE[
                name
            ]
        )

        pose[
            :,
            joint_id,
        ] = (
            stand_value
            + blend_tensor
            * (
                crouch_value
                - stand_value
            )
        )

    return pose


# =============================================================================
# Action helpers
# =============================================================================

def pose_to_action(
    env,
    pose,
):
    term = (
        env.action_manager
        .get_term(
            "joint_pos"
        )
    )

    target = pose[
        :,
        term.target_ids,
    ]

    return (
        target
        - term.offset
    ) / term.scale


def force_jaw(
    env,
    jaw_id,
    value,
):
    if jaw_id is None:
        return

    robot = env.scene[
        "robot"
    ]

    values = torch.full(
        (
            env.num_envs,
        ),
        float(value),
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


def step_locked(
    env,
    pose,
    jaw_id,
    jaw_value,
):
    force_jaw(
        env,
        jaw_id,
        jaw_value,
    )

    env.step(
        pose_to_action(
            env,
            pose,
        )
    )

    force_jaw(
        env,
        jaw_id,
        jaw_value,
    )


# =============================================================================
# Smooth interpolation
# =============================================================================

def smoothstep(
    alpha,
):
    alpha = max(
        0.0,
        min(
            1.0,
            alpha,
        ),
    )

    return (
        alpha
        * alpha
        * (
            3.0
            - 2.0
            * alpha
        )
    )


# =============================================================================
# Orientation / metrics
# =============================================================================

def orientation_error_deg(
    current_quat,
    reference_quat,
):
    q = (
        current_quat
        / torch.linalg.vector_norm(
            current_quat,
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
        q * ref,
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


def trace_row(
    env,
    standing_reference,
    desired_pose,
    ids,
    stage,
    time_s,
):
    robot = env.scene[
        "robot"
    ]

    terrain_z = (
        env.scene
        .terrain
        .env_origins[
            0,
            2,
        ]
    )

    orientation = (
        orientation_error_deg(
            robot.data
            .root_link_quat_w,

            standing_reference,
        )[
            0
        ]
    )

    root_height = (
        robot.data
        .root_link_pos_w[
            0,
            2,
        ]

        - terrain_z
    )

    root_speed = (
        torch.linalg.vector_norm(
            robot.data
            .root_link_lin_vel_w[
                0
            ]
        )
    )

    root_ang_speed = (
        torch.linalg.vector_norm(
            robot.data
            .root_link_ang_vel_w[
                0
            ]
        )
    )

    controlled_ids = torch.tensor(
        list(
            ids[
                "body_joint_ids"
            ].values()
        ),
        device=env.device,
        dtype=torch.long,
    )

    actual = (
        robot.data
        .joint_pos[
            0,
            controlled_ids,
        ]
    )

    desired = (
        desired_pose[
            0,
            controlled_ids,
        ]
    )

    tracking_rms = torch.sqrt(
        torch.mean(
            (
                actual
                - desired
            )
            ** 2
        )
    )

    return {
        "time_s":
            float(
                time_s
            ),

        "stage":
            stage,

        "root_height_m":
            float(
                root_height
                .detach()
                .cpu()
                .item()
            ),

        "orientation_deg":
            float(
                orientation
                .detach()
                .cpu()
                .item()
            ),

        "root_speed_mps":
            float(
                root_speed
                .detach()
                .cpu()
                .item()
            ),

        "root_ang_speed_rad_s":
            float(
                root_ang_speed
                .detach()
                .cpu()
                .item()
            ),

        "joint_tracking_rms_rad":
            float(
                tracking_rms
                .detach()
                .cpu()
                .item()
            ),
    }


# =============================================================================
# Video writer
# =============================================================================

class Video:

    def __init__(
        self,
        path,
        fps,
    ):
        try:
            import imageio.v2 as imageio

        except ImportError as exc:
            raise RuntimeError(
                "Install video support with:\n"
                "uv add imageio imageio-ffmpeg"
            ) from exc

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.writer = (
            imageio.get_writer(
                str(
                    path
                ),

                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
            )
        )

        self.fps = fps

    def frame(
        self,
        env,
    ):
        frame = env.render()

        if frame is None:
            raise RuntimeError(
                "env.render() returned None"
            )

        frame = np.asarray(
            frame
        )

        if frame.dtype != np.uint8:

            if (
                np.issubdtype(
                    frame.dtype,
                    np.floating,
                )

                and

                frame.max()
                <= 1.0
            ):
                frame = (
                    frame
                    * 255.0
                )

            frame = (
                np.clip(
                    frame,
                    0,
                    255,
                )
                .astype(
                    np.uint8
                )
            )

        return frame

    def append(
        self,
        env,
    ):
        self.writer.append_data(
            self.frame(
                env
            )
        )

    def hold(
        self,
        env,
        seconds,
    ):
        frame = self.frame(
            env
        )

        count = max(
            1,
            round(
                seconds
                * self.fps
            ),
        )

        for _ in range(
            count
        ):
            self.writer.append_data(
                frame
            )

    def close(
        self,
    ):
        self.writer.close()


# =============================================================================
# Motion runner
# =============================================================================

def run_motion(
    env,
    target_pose,
    move_seconds,
    hold_seconds,
    jaw_id,
    jaw_value,
    standing_reference,
    ids,
    stage,
    video,
    rows,
    time_s,
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
        raw_alpha = (
            step + 1
        ) / move_steps

        alpha = smoothstep(
            raw_alpha
        )

        desired = (
            start_pose
            + alpha
            * (
                target_pose
                - start_pose
            )
        )

        step_locked(
            env,
            desired,
            jaw_id,
            jaw_value,
        )

        time_s += (
            env.step_dt
        )

        rows.append(
            trace_row(
                env,
                standing_reference,
                desired,
                ids,
                stage,
                time_s,
            )
        )

        video.append(
            env
        )

    hold_steps = max(
        1,
        math.ceil(
            hold_seconds
            / env.step_dt
        ),
    )

    for _ in range(
        hold_steps
    ):
        step_locked(
            env,
            target_pose,
            jaw_id,
            jaw_value,
        )

        time_s += (
            env.step_dt
        )

        rows.append(
            trace_row(
                env,
                standing_reference,
                target_pose,
                ids,
                f"{stage}_hold",
                time_s,
            )
        )

        video.append(
            env
        )

    return time_s


# =============================================================================
# CSV
# =============================================================================

def write_csv(
    path,
    rows,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
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

        or

        (
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    args.device = device

    if device.startswith(
        "cuda"
    ):

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable"
            )

        gpu = (
            int(
                device.split(
                    ":"
                )[
                    1
                ]
            )

            if ":" in device

            else 0
        )

        torch.cuda.set_device(
            gpu
        )

        os.environ[
            "MUJOCO_EGL_DEVICE_ID"
        ] = str(
            gpu
        )

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(gpu)}"
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

        render_mode="rgb_array",
    )

    env.reset()

    ids = resolve_ids(
        env
    )

    grape_relative = (
        initial_grape_relative(
            env
        )
    )

    if args.grape_x_m is not None:
        grape_relative[
            0
        ] = args.grape_x_m

    if args.grape_y_m is not None:
        grape_relative[
            1
        ] = args.grape_y_m

    grape_relative[
        2
    ] = GRAPE_HALF_HEIGHT

    set_fixed_grape(
        env,
        grape_relative,
    )

    robot = env.scene[
        "robot"
    ]

    standing_reference = (
        robot.data
        .root_link_quat_w
        .clone()
    )

    stand_target = (
        build_crouch_pose_batch(
            robot,
            0.0,

            ids[
                "body_joint_ids"
            ],
        )
    )

    crouch_target = (
        build_crouch_pose_batch(
            robot,
            args.crouch_blend,

            ids[
                "body_joint_ids"
            ],
        )
    )

    fps = (
        args.video_fps

        if args.video_fps is not None

        else max(
            1,
            round(
                1.0
                / env.step_dt
            ),
        )
    )

    video_path = (
        output_dir
        / "00_forced_body_descent.mp4"
    )

    csv_path = (
        output_dir
        / "00_forced_body_descent.csv"
    )

    video = Video(
        video_path,
        fps,
    )

    rows = []

    time_s = 0.0

    try:
        force_jaw(
            env,

            ids[
                "jaw_id"
            ],

            args.jaw_open_rad,
        )

        video.hold(
            env,
            args.video_lead_seconds,
        )

        # Record the untouched reset state.
        rows.append(
            trace_row(
                env,
                standing_reference,

                robot.data
                .joint_pos
                .clone(),

                ids,
                "reset",
                time_s,
            )
        )

        print()
        print(
            "=" * 72
        )

        print(
            "A. MOVE RESET -> STAND_POSE"
        )

        print(
            "=" * 72
        )

        time_s = run_motion(
            env=env,

            target_pose=stand_target,

            move_seconds=(
                args.stand_setup_seconds
            ),

            hold_seconds=(
                args.stand_settle_seconds
            ),

            jaw_id=ids[
                "jaw_id"
            ],

            jaw_value=(
                args.jaw_open_rad
            ),

            standing_reference=(
                standing_reference
            ),

            ids=ids,

            stage="stand_setup",

            video=video,

            rows=rows,

            time_s=time_s,
        )

        stand_row = (
            rows[
                -1
            ]
        )

        print(
            f"After stand: "
            f"orientation="
            f"{stand_row['orientation_deg']:.1f}° | "
            f"root="
            f"{stand_row['root_height_m'] * 1000:.1f}mm | "
            f"tracking RMS="
            f"{stand_row['joint_tracking_rms_rad']:.3f}rad"
        )

        print()
        print(
            "=" * 72
        )

        print(
            f"B. STAND_POSE -> "
            f"{args.crouch_blend:.2f} CROUCH"
        )

        print(
            "=" * 72
        )

        time_s = run_motion(
            env=env,

            target_pose=(
                crouch_target
            ),

            move_seconds=(
                args.fold_seconds
            ),

            hold_seconds=(
                args.body_settle_seconds
            ),

            jaw_id=ids[
                "jaw_id"
            ],

            jaw_value=(
                args.jaw_open_rad
            ),

            standing_reference=(
                standing_reference
            ),

            ids=ids,

            stage="crouch_descent",

            video=video,

            rows=rows,

            time_s=time_s,
        )

        final_row = (
            rows[
                -1
            ]
        )

        print(
            f"After crouch: "
            f"orientation="
            f"{final_row['orientation_deg']:.1f}° | "
            f"root="
            f"{final_row['root_height_m'] * 1000:.1f}mm | "
            f"tracking RMS="
            f"{final_row['joint_tracking_rms_rad']:.3f}rad"
        )

        video.hold(
            env,
            args.video_tail_seconds,
        )

    finally:
        video.close()

        env.close()

    write_csv(
        csv_path,
        rows,
    )

    max_orientation_row = max(
        rows,

        key=lambda row:
            row[
                "orientation_deg"
            ],
    )

    min_root_row = min(
        rows,

        key=lambda row:
            row[
                "root_height_m"
            ],
    )

    stand_rows = [
        row
        for row in rows
        if row[
            "stage"
        ].startswith(
            "stand_setup"
        )
    ]

    crouch_rows = [
        row
        for row in rows
        if row[
            "stage"
        ].startswith(
            "crouch_descent"
        )
    ]

    max_stand_orientation = max(
        row[
            "orientation_deg"
        ]
        for row
        in stand_rows
    )

    max_crouch_orientation = max(
        row[
            "orientation_deg"
        ]
        for row
        in crouch_rows
    )

    if (
        max_stand_orientation
        >= args.failure_angle_deg
    ):
        conclusion = (
            "FAILS DURING STAND SETUP: "
            "the STAND_POSE / reset state is already incompatible "
            "with this GrapePick environment."
        )

    elif (
        max_crouch_orientation
        >= args.failure_angle_deg
    ):
        conclusion = (
            "STAND IS OK, FALLS DURING CROUCH: "
            "the stand-to-crouch trajectory or CROUCH_POSE is the problem."
        )

    else:
        conclusion = (
            "CROUCH STAYED WITHIN THE ORIENTATION LIMIT. "
            "The body descent is now good enough to reconnect "
            "to the grape search."
        )

    summary = {
        "task":
            TASK_ID,

        "device":
            device,

        "base_z_m":
            args.base_z,

        "crouch_blend":
            args.crouch_blend,

        "stand_setup_seconds":
            args.stand_setup_seconds,

        "stand_settle_seconds":
            args.stand_settle_seconds,

        "fold_seconds":
            args.fold_seconds,

        "body_settle_seconds":
            args.body_settle_seconds,

        "failure_angle_deg":
            args.failure_angle_deg,

        "max_orientation_deg":
            max_orientation_row[
                "orientation_deg"
            ],

        "max_orientation_stage":
            max_orientation_row[
                "stage"
            ],

        "max_orientation_time_s":
            max_orientation_row[
                "time_s"
            ],

        "min_root_height_m":
            min_root_row[
                "root_height_m"
            ],

        "max_stand_orientation_deg":
            max_stand_orientation,

        "max_crouch_orientation_deg":
            max_crouch_orientation,

        "conclusion":
            conclusion,

        "video":
            str(
                video_path
            ),

        "csv":
            str(
                csv_path
            ),
    }

    (
        output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",

        encoding="utf-8",
    )

    print()
    print(
        "=" * 72
    )

    print(
        "RESULT"
    )

    print(
        "=" * 72
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print(
        conclusion
    )

    print()
    print(
        f"WATCH: "
        f"{video_path}"
    )


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
    )

    # Start with original GrapePick reset height.
    parser.add_argument(
        "--base-z",
        type=float,
        default=0.125,
    )

    parser.add_argument(
        "--grape-x-m",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--grape-y-m",
        type=float,
        default=None,
    )

    # Reset -> STAND_POSE.
    parser.add_argument(
        "--stand-setup-seconds",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--stand-settle-seconds",
        type=float,
        default=0.75,
    )

    # STAND_POSE -> crouch.
    parser.add_argument(
        "--fold-seconds",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--body-settle-seconds",
        type=float,
        default=0.75,
    )

    # 1.0 = full CROUCH_POSE.
    parser.add_argument(
        "--crouch-blend",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--jaw-open-rad",
        type=float,
        default=0.50,
    )

    # Diagnostic only. It never aborts the script.
    parser.add_argument(
        "--failure-angle-deg",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--video-width",
        type=int,
        default=960,
    )

    parser.add_argument(
        "--video-height",
        type=int,
        default=720,
    )

    parser.add_argument(
        "--video-fps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--video-lead-seconds",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--video-tail-seconds",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--camera-distance",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--camera-elevation",
        type=float,
        default=-12.0,
    )

    parser.add_argument(
        "--camera-azimuth",
        type=float,
        default=90.0,
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
            "grape-crouch-debug"
        ),
    )

    args = parser.parse_args()

    if not (
        0.0
        <= args.crouch_blend
        <= 1.0
    ):
        parser.error(
            "--crouch-blend must be between 0 and 1"
        )

    return args


if __name__ == "__main__":
    run(
        parse_args()
    )