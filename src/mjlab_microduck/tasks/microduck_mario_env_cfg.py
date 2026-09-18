"""Microduck physical NES-controller task for a Mario-style platform game.

The policy receives ``[dpad_x, dpad_y, ab_mode]`` in the established 3D twist
slot and must move real spring-centered controller surfaces with its feet.
``ab_mode`` is -1=B, +1=A, and +2=A+B. The game consumes measured controller
joint state, so requests cannot bypass the simulated robot.

Actor observations remain exactly 61D. Controller joint state is critic-only
privileged state and is not required on the real robot.
"""

import math
from copy import deepcopy

from mjlab.managers import ObservationTermCfg, RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_MARIO_MONITOR_CFG,
    MICRODUCK_NES_CONTROLLER_CFG,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)


EPISODE_LENGTH_S = 20.0
BUTTON_RESAMPLE_S = (0.25, 0.75)
ACTIVATE_ANGLE = math.radians(1.4)
RELEASE_ANGLE = math.radians(0.8)
CHORD_PRESS_TRAVEL = 0.0016
CHORD_RELEASE_TRAVEL = 0.0010


def make_microduck_mario_env_cfg(play: bool = False):
    """Create the flat-ground physical game-controller training environment."""

    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)
    # Robot must remain first: reset events assume its free joint starts qpos.
    cfg.scene.entities = {
        "robot": cfg.scene.entities["robot"],
        "nes_controller": MICRODUCK_NES_CONTROLLER_CFG,
        "mario_monitor": MICRODUCK_MARIO_MONITOR_CFG,
    }
    # The inherited sensor targets the terrain plane. Here each sole is
    # supported by a moving controller surface, so contact must be measured
    # against those two physical geoms instead.
    controller_feet_contact = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="geom",
            pattern=r"^(dpad_surface|ab_surface)$",
            entity="nes_controller",
        ),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    cfg.scene.sensors = tuple(
        controller_feet_contact if sensor.name == "feet_ground_contact" else sensor
        for sensor in cfg.scene.sensors
    )
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.sim.nconmax = 80

    # Fixed spawn aligned with the physical controller. Random x/y/yaw would
    # move the duck away from pads whose locations are fixed in the env frame.
    reset_base = cfg.events["reset_base"]
    reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.135, 0.135),
        "yaw": (0.0, 0.0),
    }

    cfg.commands["twist"] = microduck_mdp.MarioNesCommandCfg(
        resampling_time_range=BUTTON_RESAMPLE_S,
    )
    # Head/body slots stay present in the observation but are unused here.
    cfg.commands.pop("head_pose", None)
    cfg.commands.pop("body_pose", None)
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding,
            params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding,
            params={"dim": 6},
        )

    # The actor cannot sense controller joint state on hardware. The asymmetric
    # critic gets all four physical axes, not decoded emulator button state.
    cfg.observations["critic"].terms["nes_controller_state"] = ObservationTermCfg(
        func=microduck_mdp.mario_nes_joint_state,
        params={"asset_name": "nes_controller"},
    )

    for name in (
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "head_pose_tracking",
        "body_pose_tracking",
        "head_pose_bias",
    ):
        cfg.rewards.pop(name, None)

    controller_reward_params = {
        "command_name": "twist",
        "asset_name": "nes_controller",
        "activate_angle": ACTIVATE_ANGLE,
        "release_angle": RELEASE_ANGLE,
        "chord_press_travel": CHORD_PRESS_TRAVEL,
        "chord_release_travel": CHORD_RELEASE_TRAVEL,
    }
    cfg.rewards["requested_button"] = RewardTermCfg(
        func=microduck_mdp.mario_requested_button_reward,
        weight=8.0,
        params=controller_reward_params,
    )
    cfg.rewards["unrequested_button"] = RewardTermCfg(
        func=microduck_mdp.mario_unrequested_button_cost,
        weight=-5.0,
        params=controller_reward_params,
    )
    # Losing support can make a button press easier in simulation but violates
    # the physical design. This is a cost (not a constant positive jackpot).
    cfg.rewards["foot_contact_loss"] = RewardTermCfg(
        func=microduck_mdp.feet_contact_loss_cost,
        weight=-4.0,
        params={
            "sensor_name": controller_feet_contact.name,
        },
    )

    # Small weight shifts should be discovered before smoothness is tightened.
    cfg.rewards["action_rate_l2"].weight = -0.02
    cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
        {"step": 0, "weight": -0.02},
        {"step": 500 * 24, "weight": -0.05},
        {"step": 1_000 * 24, "weight": -0.10},
        {"step": 2_000 * 24, "weight": -0.20},
    ]

    # Remove curricula that reference deleted velocity/head/body command or
    # reward terms. Retain action smoothing and sim2real DR curricula.
    for name in (
        "standing_envs",
        "head_pose_range",
        "body_pose_range",
        "head_pose_bias_weight",
    ):
        cfg.curriculum.pop(name, None)

    return cfg


MicroduckMarioRlCfg = deepcopy(MicroduckRlCfg)
MicroduckMarioRlCfg.experiment_name = "mario_nes_controller"
MicroduckMarioRlCfg.run_name = "mario_nes_controller"
MicroduckMarioRlCfg.max_iterations = 5_000
