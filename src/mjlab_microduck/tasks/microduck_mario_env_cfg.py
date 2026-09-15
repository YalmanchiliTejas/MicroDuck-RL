"""Microduck physical controller task for a Mario-style platform game.

The policy does not control the game directly. It receives a 3D button request
``[left, right, jump]`` in the established twist command slot and must depress
the corresponding spring-loaded floor pads. The game consumes measured pad
travel, so requested actions cannot bypass the simulated robot.

Actor observations remain exactly 61D. Pad travel is critic-only privileged
state and is not required on the real robot.
"""

from copy import deepcopy

from mjlab.managers import ObservationTermCfg, RewardTermCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.robot.microduck_constants import MICRODUCK_CONTROLLER_PADS_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)


EPISODE_LENGTH_S = 20.0
BUTTON_RESAMPLE_S = (1.5, 3.0)
PAD_PRESS_TRAVEL = 0.004
PAD_RELEASE_TRAVEL = 0.002


def make_microduck_mario_env_cfg(play: bool = False):
    """Create the flat-ground physical game-controller training environment."""

    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)
    # Robot must remain first: reset events assume its free joint starts qpos.
    cfg.scene.entities = {
        "robot": cfg.scene.entities["robot"],
        "controller_pads": MICRODUCK_CONTROLLER_PADS_CFG,
    }
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.sim.nconmax = 80

    # Fixed spawn aligned with the physical controller. Random x/y/yaw would
    # move the duck away from pads whose locations are fixed in the env frame.
    reset_base = cfg.events["reset_base"]
    reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.12, 0.12),
        "yaw": (0.0, 0.0),
    }

    cfg.commands["twist"] = microduck_mdp.MarioButtonCommandCfg(
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

    # The actor cannot sense pad travel on hardware. The asymmetric critic can
    # use it to predict whether a requested press is about to pay off.
    cfg.observations["critic"].terms["controller_pad_travel"] = ObservationTermCfg(
        func=microduck_mdp.mario_pad_travel,
        params={"asset_name": "controller_pads"},
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

    cfg.rewards["requested_pad"] = RewardTermCfg(
        func=microduck_mdp.mario_requested_pad_reward,
        weight=6.0,
        params={
            "command_name": "twist",
            "asset_name": "controller_pads",
            "press_travel": PAD_PRESS_TRAVEL,
            "release_travel": PAD_RELEASE_TRAVEL,
        },
    )
    cfg.rewards["unrequested_pad"] = RewardTermCfg(
        func=microduck_mdp.mario_unrequested_pad_cost,
        weight=-4.0,
        params={
            "command_name": "twist",
            "asset_name": "controller_pads",
            "press_travel": PAD_PRESS_TRAVEL,
            "release_travel": PAD_RELEASE_TRAVEL,
        },
    )

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
MicroduckMarioRlCfg.experiment_name = "mario_controller"
MicroduckMarioRlCfg.run_name = "mario_controller"
MicroduckMarioRlCfg.max_iterations = 5_000
