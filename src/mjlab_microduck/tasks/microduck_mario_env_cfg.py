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

from mjlab.managers import (
    EventTermCfg,
    MetricsTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
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
BUTTON_RESAMPLE_S = (0.5, 1.25)
ACTIVATE_ANGLE = math.radians(0.6)
RELEASE_ANGLE = math.radians(0.2)
CHORD_PRESS_TRAVEL = 0.0011
CHORD_RELEASE_TRAVEL = 0.0006
BUTTON_ACTIVATION_WEIGHT = 3.0
BUTTON_PROGRESS_WEIGHT = 1.0
STAND_HEIGHT = 0.130  # measured walk-model equilibrium (0.115 m) + 15 mm pad top
FOOT_ANCHOR_RADIUS = 0.025
COM_FORWARD_OFFSET = 0.012
COM_LATERAL_OFFSET = 0.010
COMMAND_LEAN_ANGLE = math.radians(6.0)
MIN_TRUNK_HEIGHT = 0.110
FULL_TRUNK_HEIGHT = 0.120
MIN_CAMERA_HEIGHT = 0.200
FULL_CAMERA_HEIGHT = 0.230
FULL_CAMERA_TILT_DEG = 12.0
MAX_CAMERA_TILT_DEG = 20.0
MIN_VIEW_ALIGNMENT = 0.85
FULL_VIEW_ALIGNMENT = 0.95
FULL_LEG_POSE_ERROR = 0.16
MAX_LEG_POSE_ERROR = 0.40


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
            mode="subtree",
            pattern="controller_root",
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

    # This is a stationary manipulation task.  The velocity template injects
    # horizontal robustness kicks (and makes them especially frequent in play
    # mode); retaining that event turns both training and checkpoint videos
    # into push-recovery tests instead of controller-control tests.
    cfg.events.pop("push_robot", None)

    # Fixed spawn aligned with the physical controller. Random x/y/yaw would
    # move the duck away from pads whose locations are fixed in the env frame.
    reset_base = cfg.events["reset_base"]
    reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.135, 0.135),
        "yaw": (0.0, 0.0),
    }
    # Fixed-base entities are represented as per-world mocap bodies by mjlab.
    # Unlike the floating robot, they are not moved to each environment origin
    # unless an explicit root-reset event does it. Without this event all 4096
    # controllers remain at global (0, 0), while almost every robot stands on
    # terrain elsewhere in the grid and can never produce controller contact.
    cfg.events["reset_nes_controller_root"] = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("nes_controller"),
        },
    )
    # Reset the passive rockers too; otherwise their previous episode state is
    # inherited after the robot and mocap root are reset.
    cfg.events["reset_nes_controller_joints"] = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("nes_controller", joint_names=(r".*",)),
        },
    )

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

    # A button request is not a walking command. The inherited velocity pose
    # term normally loosens its leg tolerances whenever twist is non-zero;
    # here that made the robot abandon its standing pose exactly when it was
    # asked to press a button. Keep the standing tolerances for every request
    # and make balance worth more than a perfect button press.
    pose_params = cfg.rewards["pose"].params
    pose_params["std_walking"] = deepcopy(pose_params["std_standing"])
    pose_params["std_running"] = deepcopy(pose_params["std_standing"])
    cfg.rewards["pose"].weight = 2.0
    cfg.rewards["upright"] = RewardTermCfg(
        func=microduck_mdp.mario_commanded_lean_reward,
        weight=5.0,
        params={
            "command_name": "twist",
            "lean_angle": COMMAND_LEAN_ANGLE,
            "std": math.radians(7.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["commanded_trunk_offset"] = RewardTermCfg(
        func=microduck_mdp.mario_commanded_trunk_offset_reward,
        weight=3.0,
        params={
            "command_name": "twist",
            "forward_offset": COM_FORWARD_OFFSET,
            "lateral_offset": COM_LATERAL_OFFSET,
            "std": 0.008,
            "robot_cfg": SceneEntityCfg(
                "robot", site_names=("left_foot", "right_foot")
            ),
        },
    )
    cfg.rewards["standing_height"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=3.0,
        params={
            "target_height": STAND_HEIGHT,
            "std": 0.012,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # The Gaussian above becomes nearly flat once a policy discovers a deep
    # crouch.  This shortfall cost keeps a useful slope back toward a usable
    # head-camera height even when the trunk has dropped far below STAND_HEIGHT.
    cfg.rewards["camera_crouch"] = RewardTermCfg(
        func=microduck_mdp.mario_crouch_cost,
        weight=-4.0,
        params={
            "camera_cfg": SceneEntityCfg("robot", site_names=("head_camera",)),
            "trunk_floor": FULL_TRUNK_HEIGHT,
            "camera_floor": FULL_CAMERA_HEIGHT,
            "trunk_scale": 0.020,
            "camera_scale": 0.040,
        },
    )
    leg_pose_cfg = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",)
    )
    cfg.rewards["leg_pose_l1"] = RewardTermCfg(
        func=microduck_mdp.mario_leg_pose_l1_cost,
        weight=-2.0,
        params={"asset_cfg": leg_pose_cfg, "scale": 0.35, "max_cost": 2.0},
    )
    cfg.rewards["neutral_head_pose"] = RewardTermCfg(
        func=microduck_mdp.mario_selected_pose_reward,
        weight=1.5,
        params={
            "std": 0.15,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=(r"^(?!passive_).*(neck|head).*",)
            ),
        },
    )
    cfg.rewards["body_ang_vel"].weight = -0.15
    cfg.rewards["angular_momentum"].weight = -0.05

    controller_activation_params = {
        "command_name": "twist",
        "asset_name": "nes_controller",
        "activate_angle": ACTIVATE_ANGLE,
        "release_angle": RELEASE_ANGLE,
        "chord_press_travel": CHORD_PRESS_TRAVEL,
        "chord_release_travel": CHORD_RELEASE_TRAVEL,
    }
    anchored_button_params = {
        **controller_activation_params,
        "anchor_radius": FOOT_ANCHOR_RADIUS,
        "anchor_sensor_name": controller_feet_contact.name,
        "robot_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
        "controller_cfg": SceneEntityCfg(
            "nes_controller",
            body_names=("dpad_platform", "ab_rocker_platform"),
        ),
    }
    camera_ready_params = {
        "camera_cfg": SceneEntityCfg("robot", site_names=("head_camera",)),
        "min_trunk_height": MIN_TRUNK_HEIGHT,
        "full_trunk_height": FULL_TRUNK_HEIGHT,
        "min_camera_height": MIN_CAMERA_HEIGHT,
        "full_camera_height": FULL_CAMERA_HEIGHT,
        "full_tilt_deg": FULL_CAMERA_TILT_DEG,
        "max_tilt_deg": MAX_CAMERA_TILT_DEG,
        "min_view_alignment": MIN_VIEW_ALIGNMENT,
        "full_view_alignment": FULL_VIEW_ALIGNMENT,
    }
    standing_pose_params = {
        "standing_pose_cfg": leg_pose_cfg,
        "full_pose_error": FULL_LEG_POSE_ERROR,
        "max_pose_error": MAX_LEG_POSE_ERROR,
        "require_exclusive": True,
    }
    cfg.rewards["requested_button"] = RewardTermCfg(
        func=microduck_mdp.mario_requested_button_reward,
        weight=BUTTON_ACTIVATION_WEIGHT,
        params={
            **anchored_button_params,
            **camera_ready_params,
            **standing_pose_params,
        },
    )
    # Keep this separate in the logs: requested_button reports physical
    # activation while planted and camera-ready; this term supplies a gradient
    # before the hard activation point, subject to the same gates.
    cfg.rewards["requested_button_progress"] = RewardTermCfg(
        func=microduck_mdp.mario_requested_button_progress_reward,
        weight=BUTTON_PROGRESS_WEIGHT,
        params={
            "command_name": "twist",
            "asset_name": "nes_controller",
            "activate_angle": ACTIVATE_ANGLE,
            "release_angle": RELEASE_ANGLE,
            "chord_press_travel": CHORD_PRESS_TRAVEL,
            "chord_release_travel": CHORD_RELEASE_TRAVEL,
            "anchor_radius": FOOT_ANCHOR_RADIUS,
            "anchor_sensor_name": controller_feet_contact.name,
            "robot_cfg": SceneEntityCfg(
                "robot", site_names=("left_foot", "right_foot")
            ),
            "controller_cfg": SceneEntityCfg(
                "nes_controller",
                body_names=("dpad_platform", "ab_rocker_platform"),
            ),
            **camera_ready_params,
            **standing_pose_params,
        },
    )
    cfg.rewards["unrequested_button"] = RewardTermCfg(
        func=microduck_mdp.mario_unrequested_button_cost,
        weight=-2.0,
        params=controller_activation_params,
    )
    # Losing support can make a button press easier in simulation but violates
    # the physical design. This is a cost (not a constant positive jackpot).
    cfg.rewards["foot_contact_loss"] = RewardTermCfg(
        func=microduck_mdp.feet_contact_loss_cost,
        weight=-6.0,
        params={
            "sensor_name": controller_feet_contact.name,
        },
    )
    cfg.rewards["foot_anchor"] = RewardTermCfg(
        func=microduck_mdp.mario_foot_anchor_cost,
        weight=-4.0,
        params={
            "deadzone": 0.008,
            "scale": 0.020,
            "robot_cfg": SceneEntityCfg(
                "robot", site_names=("left_foot", "right_foot")
            ),
            "controller_cfg": SceneEntityCfg(
                "nes_controller",
                body_names=("dpad_platform", "ab_rocker_platform"),
            ),
        },
    )
    cfg.rewards["foot_planar_speed"] = RewardTermCfg(
        func=microduck_mdp.mario_foot_planar_speed_cost,
        weight=-2.0,
        params={
            "speed_scale": 0.10,
            "max_cost": 2.0,
            "robot_cfg": SceneEntityCfg(
                "robot", site_names=("left_foot", "right_foot")
            ),
        },
    )

    # Unweighted audit signals.  These show whether reward growth corresponds
    # to a physically usable policy rather than another compromise pose.
    cfg.metrics["camera_ready"] = MetricsTermCfg(
        func=microduck_mdp.mario_camera_ready,
        params={
            "camera_cfg": camera_ready_params["camera_cfg"],
            **{
                key: value
                for key, value in camera_ready_params.items()
                if key != "camera_cfg"
            },
        },
    )
    cfg.metrics["standing_pose_ready"] = MetricsTermCfg(
        func=microduck_mdp.mario_standing_pose_ready,
        params={
            "asset_cfg": leg_pose_cfg,
            "full_error": FULL_LEG_POSE_ERROR,
            "max_error": MAX_LEG_POSE_ERROR,
        },
    )
    cfg.metrics["requested_button_clean"] = MetricsTermCfg(
        func=microduck_mdp.mario_requested_button_reward,
        params={
            **anchored_button_params,
            **camera_ready_params,
            **standing_pose_params,
        },
    )
    cfg.metrics["requested_button_success"] = MetricsTermCfg(
        func=microduck_mdp.mario_clean_button_success,
        params={
            "success_threshold": 0.95,
            **anchored_button_params,
            **camera_ready_params,
            **standing_pose_params,
        },
    )
    cfg.metrics["feet_anchored"] = MetricsTermCfg(
        func=microduck_mdp.mario_feet_anchored,
        params={
            "anchor_radius": FOOT_ANCHOR_RADIUS,
            "sensor_name": controller_feet_contact.name,
            "robot_cfg": anchored_button_params["robot_cfg"],
            "controller_cfg": anchored_button_params["controller_cfg"],
        },
    )
    cfg.metrics["unrequested_button_activation"] = MetricsTermCfg(
        func=microduck_mdp.mario_unrequested_button_cost,
        params=controller_activation_params,
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
# A warm-started balance mean should not immediately be destroyed by the
# velocity recipe's std=1.0 random actions. The source checkpoint's learned
# std is deliberately not copied; 0.20 leaves task exploration without the
# catastrophic first-step thrashing seen in the from-scratch run.
MicroduckMarioRlCfg.actor.distribution_cfg["init_std"] = 0.20
