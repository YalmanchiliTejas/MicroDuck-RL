import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import pytest
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_mario_env_cfg import (
    MicroduckMarioRlCfg,
    make_microduck_mario_env_cfg,
)


def test_mario_scene_keeps_robot_first_and_adds_controller():
    cfg = make_microduck_mario_env_cfg()
    assert list(cfg.scene.entities) == ["robot", "nes_controller", "mario_monitor"]


def test_mario_command_uses_existing_three_dimensional_twist_slot():
    cfg = make_microduck_mario_env_cfg()
    assert cfg.commands["twist"].class_type is microduck_mdp.MarioNesCommand
    assert sum(cfg.commands["twist"].category_weights) == 1.0
    assert cfg.commands["twist"].resampling_time_range == (0.5, 1.25)
    assert "head_pose" not in cfg.commands
    assert "body_pose" not in cfg.commands
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        assert terms["command"].func is mdp.generated_commands
        assert terms["command"].params["command_name"] == "twist"
        assert terms["head_command"].params["dim"] == 4
        assert terms["body_command"].params["dim"] == 6


def test_controller_state_is_privileged_and_not_added_to_actor():
    cfg = make_microduck_mario_env_cfg()
    assert "nes_controller_state" not in cfg.observations["actor"].terms
    assert "nes_controller_state" in cfg.observations["critic"].terms


def test_controller_reward_signs_cannot_reward_wrong_button_or_lifted_feet():
    rewards = make_microduck_mario_env_cfg().rewards
    assert rewards["requested_button"].weight > 0.0
    assert rewards["requested_button_progress"].weight > 0.0
    assert rewards["unrequested_button"].weight < 0.0
    assert rewards["foot_contact_loss"].weight < 0.0
    for name in ("track_linear_velocity", "air_time", "foot_clearance"):
        assert name not in rewards
    params = rewards["requested_button"].params
    assert params["release_angle"] < params["activate_angle"]
    assert params["activate_angle"] == pytest.approx(math.radians(0.6))
    assert params["release_angle"] == pytest.approx(math.radians(0.2))
    assert params["chord_release_travel"] == 0.0006
    assert params["chord_press_travel"] == 0.0011


def test_balance_reward_dominates_button_reward_and_keeps_standing_pose():
    rewards = make_microduck_mario_env_cfg().rewards
    total_button_weight = (
        rewards["requested_button"].weight + rewards["requested_button_progress"].weight
    )
    assert rewards["upright"].weight > total_button_weight
    assert rewards["upright"].func is microduck_mdp.mario_commanded_lean_reward
    assert rewards["foot_contact_loss"].weight < -total_button_weight
    assert rewards["body_ang_vel"].weight == -0.15
    assert rewards["angular_momentum"].weight == -0.05
    assert rewards["pose"].weight == 2.0
    pose_params = rewards["pose"].params
    assert pose_params["std_walking"] == pose_params["std_standing"]
    assert pose_params["std_running"] == pose_params["std_standing"]
    assert rewards["commanded_trunk_offset"].weight > 0.0
    assert rewards["standing_height"].params["target_height"] == 0.130
    assert rewards["camera_crouch"].weight < 0.0
    assert rewards["camera_crouch"].params["trunk_floor"] == 0.120
    assert rewards["camera_crouch"].params["camera_floor"] == 0.230
    assert rewards["neutral_head_pose"].func is microduck_mdp.pose_target_match
    assert rewards["foot_anchor"].weight < 0.0
    assert rewards["foot_planar_speed"].weight < 0.0


def test_mario_never_inherits_velocity_pushes_even_in_play():
    assert "push_robot" not in make_microduck_mario_env_cfg(play=False).events
    assert "push_robot" not in make_microduck_mario_env_cfg(play=True).events


def test_button_rewards_are_gated_by_both_foot_anchors():
    rewards = make_microduck_mario_env_cfg().rewards
    for name in ("requested_button", "requested_button_progress"):
        params = rewards[name].params
        assert params["anchor_radius"] > 0.0
        assert params["anchor_sensor_name"] == "feet_ground_contact"
        assert params["camera_cfg"].site_names == ("head_camera",)
        assert params["min_trunk_height"] < params["full_trunk_height"]
        assert params["min_camera_height"] < params["full_camera_height"]
        assert params["min_view_alignment"] < params["full_view_alignment"]
        assert params["robot_cfg"].site_names == ("left_foot", "right_foot")
        assert params["controller_cfg"].body_names == (
            "dpad_platform",
            "ab_rocker_platform",
        )


def test_spawn_is_aligned_with_fixed_pad_layout():
    cfg = make_microduck_mario_env_cfg()
    pose = cfg.events["reset_base"].params["pose_range"]
    assert pose["x"] == (0.0, 0.0)
    assert pose["y"] == (0.0, 0.0)
    assert pose["z"] == (0.135, 0.135)
    assert pose["yaw"] == (0.0, 0.0)
    controller_root = cfg.events["reset_nes_controller_root"]
    assert controller_root.func is mdp.reset_root_state_uniform
    assert controller_root.params["pose_range"] == {}
    assert controller_root.params["asset_cfg"].name == "nes_controller"
    controller_joints = cfg.events["reset_nes_controller_joints"]
    assert controller_joints.func is mdp.reset_joints_by_offset
    assert controller_joints.params["position_range"] == (0.0, 0.0)
    assert controller_joints.params["velocity_range"] == (0.0, 0.0)
    assert controller_joints.params["asset_cfg"].name == "nes_controller"


def test_mario_runner_has_distinct_experiment_name():
    assert MicroduckMarioRlCfg.experiment_name == "mario_nes_controller"


def test_unloaded_controller_settles_inside_all_release_thresholds():
    path = (
        Path(__file__).parents[1]
        / "src/mjlab_microduck/robot/microduck/controller_nes.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    for _ in range(2_000):
        mujoco.mj_step(model, data)
    for name in ("passive_dpad_x", "passive_dpad_y", "passive_ab_rocker"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        angle = abs(float(data.qpos[model.jnt_qposadr[joint_id]]))
        assert angle < math.radians(0.5)
    press_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "passive_ab_press")
    press_travel = -float(data.qpos[model.jnt_qposadr[press_id]])
    assert press_travel < 0.0009


def test_compact_command_decodes_all_six_buttons_and_chords():
    command = torch.tensor(
        [
            [1.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
            [0.0, 1.0, -1.0],
            [0.0, -1.0, 0.0],
        ]
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_command=lambda _name: command)
    )
    requested = microduck_mdp.mario_nes_requested_buttons(env).bool()
    # Button order: UP, DOWN, LEFT, RIGHT, A, B.
    assert requested.tolist() == [
        [False, False, False, True, True, True],
        [False, False, True, False, True, False],
        [True, False, False, False, False, True],
        [False, True, False, False, False, False],
    ]


def test_foot_sensor_targets_controller_surfaces_not_floor():
    sensor = next(
        sensor
        for sensor in make_microduck_mario_env_cfg().scene.sensors
        if sensor.name == "feet_ground_contact"
    )
    assert sensor.secondary.entity == "nes_controller"
    assert sensor.secondary.mode == "subtree"
    assert sensor.secondary.pattern == "controller_root"


def test_feet_grounded_counts_feet_not_contact_points():
    sensor = SimpleNamespace(
        data=SimpleNamespace(found=torch.tensor([[3.0, 0.0], [1.0, 1.0], [0.0, 0.0]]))
    )
    env = SimpleNamespace(
        num_envs=3,
        device=torch.device("cpu"),
        scene=SimpleNamespace(sensors={"feet": sensor}),
    )
    grounded = microduck_mdp.feet_grounded_reward(env, "feet")
    assert grounded.tolist() == [0.5, 1.0, 0.0]


def test_requested_button_has_dense_progress_before_activation():
    names = list(microduck_mdp._MARIO_NES_JOINTS)

    class FakeController:
        data = SimpleNamespace(
            joint_pos=torch.tensor([[math.radians(0.1), 0.0, 0.0, 0.0]])
        )

        @staticmethod
        def find_joints(patterns):
            name = patterns[0].removeprefix("^").removesuffix("$")
            return [names.index(name)], [name]

    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        scene={"nes_controller": FakeController()},
        command_manager=SimpleNamespace(
            get_command=lambda _name: torch.tensor([[1.0, 0.0, 0.0]])
        ),
    )
    activation = microduck_mdp.mario_nes_activation(env)[0, 3]
    activation_reward = microduck_mdp.mario_requested_button_reward(env)[0]
    progress_reward = microduck_mdp.mario_requested_button_progress_reward(env)[0]
    assert activation == 0.0
    assert activation_reward == 0.0
    assert 0.0 < progress_reward < 1.0


def _resolved_cfg(name, *, site_ids=None, body_ids=None):
    cfg = SceneEntityCfg(name)
    if site_ids is not None:
        cfg.site_ids = site_ids
    if body_ids is not None:
        cfg.body_ids = body_ids
    return cfg


def test_commanded_trunk_and_tilt_targets_encode_stationary_directional_lean():
    angle = math.radians(6.0)
    commands = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # neutral
            [0.0, 1.0, 0.0],  # up: forward
            [1.0, 0.0, 0.0],  # right: robot-frame -Y CoM, +roll
        ]
    )
    half = angle / 2.0
    quats = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [math.cos(half), 0.0, math.sin(half), 0.0],
            [math.cos(half), math.sin(half), 0.0, 0.0],
        ]
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            site_pos_w=torch.zeros(3, 2, 3),
            root_link_pos_w=torch.tensor(
                [[0.0, 0.0, 0.13], [0.012, 0.0, 0.13], [0.0, -0.010, 0.13]]
            ),
            root_link_quat_w=quats,
        )
    )
    env = SimpleNamespace(
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_command=lambda _name: commands),
    )
    robot_cfg = _resolved_cfg("robot", site_ids=[0, 1])
    trunk_score = microduck_mdp.mario_commanded_trunk_offset_reward(
        env, robot_cfg=robot_cfg
    )
    lean_score = microduck_mdp.mario_commanded_lean_reward(env)
    assert torch.allclose(trunk_score, torch.ones(3))
    assert torch.allclose(lean_score, torch.ones(3), atol=1e-6)


def test_button_reward_drops_to_zero_when_either_foot_leaves_its_pad():
    names = list(microduck_mdp._MARIO_NES_JOINTS)
    pad_centers = torch.tensor([[[0.007, 0.041, 0.009], [0.007, -0.041, 0.009]]] * 3)
    feet = pad_centers.clone()
    feet[1, 0, 0] += 0.04

    class FakeController:
        data = SimpleNamespace(
            joint_pos=torch.tensor([[math.radians(0.6), 0.0, 0.0, 0.0]] * 3),
            body_link_pos_w=pad_centers,
        )

        @staticmethod
        def find_joints(patterns):
            name = patterns[0].removeprefix("^").removesuffix("$")
            return [names.index(name)], [name]

    class FakeScene(dict):
        def __init__(self, **entities):
            super().__init__(entities)
            self.sensors = {
                "feet_ground_contact": SimpleNamespace(
                    data=SimpleNamespace(
                        found=torch.tensor(
                            [[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
                        )
                    )
                )
            }

    robot = SimpleNamespace(data=SimpleNamespace(site_pos_w=feet))
    env = SimpleNamespace(
        num_envs=3,
        device=torch.device("cpu"),
        scene=FakeScene(robot=robot, nes_controller=FakeController()),
        command_manager=SimpleNamespace(
            get_command=lambda _name: torch.tensor([[1.0, 0.0, 0.0]] * 3)
        ),
    )
    score = microduck_mdp.mario_requested_button_reward(
        env,
        anchor_radius=0.025,
        anchor_sensor_name="feet_ground_contact",
        robot_cfg=_resolved_cfg("robot", site_ids=[0, 1]),
        controller_cfg=_resolved_cfg("nes_controller", body_ids=[0, 1]),
    )
    assert score.tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_foot_anchor_and_planar_speed_costs_penalize_walking():
    pad_centers = torch.tensor([[[0.0, 0.04, 0.0], [0.0, -0.04, 0.0]]])
    feet = pad_centers.clone()
    feet[:, 0, 0] += 0.028
    robot = SimpleNamespace(
        data=SimpleNamespace(
            site_pos_w=feet,
            site_lin_vel_w=torch.tensor([[[0.10, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        )
    )
    controller = SimpleNamespace(data=SimpleNamespace(body_link_pos_w=pad_centers))
    env = SimpleNamespace(scene={"robot": robot, "nes_controller": controller})
    robot_cfg = _resolved_cfg("robot", site_ids=[0, 1])
    controller_cfg = _resolved_cfg("nes_controller", body_ids=[0, 1])
    anchor = microduck_mdp.mario_foot_anchor_cost(
        env,
        deadzone=0.008,
        scale=0.020,
        robot_cfg=robot_cfg,
        controller_cfg=controller_cfg,
    )
    speed = microduck_mdp.mario_foot_planar_speed_cost(
        env, speed_scale=0.10, robot_cfg=robot_cfg
    )
    assert anchor.item() == pytest.approx(0.5)
    assert speed.item() == pytest.approx(0.5)


def test_camera_readiness_rejects_crouch_low_camera_tilt_and_bad_view():
    angle = math.radians(25.0)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor(
                [
                    [0.0, 0.0, 0.13],
                    [0.0, 0.0, 0.10],
                    [0.0, 0.0, 0.13],
                    [0.0, 0.0, 0.13],
                    [0.0, 0.0, 0.13],
                ]
            ),
            site_pos_w=torch.tensor(
                [
                    [[0.0, 0.0, 0.26]],
                    [[0.0, 0.0, 0.26]],
                    [[0.0, 0.0, 0.19]],
                    [[0.0, 0.0, 0.26]],
                    [[0.0, 0.0, 0.26]],
                ]
            ),
            site_quat_w=torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0]]] * 4
                + [[[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]]]
            ),
            root_link_quat_w=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]] * 3
                + [[math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0]]
                + [[1.0, 0.0, 0.0, 0.0]]
            ),
        )
    )

    class FakeScene(dict):
        def __init__(self):
            super().__init__(robot=robot)
            self.terrain = SimpleNamespace(env_origins=torch.zeros(5, 3))

    env = SimpleNamespace(scene=FakeScene())
    camera_cfg = _resolved_cfg("robot", site_ids=[0])
    readiness = microduck_mdp.mario_camera_ready(env, camera_cfg)
    crouch = microduck_mdp.mario_crouch_cost(env, camera_cfg)
    assert readiness.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0])
    assert crouch[0].item() == 0.0
    assert crouch[1].item() > 0.0
    assert crouch[2].item() > 0.0
