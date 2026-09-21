import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import pytest
import torch
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
        rewards["requested_button"].weight
        + rewards["requested_button_progress"].weight
    )
    assert rewards["upright"].weight > total_button_weight
    assert rewards["foot_contact_loss"].weight < -total_button_weight
    assert rewards["body_ang_vel"].weight == -0.15
    assert rewards["angular_momentum"].weight == -0.05
    assert rewards["pose"].weight == 2.0
    pose_params = rewards["pose"].params
    assert pose_params["std_walking"] == pose_params["std_standing"]
    assert pose_params["std_running"] == pose_params["std_standing"]


def test_spawn_is_aligned_with_fixed_pad_layout():
    pose = make_microduck_mario_env_cfg().events["reset_base"].params["pose_range"]
    assert pose["x"] == (0.0, 0.0)
    assert pose["y"] == (0.0, 0.0)
    assert pose["z"] == (0.135, 0.135)
    assert pose["yaw"] == (0.0, 0.0)


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
    press_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "passive_ab_press"
    )
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
        data=SimpleNamespace(
            found=torch.tensor([[3.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
        )
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
