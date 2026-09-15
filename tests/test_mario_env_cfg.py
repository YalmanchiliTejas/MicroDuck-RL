from pathlib import Path

import mujoco
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_mario_env_cfg import (
    MicroduckMarioRlCfg,
    make_microduck_mario_env_cfg,
)


def test_mario_scene_keeps_robot_first_and_adds_controller():
    cfg = make_microduck_mario_env_cfg()
    assert list(cfg.scene.entities) == ["robot", "controller_pads"]


def test_mario_command_uses_existing_three_dimensional_twist_slot():
    cfg = make_microduck_mario_env_cfg()
    assert cfg.commands["twist"].class_type is microduck_mdp.MarioButtonCommand
    assert "head_pose" not in cfg.commands
    assert "body_pose" not in cfg.commands
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        assert terms["command"].func is mdp.generated_commands
        assert terms["command"].params["command_name"] == "twist"
        assert terms["head_command"].params["dim"] == 4
        assert terms["body_command"].params["dim"] == 6


def test_pad_state_is_privileged_and_not_added_to_actor():
    cfg = make_microduck_mario_env_cfg()
    assert "controller_pad_travel" not in cfg.observations["actor"].terms
    assert "controller_pad_travel" in cfg.observations["critic"].terms


def test_pad_reward_signs_cannot_reward_wrong_button():
    rewards = make_microduck_mario_env_cfg().rewards
    assert rewards["requested_pad"].weight > 0.0
    assert rewards["unrequested_pad"].weight < 0.0
    for name in ("track_linear_velocity", "air_time", "foot_clearance"):
        assert name not in rewards
    assert rewards["requested_pad"].params["release_travel"] == 0.002
    assert rewards["requested_pad"].params["press_travel"] == 0.004


def test_spawn_is_aligned_with_fixed_pad_layout():
    pose = make_microduck_mario_env_cfg().events["reset_base"].params["pose_range"]
    assert pose["x"] == (0.0, 0.0)
    assert pose["y"] == (0.0, 0.0)
    assert pose["yaw"] == (0.0, 0.0)


def test_mario_runner_has_distinct_experiment_name():
    assert MicroduckMarioRlCfg.experiment_name == "mario_controller"


def test_unloaded_pads_settle_below_release_threshold():
    path = (
        Path(__file__).parents[1]
        / "src/mjlab_microduck/robot/microduck/controller_pads.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    for _ in range(2_000):
        mujoco.mj_step(model, data)
    for name in ("passive_left_pad", "passive_right_pad", "passive_jump_pad"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        travel = -float(data.qpos[model.jnt_qposadr[joint_id]])
        assert travel < 0.002
