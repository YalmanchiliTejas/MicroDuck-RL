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
    weights = cfg.commands["twist"].category_weights
    assert [index for index, weight in enumerate(weights) if weight > 0.0] == [
        0, 1, 2, 5
    ]
    assert cfg.commands["twist"].resampling_time_range == (1.5, 2.5)
    assert cfg.commands["twist"].neutral_transition_time_range == (0.5, 0.5)
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
    assert rewards["left_button_progress"].weight > 0.0
    assert rewards["right_button_progress"].weight > 0.0
    assert rewards["unrequested_button"].weight < 0.0
    assert rewards["foot_contact_loss"].weight < 0.0
    for name in ("track_linear_velocity", "air_time", "foot_clearance"):
        assert name not in rewards
    params = rewards["requested_button"].params
    assert params["release_angle"] < params["activate_angle"]
    assert params["activate_angle"] == pytest.approx(0.0007)
    assert params["release_angle"] == pytest.approx(0.0003)
    assert params["chord_release_travel"] == 0.0005
    assert params["chord_press_travel"] == 0.0007
    assert params["use_chord"] is False
    assert params["enabled_buttons"] == (
        False, False, True, True, True, True
    )
    assert rewards["foot_anchor"].func is microduck_mdp.mario_commanded_foot_anchor_cost


def test_task_reward_dominates_idle_posture_credit():
    rewards = make_microduck_mario_env_cfg().rewards
    survival_budget = sum(rewards[name].weight for name in (
        "upright", "pose", "commanded_trunk_offset", "standing_height",
        "neutral_head_pose", "alive",
    ))
    single_press_budget = rewards["requested_button"].weight + max(
        rewards[f"{leg}_requested_button"].weight
        + rewards[f"{leg}_button_progress"].weight
        for leg in ("left", "right")
    )
    assert single_press_budget > survival_budget
    assert rewards["upright"].func is microduck_mdp.mario_commanded_lean_reward
    assert rewards["foot_contact_loss"].weight <= -8.0
    assert rewards["body_ang_vel"].weight == -0.40
    assert rewards["angular_momentum"].weight == -0.15
    assert (
        rewards["angular_momentum"].func
        is microduck_mdp.mario_normalized_angular_momentum_cost
    )
    assert rewards["angular_momentum"].params["reference"] == pytest.approx(0.01)
    assert rewards["pose"].weight == 0.0
    pose_params = rewards["pose"].params
    assert pose_params["std_walking"] == pose_params["std_standing"]
    assert pose_params["std_running"] == pose_params["std_standing"]
    assert rewards["commanded_trunk_offset"].weight > 0.0
    assert rewards["commanded_trunk_offset"].params["forward_offset"] == 0.0
    assert rewards["commanded_trunk_offset"].params["lateral_offset"] == 0.0
    assert rewards["commanded_trunk_offset"].params["transition_grace_s"] == 0.0
    assert rewards["upright"].params["lean_angle"] == 0.0
    assert rewards["standing_height"].params["target_height"] == 0.130
    assert rewards["camera_crouch"].weight < 0.0
    assert "camera_view" not in rewards
    assert rewards["camera_crouch"].params["trunk_floor"] == 0.128
    assert rewards["camera_crouch"].params["camera_floor"] == 0.230
    assert rewards["leg_pose_l1"].weight < 0.0
    assert (
        rewards["neutral_head_pose"].func
        is microduck_mdp.mario_selected_pose_reward
    )
    assert rewards["foot_anchor"].weight < 0.0
    assert rewards["foot_planar_speed"].weight < 0.0
    assert rewards["commanded_foot_clearance"].weight == 0.0
    assert rewards["commanded_foot_clearance"].params["lift_height"] == 0.0
    assert (
        rewards["foot_contact_loss"].func
        is microduck_mdp.feet_contact_loss_cost
    )
    assert rewards["foot_planar_speed"].params["sensor_name"] == (
        "feet_ground_contact"
    )
    foot_pose = rewards["commanded_foot_pose"].params
    assert foot_pose["position_offset"] == pytest.approx(0.012)
    assert foot_pose["lateral_offset"] == pytest.approx(0.0105)
    assert foot_pose["left_neutral_x"] == pytest.approx(-0.0075)
    assert foot_pose["target_tilt"] == pytest.approx(math.radians(6.0))
    for name in ("requested_button", "left_requested_button", "right_requested_button"):
        assert "foot_pose_params" not in rewards[name].params
    assert rewards["foot_planar_speed"].params["transition_grace_s"] == (
        pytest.approx(0.25)
    )


def test_mario_runner_reduces_entropy_pressure_for_stationary_control():
    from mjlab_microduck.tasks.microduck_mario_env_cfg import MicroduckMarioRlCfg

    assert MicroduckMarioRlCfg.actor.distribution_cfg["init_std"] == pytest.approx(0.20)
    assert MicroduckMarioRlCfg.algorithm.entropy_coef == pytest.approx(0.002)


def test_mario_action_smoothing_does_not_tighten_on_a_clock():
    curriculum = make_microduck_mario_env_cfg().curriculum["action_rate_weight"]
    stages = curriculum.params["weight_stages"]
    assert [stage["weight"] for stage in stages] == [-0.02]


def test_mario_com_randomization_stays_at_five_mm():
    curricula = make_microduck_mario_env_cfg().curriculum
    for name in ("com_range", "head_com_range"):
        assert curricula[name].params["range_stages"] == [
            {"step": 0, "range": 0.005},
        ]


def test_mario_angular_momentum_is_normalized_to_robot_scale():
    momentum = torch.tensor([[0.006, 0.008, 0.0], [0.0, 0.0, 0.020]])
    sensor = SimpleNamespace(data=momentum)
    scene = SimpleNamespace(sensors={"robot/root_angmom": sensor})
    env = SimpleNamespace(scene=scene, extras={"log": {}})

    cost = microduck_mdp.mario_normalized_angular_momentum_cost(
        env, "robot/root_angmom", reference=0.01
    )

    assert cost.tolist() == pytest.approx([1.0, 4.0])
    assert env.extras["log"]["Metrics/angular_momentum_mean"] == pytest.approx(
        0.015
    )


def test_mario_never_inherits_velocity_pushes_even_in_play():
    assert "push_robot" not in make_microduck_mario_env_cfg(play=False).events
    assert "push_robot" not in make_microduck_mario_env_cfg(play=True).events


def test_button_rewards_are_gated_by_both_foot_anchors():
    rewards = make_microduck_mario_env_cfg().rewards
    for name in ("requested_button", "left_requested_button", "right_requested_button",
                 "left_button_progress", "right_button_progress"):
        params = rewards[name].params
        assert params["anchor_radius"] > 0.0
        assert params["anchor_sensor_name"] == "feet_ground_contact"
        assert params["camera_cfg"].site_names == ("head_camera",)
        assert params["min_trunk_height"] < params["full_trunk_height"]
        assert params["min_camera_height"] < params["full_camera_height"]
        assert params["min_view_alignment"] < params["full_view_alignment"]
        assert params["standing_pose_cfg"].joint_names == (
            r"^(?!passive_|.*neck.*|.*head.*).*",
        )
        assert params["full_pose_error"] < params["max_pose_error"]
        assert params["require_exclusive"] is True
        if name.endswith("button_progress"):
            assert params["transition_grace_s"] == 0.0
        else:
            assert params["transition_grace_s"] > 0.0
        assert params["robot_cfg"].site_names == ("left_foot", "right_foot")
        assert params["controller_cfg"].body_names == (
            "dpad_platform",
            "ab_platform",
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


def test_mario_command_curriculum_stages_singles_then_jump_combos():
    train_cfg = make_microduck_mario_env_cfg(play=False)
    play_cfg = make_microduck_mario_env_cfg(play=True)
    curriculum = train_cfg.curriculum["mario_command_stage"]
    stages = curriculum.params["weight_stages"]
    assert stages[0]["step"] == 0
    assert len(stages) == 3
    assert [i for i, weight in enumerate(stages[0]["weights"]) if weight > 0] == [
        0, 1, 2, 5
    ]
    assert stages[0]["weights"][1] == pytest.approx(0.40)
    assert stages[1]["weights"] == (0.25, 0.25, 0.25, 0, 0, 0.25, 0,
                                     0, 0, 0, 0, 0, 0, 0)
    assert stages[2]["step"] == 2_500 * 24
    assert stages[2]["weights"][8] > 0.0
    assert stages[2]["weights"][11] > 0.0
    assert stages[2]["weights"][7] == 0.0
    assert stages[2]["weights"][10] == 0.0
    assert stages[2]["weights"][13] == 0.0
    assert curriculum.params["discovery_button_index"] == 2
    assert curriculum.params["discovery_unlock_success"] == pytest.approx(0.25)
    assert train_cfg.commands["twist"].category_weights == stages[0]["weights"]
    assert "mario_command_stage" not in play_cfg.curriculum
    assert "mario_transition_stage" not in play_cfg.curriculum
    assert play_cfg.commands["twist"].neutral_transition_time_range is None
    assert play_cfg.commands["twist"].category_weights == stages[1]["weights"]


def test_mario_combo_finetune_samples_combinations_without_dropping_singles(
    monkeypatch,
):
    monkeypatch.setenv("MARIO_COMBO_FINETUNE", "1")
    cfg = make_microduck_mario_env_cfg(play=False)
    weights = cfg.commands["twist"].category_weights
    assert sum(weights) == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.10)  # LEFT
    assert weights[2] == pytest.approx(0.10)  # RIGHT
    assert weights[5] == pytest.approx(0.20)  # JUMP
    assert weights[8] == pytest.approx(0.30)  # LEFT+JUMP
    assert weights[11] == pytest.approx(0.30)  # RIGHT+JUMP
    assert cfg.commands["twist"].neutral_transition_time_range == (0.5, 0.5)
    assert cfg.commands["twist"].allow_sampled_neutral is False
    assert "mario_command_stage" not in cfg.curriculum
    assert "mario_transition_stage" not in cfg.curriculum

    # The environment flag is training-only; evaluation remains balanced and
    # continues to expose direct transitions for an honest checkpoint test.
    play_cfg = make_microduck_mario_env_cfg(play=True)
    assert play_cfg.commands["twist"].category_weights == (
        0.25, 0.25, 0.25, 0, 0, 0.25, 0,
        0, 0, 0, 0, 0, 0, 0,
    )
    assert play_cfg.commands["twist"].neutral_transition_time_range is None


def test_mario_combo_finetune_flag_rejects_typos(monkeypatch):
    monkeypatch.setenv("MARIO_COMBO_FINETUNE", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        make_microduck_mario_env_cfg()


def test_mario_consolidation_rehearses_neutral_and_every_deployed_action(
    monkeypatch,
):
    monkeypatch.setenv("MARIO_CONSOLIDATION_FINETUNE", "1")
    cfg = make_microduck_mario_env_cfg(play=False)
    weights = cfg.commands["twist"].category_weights
    assert sum(weights) == pytest.approx(1.0)
    assert weights[0] == pytest.approx(0.20)  # sustained neutral
    assert weights[1] == pytest.approx(0.20)  # LEFT
    assert weights[2] == pytest.approx(0.10)  # RIGHT
    assert weights[5] == pytest.approx(0.15)  # JUMP
    assert weights[8] == pytest.approx(0.15)  # LEFT+JUMP
    assert weights[11] == pytest.approx(0.20)  # RIGHT+JUMP
    assert cfg.commands["twist"].allow_sampled_neutral is True
    assert cfg.commands["twist"].neutral_transition_time_range == (0.5, 0.5)
    assert "mario_command_stage" not in cfg.curriculum
    assert "mario_transition_stage" not in cfg.curriculum


def test_mario_finetune_modes_are_mutually_exclusive(monkeypatch):
    monkeypatch.setenv("MARIO_COMBO_FINETUNE", "1")
    monkeypatch.setenv("MARIO_CONSOLIDATION_FINETUNE", "1")
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_microduck_mario_env_cfg()


def test_mario_transition_curriculum_waits_for_competence_before_direct_switches():
    cfg = make_microduck_mario_env_cfg()
    params = cfg.curriculum["mario_transition_stage"].params
    term = object.__new__(microduck_mdp.MarioNesCommand)
    term.cfg = SimpleNamespace(neutral_transition_time_range=(0.5, 0.5))
    env = SimpleNamespace(
        common_step_counter=4_000 * 24,
        device=torch.device("cpu"),
        command_manager=SimpleNamespace(get_term=lambda _name: term),
    )
    stage = microduck_mdp.mario_transition_settle_curriculum(
        env, torch.arange(1), **params
    )
    assert stage.item() == 0.0
    assert term.cfg.neutral_transition_time_range == (0.5, 0.5)

    term._mario_combo_unlocked = True
    term._mario_combo_unlocked_step = 2_500 * 24
    env.common_step_counter = 3_000 * 24
    stage = microduck_mdp.mario_transition_settle_curriculum(
        env, torch.arange(1), **params
    )
    assert stage.item() == 1.0
    assert term.cfg.neutral_transition_time_range == (0.25, 0.25)

    # A late competence unlock must still receive a full 1000 iterations at
    # the shorter settling stage instead of jumping directly to no buffer.
    term._mario_combo_unlocked_step = 3_400 * 24
    env.common_step_counter = 3_500 * 24
    stage = microduck_mdp.mario_transition_settle_curriculum(
        env, torch.arange(1), **params
    )
    assert stage.item() == 1.0
    assert term.cfg.neutral_transition_time_range == (0.25, 0.25)

    env.common_step_counter = 4_400 * 24
    stage = microduck_mdp.mario_transition_settle_curriculum(
        env, torch.arange(1), **params
    )
    assert stage.item() == 2.0
    assert term.cfg.neutral_transition_time_range is None


def test_mario_command_inserts_neutral_between_active_requests():
    term = object.__new__(microduck_mdp.MarioNesCommand)
    term._env = SimpleNamespace(
        num_envs=2, device=torch.device("cpu"), step_dt=0.02
    )
    term.cfg = SimpleNamespace(
        category_weights=(0.0, 1.0) + (0.0,) * 12,
        neutral_transition_time_range=(0.5, 0.5),
    )
    term._command = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    term.command_age = torch.ones(2)
    term.time_left = torch.full((2,), 2.0)
    term._resample_command(torch.arange(2))
    assert term._command.tolist() == [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    assert term.time_left.tolist() == pytest.approx([0.5, 2.0])
    assert term.command_age.tolist() == pytest.approx([-0.02, -0.02])


def test_mario_command_can_sample_sustained_neutral_after_settling():
    term = object.__new__(microduck_mdp.MarioNesCommand)
    term._env = SimpleNamespace(
        num_envs=1, device=torch.device("cpu"), step_dt=0.02
    )
    term.cfg = SimpleNamespace(
        category_weights=(1.0,) + (0.0,) * 13,
        neutral_transition_time_range=(0.5, 0.5),
        allow_sampled_neutral=True,
    )
    term._command = torch.zeros(1, 3)
    term.command_age = torch.ones(1)
    term.time_left = torch.full((1,), 2.0)
    term._resample_command(torch.arange(1))
    assert term._command.tolist() == [[0.0, 0.0, 0.0]]
    # The command manager keeps the normal resampling time rather than the
    # short transition-settle time, producing genuine idle rehearsal.
    assert term.time_left.tolist() == pytest.approx([2.0])
    assert term.command_age.tolist() == pytest.approx([-0.02])


def test_mario_external_command_survives_episode_resampling():
    term = object.__new__(microduck_mdp.MarioNesCommand)
    term._env = SimpleNamespace(
        num_envs=2, device=torch.device("cpu"), step_dt=0.02
    )
    term._command = torch.zeros(2, 3)
    term.command_age = torch.ones(2)
    term.time_left = torch.zeros(2)
    term._external_command = None

    term.set_external_command(torch.tensor([1.0, 0.0, 1.0]))
    assert term.command.tolist() == [[1.0, 0.0, 1.0]] * 2
    assert torch.isinf(term.time_left).all()
    assert term.command_age.tolist() == pytest.approx([-0.02, -0.02])

    # This is the path CommandManager takes when an episode auto-resets.
    term._command.zero_()
    term.time_left.zero_()
    term.command_age.fill_(3.0)
    term._resample_command(torch.tensor([1]))
    assert term.command.tolist() == [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]
    assert torch.isinf(term.time_left[1])
    assert term.command_age[1].item() == pytest.approx(-0.02)


def test_mario_external_command_age_resets_only_on_change():
    term = object.__new__(microduck_mdp.MarioNesCommand)
    term._env = SimpleNamespace(
        num_envs=1, device=torch.device("cpu"), step_dt=0.02
    )
    term._command = torch.tensor([[1.0, 0.0, 0.0]])
    term.command_age = torch.tensor([1.25])
    term.time_left = torch.zeros(1)
    term._external_command = None

    term.set_external_command(torch.tensor([1.0, 0.0, 0.0]))
    assert term.command_age.item() == pytest.approx(1.25)
    term.set_external_command(torch.tensor([-1.0, 0.0, 0.0]))
    assert term.command_age.item() == pytest.approx(-0.02)


def test_mario_balance_reward_closes_press_then_crash_loophole():
    cfg = make_microduck_mario_env_cfg()
    rewards = cfg.rewards
    assert rewards["alive"].func is microduck_mdp.is_alive
    assert rewards["alive"].weight > 0.0
    assert rewards["fell_over"].func is microduck_mdp.mario_fall_cost
    assert rewards["fell_over"].weight < 0.0
    policy_step_dt = cfg.sim.mujoco.timestep * cfg.decimation
    assert rewards["fell_over"].weight * policy_step_dt == pytest.approx(-10.0)
    assert rewards["commanded_trunk_offset"].weight >= 1.0
    assert rewards["commanded_trunk_offset"].params["transition_grace_s"] == 0.0
    assert rewards["upright"].weight >= 2.0


def test_mario_fall_cost_only_reads_named_failure_term():
    terms = {
        "fell_over": torch.tensor([False, True, False]),
        "time_out": torch.tensor([True, False, False]),
    }
    env = SimpleNamespace(
        termination_manager=SimpleNamespace(get_term=terms.__getitem__)
    )
    assert microduck_mdp.mario_fall_cost(env).tolist() == [0.0, 1.0, 0.0]


def test_mario_curriculum_rebalances_left_before_unlocking_combos():
    curriculum = make_microduck_mario_env_cfg().curriculum["mario_command_stage"]
    params = curriculum.params
    stages = params["weight_stages"]
    term = SimpleNamespace(
        cfg=SimpleNamespace(category_weights=stages[0]["weights"])
    )
    commands = torch.tensor(
        [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
         [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    metrics = SimpleNamespace(
        active_terms=["requested_button_success", "command_ready"],
        _step_values=torch.tensor(
            [[1.0, 1.0], [0.2, 1.0], [0.8, 1.0], [0.8, 1.0]]
        ),
    )
    env = SimpleNamespace(
        common_step_counter=2_500 * 24,
        device=torch.device("cpu"),
        command_manager=SimpleNamespace(
            get_term=lambda _name: term,
            get_command=lambda _name: commands,
        ),
        metrics_manager=metrics,
    )
    stage = microduck_mdp.mario_command_category_curriculum(
        env, torch.arange(4), **params
    )
    assert stage.item() == 0.0
    assert term.cfg.category_weights == stages[0]["weights"]
    env.common_step_counter += 250
    metrics._step_values[:, 0] = torch.tensor([1.0, 0.9, 0.2, 0.2])
    stage = microduck_mdp.mario_command_category_curriculum(
        env, torch.arange(4), **params
    )
    assert stage.item() == 1.0
    assert term.cfg.category_weights == stages[1]["weights"]
    env.common_step_counter += 250
    # Two perfect skills must not hide an absent third skill (mean = 2/3).
    metrics._step_values[1:, 0] = torch.tensor([1.0, 1.0, 0.0])
    stage = microduck_mdp.mario_command_category_curriculum(
        env, torch.arange(4), **params
    )
    assert stage.item() == 1.0
    # Missing observations for JUMP must not count as evidence of readiness.
    env.common_step_counter += 250
    metrics._step_values[3, 1] = 0.0
    metrics._step_values[1:, 0] = 0.9
    stage = microduck_mdp.mario_command_category_curriculum(
        env, torch.arange(4), **params
    )
    assert stage.item() == 1.0
    env.common_step_counter += 250
    metrics._step_values[3, 1] = 1.0
    metrics._step_values[1:, 0] = 0.9
    stage = microduck_mdp.mario_command_category_curriculum(
        env, torch.arange(4), **params
    )
    assert stage.item() == 2.0
    assert term.cfg.category_weights == stages[2]["weights"]


def test_mario_command_grace_tracks_time_since_resample():
    term = object.__new__(microduck_mdp.MarioNesCommand)
    term.command_age = torch.tensor([0.05, 0.20])
    term._command = torch.zeros(2, 3)
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _name: term)
    )
    ready = microduck_mdp.mario_command_ready(
        env, transition_grace_s=0.15
    )
    assert ready.tolist() == [0.0, 1.0]


def test_transition_contact_cost_allows_one_lift_but_never_zero_support():
    term = object.__new__(microduck_mdp.MarioNesCommand)
    term.command_age = torch.tensor([0.10, 0.10, 0.10, 0.30, 0.30, 0.30])
    term._command = torch.zeros(6, 3)
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            found=torch.tensor(
                [
                    [1.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 0.0],
                ]
            )
        )
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(sensors={"feet": sensor}),
        command_manager=SimpleNamespace(get_term=lambda _name: term),
    )

    cost = microduck_mdp.mario_transition_contact_loss_cost(
        env, sensor_name="feet", transition_grace_s=0.25
    )

    assert cost.tolist() == pytest.approx([0.0, 0.0, 1.0, 0.0, 0.5, 1.0])


def test_unloaded_controller_settles_inside_all_release_thresholds():
    path = (
        Path(__file__).parents[1]
        / "src/mjlab_microduck/robot/microduck/controller_nes.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    for _ in range(2_000):
        mujoco.mj_step(model, data)
    for name in microduck_mdp._MARIO_NES_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        travel = -float(data.qpos[model.jnt_qposadr[joint_id]])
        assert travel < 0.0003


def test_controller_contains_no_shared_rocker_or_chord_joint():
    path = (
        Path(__file__).parents[1]
        / "src/mjlab_microduck/robot/microduck/controller_nes.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(path))
    joint_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    }
    assert joint_names == set(microduck_mdp._MARIO_NES_JOINTS)
    assert not any("rocker" in name or "press" in name for name in joint_names)


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


def test_independent_a_key_does_not_activate_b():
    names = list(microduck_mdp._MARIO_NES_JOINTS)

    class FakeController:
        data = SimpleNamespace(
            joint_pos=torch.tensor([[0.0, 0.0, -0.0010, 0.0]])
        )

        @staticmethod
        def find_joints(patterns):
            name = patterns[0].removeprefix("^").removesuffix("$")
            return [names.index(name)], [name]

    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        scene={"nes_controller": FakeController()},
    )
    game_activation = microduck_mdp.mario_nes_activation(env)
    game_progress = microduck_mdp.mario_nes_progress(env)
    assert game_activation[0, 4:].tolist() == [1.0, 0.0]
    assert game_progress[0, 4:].tolist() == [1.0, 0.0]


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
            joint_pos=torch.tensor([[0.0, -0.0002, 0.0, 0.0]])
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


def test_requested_button_requires_wrong_buttons_to_be_released():
    names = list(microduck_mdp._MARIO_NES_JOINTS)

    class FakeController:
        data = SimpleNamespace(
            # RIGHT is requested and fully active, but A is active too.
            joint_pos=torch.tensor([[0.0, -0.0007, -0.0007, 0.0]])
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
    permissive = microduck_mdp.mario_requested_button_reward(env)
    exclusive = microduck_mdp.mario_requested_button_reward(
        env, require_exclusive=True
    )
    success = microduck_mdp.mario_clean_button_success(
        env, require_exclusive=True
    )
    assert permissive.item() == pytest.approx(1.0)
    assert exclusive.item() < 1e-5
    assert success.item() == 0.0


def test_clean_success_counts_released_neutral_as_success():
    names = list(microduck_mdp._MARIO_NES_JOINTS)

    class FakeController:
        data = SimpleNamespace(joint_pos=torch.zeros(1, 4))

        @staticmethod
        def find_joints(patterns):
            name = patterns[0].removeprefix("^").removesuffix("$")
            return [names.index(name)], [name]

    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        scene={"nes_controller": FakeController()},
        command_manager=SimpleNamespace(
            get_command=lambda _name: torch.zeros(1, 3)
        ),
    )
    success = microduck_mdp.mario_clean_button_success(
        env,
        enabled_buttons=(False, False, True, True, True, False),
    )
    assert success.item() == 1.0


def test_clean_success_thresholds_readiness_components_not_their_product(
    monkeypatch,
):
    names = list(microduck_mdp._MARIO_NES_JOINTS)

    class FakeController:
        data = SimpleNamespace(
            joint_pos=torch.tensor([[0.0, -0.0007, 0.0, 0.0]])
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
    monkeypatch.setattr(
        microduck_mdp,
        "mario_camera_ready",
        lambda *args, **kwargs: torch.tensor([0.8]),
    )
    monkeypatch.setattr(
        microduck_mdp,
        "mario_standing_pose_ready",
        lambda *args, **kwargs: torch.tensor([0.8]),
    )
    success = microduck_mdp.mario_clean_button_success(
        env,
        camera_cfg=SceneEntityCfg("robot"),
        standing_pose_cfg=SceneEntityCfg("robot"),
        enabled_buttons=(False, False, True, True, True, False),
    )
    assert success.item() == 1.0


def test_all_physical_keys_participate_in_wrong_button_exclusivity():
    names = list(microduck_mdp._MARIO_NES_JOINTS)

    class FakeController:
        data = SimpleNamespace(
            # RIGHT is requested and active, but B is active too.
            joint_pos=torch.tensor([[0.0, -0.0007, 0.0, -0.0007]])
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
    enabled = (True, True, True, True, True, True)
    score = microduck_mdp.mario_requested_button_reward(
        env, require_exclusive=True, enabled_buttons=enabled
    )
    wrong_cost = microduck_mdp.mario_unrequested_button_cost(
        env, enabled_buttons=enabled
    )
    assert score.item() < 1e-6
    assert wrong_cost.item() == pytest.approx(1.0)


def test_unrequested_button_cost_keeps_growing_past_activation():
    names = list(microduck_mdp._MARIO_NES_JOINTS)

    class FakeController:
        data = SimpleNamespace(
            joint_pos=torch.tensor(
                [
                    [0.0, 0.0, -0.0007, 0.0],
                    [0.0, 0.0, -0.0013, 0.0],
                ]
            )
        )

        @staticmethod
        def find_joints(patterns):
            name = patterns[0].removeprefix("^").removesuffix("$")
            return [names.index(name)], [name]

    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        scene={"nes_controller": FakeController()},
        # Request UP, making A an unrequested button in both samples.
        command_manager=SimpleNamespace(
            get_command=lambda _name: torch.tensor(
                [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
            )
        ),
    )
    cost = microduck_mdp.mario_unrequested_button_cost(env)
    assert cost.tolist() == pytest.approx([1.0, 2.5], abs=1e-5)


def test_unrequested_button_cost_ignores_motion_inside_release_deadband():
    names = list(microduck_mdp._MARIO_NES_JOINTS)

    class FakeController:
        data = SimpleNamespace(
            joint_pos=torch.tensor(
                [
                    [-0.0002, 0.0, 0.0, 0.0],
                    [-0.0004, 0.0, 0.0, 0.0],
                ]
            )
        )

        @staticmethod
        def find_joints(patterns):
            name = patterns[0].removeprefix("^").removesuffix("$")
            return [names.index(name)], [name]

    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        scene={"nes_controller": FakeController()},
        command_manager=SimpleNamespace(
            get_command=lambda _name: torch.zeros(2, 3)
        ),
    )
    cost = microduck_mdp.mario_unrequested_button_cost(env)
    assert cost.tolist() == pytest.approx([0.0, 0.25], abs=1e-5)


def test_calibrated_foot_targets_drive_only_requested_mario_axes():
    from mjlab_microduck.tasks.microduck_mario_env_cfg import (
        DPAD_POSITION_OFFSET,
        FOOT_POSITION_OFFSET,
        LEFT_NEUTRAL_FOOT_X,
        LEFT_NEUTRAL_FOOT_Y,
        RIGHT_NEUTRAL_FOOT_X,
        RIGHT_NEUTRAL_FOOT_Y,
    )

    commands = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # neutral
            [-1.0, 0.0, 0.0],  # LEFT
            [1.0, 0.0, 0.0],  # RIGHT
            [0.0, 0.0, 1.0],  # A
            [0.0, 0.0, -1.0],  # B
            [1.0, 0.0, 1.0],  # RIGHT+A
        ]
    )
    target = microduck_mdp._mario_commanded_foot_target_xy(
        commands,
        FOOT_POSITION_OFFSET,
        DPAD_POSITION_OFFSET,
        LEFT_NEUTRAL_FOOT_X,
        LEFT_NEUTRAL_FOOT_Y,
        RIGHT_NEUTRAL_FOOT_X,
        RIGHT_NEUTRAL_FOOT_Y,
    )
    assert target[0, 0].tolist() == pytest.approx(
        [LEFT_NEUTRAL_FOOT_X, LEFT_NEUTRAL_FOOT_Y]
    )
    assert target[1, 0, 0] == pytest.approx(
        LEFT_NEUTRAL_FOOT_X - DPAD_POSITION_OFFSET
    )
    assert target[2, 0, 0] == pytest.approx(
        LEFT_NEUTRAL_FOOT_X + DPAD_POSITION_OFFSET
    )
    assert target[1, 0, 1] == pytest.approx(LEFT_NEUTRAL_FOOT_Y)
    assert target[2, 0, 1] == pytest.approx(LEFT_NEUTRAL_FOOT_Y)
    assert target[3, 1, 0] == pytest.approx(
        RIGHT_NEUTRAL_FOOT_X + FOOT_POSITION_OFFSET
    )
    assert target[4, 1, 0] == pytest.approx(
        RIGHT_NEUTRAL_FOOT_X - FOOT_POSITION_OFFSET
    )
    assert target[5, 0].tolist() == pytest.approx(target[2, 0].tolist())
    assert target[5, 1].tolist() == pytest.approx(target[3, 1].tolist())
    assert target[1, 0, 0] == pytest.approx(-0.018)
    assert target[2, 0, 0] == pytest.approx(0.003)
    assert target[3, 1, 0] == pytest.approx(0.012)
    assert torch.all(torch.linalg.vector_norm(target, dim=-1) < 0.060)


def _resolved_cfg(name, *, site_ids=None, body_ids=None):
    cfg = SceneEntityCfg(name)
    if site_ids is not None:
        cfg.site_ids = site_ids
    if body_ids is not None:
        cfg.body_ids = body_ids
    return cfg


def test_mario_commands_keep_trunk_centered_and_upright():
    commands = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # neutral
            [0.0, 1.0, 0.0],  # up: forward
            [1.0, 0.0, 0.0],  # right: robot-frame -Y CoM, +roll
        ]
    )
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            site_pos_w=torch.zeros(3, 2, 3),
            root_link_quat_w=quats,
            indexing=SimpleNamespace(root_body_id=0),
            data=SimpleNamespace(
                subtree_com=torch.tensor([[[0.0, 0.0, 0.13]]] * 3)
            ),
        )
    )
    env = SimpleNamespace(
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_command=lambda _name: commands),
    )
    robot_cfg = _resolved_cfg("robot", site_ids=[0, 1])
    trunk_score = microduck_mdp.mario_commanded_trunk_offset_reward(
        env, forward_offset=0.0, lateral_offset=0.0, robot_cfg=robot_cfg
    )
    lean_score = microduck_mdp.mario_commanded_lean_reward(env, lean_angle=0.0)
    assert torch.allclose(trunk_score, torch.ones(3))
    assert torch.allclose(lean_score, torch.ones(3), atol=1e-6)


def test_mario_balance_uses_whole_body_com_and_only_contacting_feet():
    feet = torch.tensor(
        [[[-0.02, 0.0, 0.0], [0.02, 0.0, 0.0]]] * 4
    )
    whole_com = torch.tensor(
        [[[-0.02, 0.0, 0.13]], [[-0.02, 0.0, 0.13]],
         [[0.0, 0.0, 0.13]], [[0.0, 0.0, 0.13]]]
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            site_pos_w=feet,
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4),
            indexing=SimpleNamespace(root_body_id=0),
            data=SimpleNamespace(subtree_com=whole_com),
        )
    )
    found = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
    )

    class FakeScene(dict):
        def __init__(self):
            super().__init__(robot=robot)
            self.sensors = {
                "feet": SimpleNamespace(data=SimpleNamespace(found=found))
            }

    env = SimpleNamespace(
        scene=FakeScene(),
        command_manager=SimpleNamespace(
            get_command=lambda _: torch.zeros(4, 3)
        ),
    )
    score = microduck_mdp.mario_commanded_trunk_offset_reward(
        env,
        std=0.004,
        support_radius=0.005,
        sensor_name="feet",
        robot_cfg=_resolved_cfg("robot", site_ids=[0, 1]),
    )
    assert score[0] == pytest.approx(1.0)
    assert score[1] < 1e-6
    assert score[2] == pytest.approx(1.0)
    assert score[3] == 0.0
    cost = microduck_mdp.mario_support_margin_cost(
        env, error_scale=.004, support_radius=.005, sensor_name="feet",
        robot_cfg=_resolved_cfg("robot", site_ids=[0, 1]),
    )
    assert cost.tolist() == pytest.approx([0., 8.75, 0., 4.75])


def test_support_margin_cost_stays_informative_outside_gaussian(monkeypatch):
    cfg = make_microduck_mario_env_cfg()
    term = cfg.rewards["support_margin"]
    balance = cfg.rewards["commanded_trunk_offset"]
    assert term.weight == 0  # Failed experiment is disabled during recovery.
    assert term.params["support_radius"] == balance.params["support_radius"]
    assert term.params["error_scale"] == balance.params["std"]
    assert term.params["sensor_name"] == balance.params["sensor_name"]
    errors = torch.tensor([0., .008, .016, .024, .032])
    monkeypatch.setattr(microduck_mdp, "_mario_support_region_error",
                        lambda *a, **k: (errors, torch.full((5,), 2)))
    cost = term.func(SimpleNamespace(), **term.params)
    assert cost.tolist() == pytest.approx([0., 1., 2., 3., 4.])
    assert (cost * term.weight <= 0).all()
    score = balance.func(SimpleNamespace(), **balance.params)
    assert score[-1] < .000001
    assert cost[-1] > cost[-2]


def test_button_reward_drops_to_zero_when_either_foot_leaves_its_pad():
    names = list(microduck_mdp._MARIO_NES_JOINTS)
    pad_centers = torch.tensor([[[0.007, 0.041, 0.009], [0.007, -0.041, 0.009]]] * 3)
    feet = pad_centers.clone()
    feet[1, 0, 0] += 0.04

    class FakeController:
        data = SimpleNamespace(
            joint_pos=torch.tensor(
                [[0.0, -0.0007, 0.0, 0.0]] * 3
            ),
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


def test_foot_anchor_and_contact_conditioned_speed_costs_penalize_sliding():
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

    class FakeScene(dict):
        def __init__(self):
            super().__init__(robot=robot, nes_controller=controller)
            self.sensors = {
                "feet_ground_contact": SimpleNamespace(
                    data=SimpleNamespace(found=torch.tensor([[1.0, 0.0]]))
                )
            }

    env = SimpleNamespace(scene=FakeScene())
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


def test_commanded_foot_pose_targets_independent_controller_axes():
    position_offset = 0.010
    target_tilt = math.radians(1.0)
    left_roll = math.radians(-5.0)
    right_roll = math.radians(5.0)
    commands = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]  # neutral, RIGHT
    )

    def quat_from_roll_pitch(roll: float, pitch: float) -> list[float]:
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        return [cr * cp, sr * cp, cr * sp, -sr * sp]

    foot_pos = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[position_offset, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    foot_quat = torch.tensor(
        [
            [
                quat_from_roll_pitch(left_roll, 0.0),
                quat_from_roll_pitch(right_roll, 0.0),
            ],
            [
                quat_from_roll_pitch(left_roll, target_tilt),
                quat_from_roll_pitch(right_roll, 0.0),
            ],
        ]
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(site_pos_w=foot_pos, site_quat_w=foot_quat)
    )
    controller = SimpleNamespace(
        data=SimpleNamespace(
            body_link_pos_w=torch.zeros(2, 2, 3),
            body_link_quat_w=torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0]] * 2] * 2
            ),
        )
    )
    env = SimpleNamespace(
        scene={"robot": robot, "nes_controller": controller},
        command_manager=SimpleNamespace(get_command=lambda _name: commands),
    )
    score = microduck_mdp.mario_commanded_foot_pose_reward(
        env,
        command_name="twist",
        position_offset=position_offset,
        target_tilt=target_tilt,
        position_std=0.008,
        angle_std=math.radians(2.0),
        left_nominal_roll=left_roll,
        right_nominal_roll=right_roll,
        robot_cfg=_resolved_cfg("robot", site_ids=[0, 1]),
        controller_cfg=_resolved_cfg("nes_controller", body_ids=[0, 1]),
    )
    assert score.tolist() == pytest.approx([0.0, 1.0], abs=1e-6)


def test_commanded_foot_clearance_lifts_while_misplaced_then_lands_at_target():
    commands = torch.tensor([[1.0, 0.0, 0.0]] * 3)  # RIGHT moves left foot in +X.
    foot_pos = torch.tensor(
        [
            [[0.0, 0.0, 0.006], [0.0, 0.0, 0.006]],  # misplaced, dragging
            [[0.0, 0.0, 0.009], [0.0, 0.0, 0.006]],  # misplaced, 3 mm lift
            [[0.007, 0.0, 0.006], [0.0, 0.0, 0.006]],  # arrived and landed
        ]
    )
    robot = SimpleNamespace(data=SimpleNamespace(site_pos_w=foot_pos))
    controller = SimpleNamespace(
        data=SimpleNamespace(
            body_link_pos_w=torch.zeros(3, 2, 3),
            body_link_quat_w=torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0]] * 2] * 3
            ),
        )
    )
    env = SimpleNamespace(
        scene={"robot": robot, "nes_controller": controller},
        command_manager=SimpleNamespace(get_command=lambda _name: commands),
    )

    score = microduck_mdp.mario_commanded_foot_clearance_reward(
        env,
        command_name="twist",
        position_offset=0.007,
        lateral_offset=0.007,
        left_neutral_x=0.0,
        left_neutral_y=0.0,
        right_neutral_x=0.0,
        right_neutral_y=0.0,
        contact_height=0.006,
        lift_height=0.003,
        full_lift_error=0.005,
        height_std=0.0015,
        robot_cfg=_resolved_cfg("robot", site_ids=[0, 1]),
        controller_cfg=_resolved_cfg("nes_controller", body_ids=[0, 1]),
    )

    assert score[1].item() > score[0].item()
    assert score[1].item() == pytest.approx(1.0)
    assert score[2].item() == pytest.approx(1.0)


def test_standing_pose_gate_and_l1_cost_reject_folded_legs():
    joint_pos = torch.tensor([[0.1, -0.1], [0.6, -0.6]])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=joint_pos,
            default_joint_pos=torch.zeros_like(joint_pos),
        )
    )
    env = SimpleNamespace(scene={"robot": robot})
    cfg = SceneEntityCfg("robot")
    cfg.joint_ids = [0, 1]
    ready = microduck_mdp.mario_standing_pose_ready(env, cfg)
    cost = microduck_mdp.mario_leg_pose_l1_cost(env, cfg)
    assert ready.tolist() == pytest.approx([1.0, 0.0])
    assert cost[1] > cost[0]


def test_mario_cfg_exposes_unweighted_policy_quality_metrics():
    metrics = make_microduck_mario_env_cfg().metrics
    assert set(metrics) >= {
        "camera_ready",
        "standing_pose_ready",
        "requested_button_clean",
        "requested_button_success",
        "feet_anchored",
        "commanded_foot_pose",
        "command_ready",
        "unrequested_button_travel",
    }


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
    components = torch.stack([
        microduck_mdp.mario_camera_ready(env, camera_cfg, component=name)
        for name in ("trunk_height", "camera_height", "trunk_tilt", "view_alignment")
    ], dim=-1)
    assert torch.allclose(components.prod(-1), readiness)
    assert (components < .5).tolist() == [
        [False, False, False, False],
        [True, False, False, False],
        [False, True, False, False],
        [False, False, True, False],
        [False, False, False, True],
    ]
    assert crouch[0].item() == 0.0
    assert crouch[1].item() > 0.0
    assert crouch[2].item() > 0.0
    view_cost = microduck_mdp.mario_camera_view_cost(env, camera_cfg)
    assert view_cost[:4].tolist() == pytest.approx([0.] * 4)
    assert view_cost[4] > 0
