import math

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_grape_pick_env_cfg import (
    DESCENT_END,
    DESCENT_SPEED_WEIGHT,
    GENTLE_MOTION_WEIGHT,
    GRAPE_HALF_HEIGHT,
    GRAPE_LIFT_HEIGHT,
    GRAPE_POSITION_NOISE,
    GP_PERIOD,
    HOLD_END,
    JAW_CLOSE_END,
    MAX_DESCENT_SPEED,
    RISE_END,
    MicroduckGrapePickRlCfg,
    make_microduck_grape_pick_env_cfg,
)
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_GRAPE_PICK_ROBOT_CFG,
    get_grape_pick_robot_spec,
    get_grape_spec,
)


def test_grape_pick_cfg_wires_physical_object_objectives():
    cfg = make_microduck_grape_pick_env_cfg()

    assert list(cfg.scene.entities) == ["robot", "grape"]
    assert "reset_grape" in cfg.events
    reset = cfg.events["reset_grape"]
    assert reset.func is microduck_mdp.reset_grape_in_front_of_robot
    assert reset.params["grape_half_height"] == GRAPE_HALF_HEIGHT
    assert reset.params["noise_xy"] == GRAPE_POSITION_NOISE

    approach = cfg.rewards["mouth_grape_proximity"]
    lift = cfg.rewards["grape_lift_tracking"]
    assert approach.weight > 0.0
    assert approach.func is microduck_mdp.grip_pocket_grape_alignment_phased
    assert approach.params["forward_std"] == 0.12
    assert approach.params["asset_cfg"].site_names == [
        "upper_mouth_grip_center", "lower_mouth_grip_center"
    ]
    assert cfg.rewards["mouth_perpendicular_to_ground"].weight == 0.0
    precision = cfg.rewards["grip_pocket_precision"]
    assert precision.func is microduck_mdp.grip_pocket_grape_distance_phased
    assert precision.weight == 10.0
    assert precision.params["std"] == 0.018
    assert lift.weight > 0.0
    assert lift.func is microduck_mdp.grape_lift_tracking_phased
    assert lift.params["target_height"] == GRAPE_LIFT_HEIGHT
    assert lift.params["grasp_std"] == 0.018
    assert lift.params["grasp_distance"] == 0.0
    assert lift.params["asset_cfg"].site_names == [
        "upper_mouth_grip_center", "lower_mouth_grip_center"
    ]
    assert lift.params["upper_sensor_name"] == "upper_grape_contact"
    assert lift.params["lower_sensor_name"] == "lower_grape_contact"
    assert "grape_dual_contact" in cfg.rewards
    assert cfg.rewards["grape_dual_contact"].weight > lift.weight
    assert cfg.rewards["grape_dual_contact"].func is microduck_mdp.grape_dual_contact_phased
    assert cfg.rewards["grape_dual_contact"].params["close_end"] == JAW_CLOSE_END
    assert cfg.rewards["grape_dual_contact"].params["hold_end"] == HOLD_END
    assert cfg.rewards["grape_pad_contacts"].params["hold_end"] == HOLD_END
    assert lift.params["hold_end"] == HOLD_END
    assert (
        cfg.rewards["grape_pad_contacts"].weight
        < cfg.rewards["grape_dual_contact"].weight
    )
    assert (
        cfg.rewards["grape_pad_contacts"].func
        is microduck_mdp.grape_pad_contact_shaping_phased
    )
    assert {sensor.name for sensor in cfg.scene.sensors} >= {
        "upper_grape_contact",
        "lower_grape_contact",
        "left_kneel_ground_contact",
        "right_kneel_ground_contact",
    }
    kneel = cfg.rewards["kneel_support_with_reach"]
    assert kneel.func is microduck_mdp.kneel_support_with_reach_phased
    assert kneel.weight > 0.0
    assert (
        cfg.rewards["feet_grounded"].func
        is microduck_mdp.feet_grounded_return_phased
    )
    assert (
        cfg.rewards["feet_flat"].func
        is microduck_mdp.feet_flat_return_phased
    )
    assert set(cfg.metrics) >= {
        "kneel_support_with_reach",
        "pad_contacts",
        "grip_pocket_precision",
        "grape_height",
        "target_grape_height",
        "mouth_grape_distance",
        "grip_center_distance",
        "height_score",
        "grasp_score",
        "phase",
        "phase_gate",
    }
    for name in (
        "grape_height",
        "target_grape_height",
        "mouth_grape_distance",
        "height_score",
        "grasp_score",
        "phase",
        "phase_gate",
    ):
        metric = cfg.metrics[name]
        assert metric.func is microduck_mdp.grape_lift_diagnostic
        assert metric.params["metric"] == name
        assert metric.params["height_std"] == lift.params["height_std"]
        assert metric.params["grasp_std"] == lift.params["grasp_std"]
    assert "mouth_ground_proximity" not in cfg.rewards
    assert "mouth_payload_force" not in cfg.rewards
    assert "sample_mouth_payload" not in cfg.events

    learned = cfg.actions["joint_pos"]
    scripted = cfg.actions["scripted_mouth"]
    assert learned.actuator_names == (r"^(?!passive_).*",)
    assert isinstance(scripted, microduck_mdp.GroundPickMouthActionCfg)
    assert scripted.close_start == DESCENT_END
    assert scripted.close_end == JAW_CLOSE_END
    assert JAW_CLOSE_END < HOLD_END


def test_grape_pick_penalizes_fast_and_hard_descent():
    cfg = make_microduck_grape_pick_env_cfg()

    descent_speed = cfg.rewards["descent_speed"]
    assert descent_speed.func is microduck_mdp.trunk_downward_velocity_penalty
    assert descent_speed.weight == DESCENT_SPEED_WEIGHT > 0.0
    assert descent_speed.params["max_down_vel"] == MAX_DESCENT_SPEED
    assert descent_speed.params["asset_cfg"].body_names == ("trunk_base",)

    gentle_motion = cfg.rewards["gentle_motion"]
    assert gentle_motion.func is microduck_mdp.trunk_vertical_accel_penalty
    assert gentle_motion.weight == GENTLE_MOTION_WEIGHT > 0.0
    assert gentle_motion.params["asset_cfg"].body_names == ("trunk_base",)

    neck_speed = cfg.rewards["neck_vel_descent"]
    assert neck_speed.func is microduck_mdp.neck_vel_descent_penalty
    assert neck_speed.weight == -0.3

    head_impact = cfg.rewards["head_impact_penalty"]
    assert head_impact.func is microduck_mdp.body_impact_cost
    assert head_impact.weight == -10.0
    assert head_impact.params == {
        "sensor_name": "head_impact_contact",
        "threshold": 1.0,
    }

    # The velocity base deliberately removes this stock term; grape-pick uses
    # explicit trunk-speed, trunk-acceleration, and protected-head penalties.
    assert "soft_landing" not in cfg.rewards

def test_grape_pick_uses_six_second_capture_and_lift_cycle():
    assert GP_PERIOD == 6.0
    assert math.isclose(DESCENT_END * GP_PERIOD, 2.0)
    assert math.isclose(JAW_CLOSE_END * GP_PERIOD, 2.2)
    assert math.isclose(HOLD_END * GP_PERIOD, 3.0)
    assert math.isclose(RISE_END * GP_PERIOD, 5.0)


def test_enlarged_grape_asset_matches_ground_height_constant():
    import mujoco

    model = get_grape_spec().compile()
    grape_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "grape_geom")

    assert torch.allclose(
        torch.as_tensor(model.geom_size[grape_geom]),
        torch.tensor([0.0144, 0.012, GRAPE_HALF_HEIGHT], dtype=torch.float64),
    )


def test_grape_pick_robot_has_separate_moving_mouth():
    model = get_grape_pick_robot_spec().compile()
    import mujoco

    mouth_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "passive_mouth"
    )
    mouth_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mouth_jaw")
    tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mouth_tip")
    lower_tip = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "lower_mouth_tip"
    )
    upper_grip = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "upper_mouth_grip_center"
    )
    lower_grip = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "lower_mouth_grip_center"
    )
    assert mouth_joint >= 0
    assert mouth_body >= 0
    assert model.site_bodyid[tip] != mouth_body
    assert model.site_bodyid[lower_tip] == mouth_body
    assert model.site_bodyid[upper_grip] != mouth_body
    assert model.site_bodyid[lower_grip] == mouth_body
    # The reward sites are about 18 mm inward from the old distal-lip sites.
    assert 0.015 < math.dist(model.site_pos[tip], model.site_pos[upper_grip]) < 0.022
    assert (
        0.015
        < math.dist(model.site_pos[lower_tip], model.site_pos[lower_grip])
        < 0.022
    )
    assert math.isclose(model.jnt_range[mouth_joint, 0], math.radians(-5.0))
    assert math.isclose(model.jnt_range[mouth_joint, 1], math.radians(30.0))


def test_grape_grip_contacts_approximate_compliant_silicone():
    import mujoco

    spec = get_grape_pick_robot_spec()
    for collision_cfg in MICRODUCK_GRAPE_PICK_ROBOT_CFG.collisions:
        collision_cfg.edit_spec(spec)
    model = spec.compile()
    for name in ("upper_mouth_grip", "lower_mouth_grip"):
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert geom >= 0
        assert model.geom_contype[geom] != 0
        assert model.geom_conaffinity[geom] != 0
        assert model.geom_condim[geom] == 6
        assert math.isclose(model.geom_margin[geom], 0.0)
        assert torch.allclose(
            torch.as_tensor(model.geom_friction[geom]),
            torch.tensor([2.0, 0.02, 0.005], dtype=torch.float64),
        )
        assert torch.allclose(
            torch.as_tensor(model.geom_solref[geom]),
            torch.tensor([0.02, 1.0], dtype=torch.float64),
        )
        assert torch.allclose(
            torch.as_tensor(model.geom_solimp[geom, :3]),
            torch.tensor([0.8, 0.95, 0.003], dtype=torch.float64),
        )


def test_scripted_mouth_opens_descends_closes_and_stays_closed():
    close_midpoint = 0.5 * (DESCENT_END + JAW_CLOSE_END)
    phases = torch.tensor(
        [0.0, DESCENT_END, close_midpoint, JAW_CLOSE_END, HOLD_END, 0.99]
    )
    opening = microduck_mdp.ground_pick_mouth_opening(
        phases, DESCENT_END, JAW_CLOSE_END
    )
    assert torch.allclose(
        opening, torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0, 0.0]), atol=1e-6
    )


def test_grape_state_is_critic_only_and_actor_contract_stays_61d():
    cfg = make_microduck_grape_pick_env_cfg()
    actor = cfg.observations["actor"].terms
    critic = cfg.observations["critic"].terms

    assert "grape_position" not in actor
    assert "grape_velocity" not in actor
    assert critic["grape_position"].func is microduck_mdp.grape_pos_in_base
    assert critic["grape_velocity"].func is microduck_mdp.grape_vel_in_base
    assert actor["head_command"].params["dim"] == 4
    assert actor["body_command"].params["dim"] == 6


def test_grape_pick_has_distinct_runner_and_flat_terrain():
    cfg = make_microduck_grape_pick_env_cfg(play=True)
    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.scene.terrain.terrain_generator is None
    assert MicroduckGrapePickRlCfg.experiment_name == "grape_pick"
    assert MicroduckGrapePickRlCfg.actor.obs_normalization is True


def test_retention_metrics_distinguish_floor_pinching_from_lift():
    phase = torch.tensor([0.4, 0.7, 0.7, 0.7])
    grape = torch.tensor([[0., 0., 0.012], [0., 0., 0.012],
                          [0., 0., 0.08], [0., 0., 0.08]])
    env = _Env(grape, grape.clone(), phase)
    env.scene.sensors = {
        "upper": _Sensor(torch.ones(4, 1, dtype=torch.bool)),
        "lower": _Sensor(torch.tensor([[True], [True], [True], [False]])),
    }
    params = dict(upper_sensor_name="upper", lower_sensor_name="lower")
    metric = microduck_mdp.grape_retention_diagnostic
    assert metric(env, "dual_contact_raw", **params).tolist() == [1, 1, 1, 0]
    assert metric(env, "rise_active", **params).tolist() == [0, 1, 1, 1]
    assert metric(env, "rise_dual_contact", **params).tolist() == [0, 1, 1, 0]
    assert metric(env, "grape_lifted_and_held", **params).tolist() == [0, 0, 1, 0]
    assert torch.allclose(metric(env, "grape_clearance", **params),
                          torch.tensor([0., 0., .068, .068]))


def test_configured_contact_solver_retains_grape_during_lift():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts" / "check_grape_contact_physics.py"
    spec = importlib.util.spec_from_file_location("grape_contact_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    retention_trial = module.retention_trial

    # Same mesh, mass, jaw command and imposed ascent: the walking contact
    # solver registers a pinch but loses it. The grasp solver retains it.
    old = retention_trial(90, baseline=True)
    new = retention_trial(90)
    assert old["capture_dual_contact_fraction"] > .5
    assert not old["retained"]
    assert new["retained"]
    assert new["min_hold_height_m"] > old["min_hold_height_m"] + .06


class _Data:
    pass


class _Asset:
    def __init__(self, data):
        self.data = data


class _Terrain:
    def __init__(self, n):
        self.env_origins = torch.zeros(n, 3)


class _Scene(dict):
    def __init__(self, robot, grape, n):
        super().__init__(robot=robot, grape=grape)
        self.terrain = _Terrain(n)


class _SensorData:
    def __init__(self, found):
        self.found = found


class _Sensor:
    def __init__(self, found):
        self.data = _SensorData(found)


class _Commands:
    def __init__(self, phases):
        angle = 2.0 * math.pi * phases
        self.command = torch.stack(
            (torch.cos(angle), torch.sin(angle), torch.zeros_like(angle)), dim=1
        )

    def get_command(self, _name):
        return self.command


class _Env:
    def __init__(self, grape_pos, mouth_pos, phases):
        n = len(phases)
        robot_data = _Data()
        robot_data.site_pos_w = (
            mouth_pos[:, None, :] if mouth_pos.dim() == 2 else mouth_pos
        )
        num_sites = robot_data.site_pos_w.shape[1]
        robot_data.site_quat_w = torch.zeros(n, num_sites, 4)
        robot_data.site_quat_w[:, :, 0] = 1.0
        grape_data = _Data()
        grape_data.root_link_pos_w = grape_pos
        grape_data.root_link_lin_vel_w = torch.zeros(n, 3)
        self.scene = _Scene(_Asset(robot_data), _Asset(grape_data), n)
        self.command_manager = _Commands(phases)
        self.num_envs = n
        self.device = "cpu"


def _mouth_cfg():
    cfg = SceneEntityCfg("robot", site_names=["mouth_tip"])
    cfg.site_ids = [0]
    return cfg


def _grip_cfg():
    cfg = SceneEntityCfg(
        "robot", site_names=["upper_mouth_grip_center", "lower_mouth_grip_center"]
    )
    cfg.site_ids = [0, 1]
    return cfg


def test_lift_reward_tracks_slewed_target_and_rejects_throwing():
    # Mid-rise target is 6.5 cm. Samples: held/on-target, held-but-early at
    # final height, and target-height grape far away from the mouth (a throw).
    phase = torch.tensor([0.6125, 0.6125, 0.6125])
    grape = torch.tensor([[0.0, 0.0, 0.065], [0.0, 0.0, 0.12], [0.0, 0.0, 0.065]])
    mouth = torch.tensor([[0.01, 0.0, 0.065], [0.01, 0.0, 0.12], [0.20, 0.0, 0.065]])
    env = _Env(grape, mouth, phase)
    env.scene.sensors = {
        "upper": _Sensor(torch.ones(3, 1, dtype=torch.bool)),
        "lower": _Sensor(torch.ones(3, 1, dtype=torch.bool)),
    }
    out = microduck_mdp.grape_lift_tracking_phased(
        env,
        upper_sensor_name="upper",
        lower_sensor_name="lower",
        asset_cfg=_mouth_cfg(),
    )

    assert out[0] > out[1]
    assert out[0] > 0.49
    assert out[2] < 1e-6


def test_lift_reward_requires_simultaneous_physical_pad_contact():
    phase = torch.tensor([0.6125, 0.6125])
    grape = torch.tensor([[0.0, 0.0, 0.065], [0.0, 0.0, 0.065]])
    mouth = torch.tensor([[0.01, 0.0, 0.065], [0.01, 0.0, 0.065]])
    env = _Env(grape, mouth, phase)
    env.scene.sensors = {
        "upper": _Sensor(torch.tensor([[True], [True]])),
        "lower": _Sensor(torch.tensor([[True], [False]])),
    }

    reward = microduck_mdp.grape_lift_tracking_phased(
        env,
        upper_sensor_name="upper",
        lower_sensor_name="lower",
        asset_cfg=_mouth_cfg(),
    )

    assert reward[0] > 0.49
    assert reward[1] == 0.0


def test_lift_diagnostics_expose_reward_components():
    phase = torch.tensor([0.6125])
    grape = torch.tensor([[0.0, 0.0, 0.065]])
    mouth = torch.tensor([[0.01, 0.0, 0.065]])
    env = _Env(grape, mouth, phase)
    params = {
        "asset_cfg": _mouth_cfg(),
        "height_std": 0.08,
        "grasp_std": 0.05,
    }

    values = {
        name: microduck_mdp.grape_lift_diagnostic(env, metric=name, **params)
        for name in (
            "grape_height",
            "target_grape_height",
            "mouth_grape_distance",
            "height_score",
            "grasp_score",
            "phase",
            "phase_gate",
        )
    }

    assert torch.allclose(values["grape_height"], torch.tensor([0.065]))
    assert torch.allclose(values["target_grape_height"], torch.tensor([0.065]))
    assert torch.allclose(values["mouth_grape_distance"], torch.tensor([0.01]))
    assert torch.allclose(values["height_score"], torch.ones(1))
    assert torch.allclose(values["grasp_score"], torch.ones(1))
    assert torch.allclose(values["phase"], phase)
    assert torch.allclose(values["phase_gate"], torch.tensor([0.5]))

    env.scene.sensors = {
        "upper": _Sensor(torch.ones(1, 1, dtype=torch.bool)),
        "lower": _Sensor(torch.ones(1, 1, dtype=torch.bool)),
    }
    reward = microduck_mdp.grape_lift_tracking_phased(
        env, upper_sensor_name="upper", lower_sensor_name="lower", **params
    )
    reconstructed = (
        values["phase_gate"] * values["height_score"] * values["grasp_score"]
    )
    assert torch.allclose(reward, reconstructed)


def test_approach_reward_targets_center_between_both_jaw_sites():
    phase = torch.tensor([0.40, 0.40])
    grape = torch.tensor([[0.0, 0.0, 0.01], [0.0, 0.0, 0.01]])
    upper = torch.tensor([[0.0, 0.0, 0.03], [0.0, 0.0, 0.03]])
    lower = torch.tensor([[0.0, 0.0, -0.01], [0.0, 0.20, -0.01]])
    mouth = torch.stack((upper, lower), dim=1)
    out = microduck_mdp.grip_pocket_grape_alignment_phased(
        _Env(grape, mouth, phase), asset_cfg=_grip_cfg()
    )

    assert out[0] > 0.99
    assert out[1] < 0.01


def test_precision_reward_strongly_prefers_grape_inside_jaw_pocket():
    phase = torch.tensor([0.4, 0.4])
    grape = torch.tensor([[0.0, 0.0, 0.01], [0.0, 0.0, 0.01]])
    centered_upper = torch.tensor([[0.0, 0.0, 0.03], [0.0, 0.0, 0.03]])
    lower = torch.tensor([[0.0, 0.0, -0.01], [0.10, 0.0, -0.01]])
    mouth = torch.stack((centered_upper, lower), dim=1)

    score = microduck_mdp.grip_pocket_grape_distance_phased(
        _Env(grape, mouth, phase), asset_cfg=_grip_cfg(), std=0.018
    )

    assert score[0] > 0.99
    assert score[1] < 0.14


def test_dual_contact_requires_both_pads_after_mouth_closes():
    phase = torch.tensor([0.4, 0.4, 0.35, 0.6])
    env = _Env(
        torch.zeros(4, 3), torch.zeros(4, 3), phase
    )
    env.scene.sensors = {
        "upper": _Sensor(torch.tensor([[True], [True], [True], [True]])),
        "lower": _Sensor(torch.tensor([[True], [False], [True], [True]])),
    }

    score = microduck_mdp.grape_dual_contact_phased(
        env,
        upper_sensor_name="upper",
        lower_sensor_name="lower",
        close_end=JAW_CLOSE_END,
        hold_end=HOLD_END,
    )

    assert score[0] > 0.0
    assert score[1] == 0.0
    assert score[2] == 0.0
    assert score[3] == 0.0


def test_single_pad_contact_provides_partial_capture_shaping():
    phase = torch.tensor([0.4, 0.4, 0.2, 0.7])
    env = _Env(torch.zeros(4, 3), torch.zeros(4, 3), phase)
    env.scene.sensors = {
        "upper": _Sensor(torch.tensor([[True], [True], [True], [True]])),
        "lower": _Sensor(torch.tensor([[True], [False], [True], [True]])),
    }

    score = microduck_mdp.grape_pad_contact_shaping_phased(
        env,
        upper_sensor_name="upper",
        lower_sensor_name="lower",
        hold_end=HOLD_END,
    )

    assert score[0] == 1.0
    assert score[1] == 0.5
    assert score[2] == 0.0
    assert score[3] == 0.0


def test_kneel_support_requires_both_legs_and_only_pays_in_down_phase():
    phase = torch.tensor([0.4, 0.4, 0.9])
    grape = torch.tensor([[0.0, 0.0, 0.01]]).expand(3, -1)
    upper = torch.tensor([[0.0, 0.0, 0.03]]).expand(3, -1)
    lower = torch.tensor([[0.0, 0.0, -0.01]]).expand(3, -1)
    mouth = torch.stack((upper, lower), dim=1)
    env = _Env(grape, mouth, phase)
    env.scene.sensors = {
        "left_kneel": _Sensor(torch.tensor([[True], [True], [True]])),
        "right_kneel": _Sensor(torch.tensor([[True], [False], [True]])),
    }

    score = microduck_mdp.kneel_support_with_reach_phased(
        env,
        left_sensor_name="left_kneel",
        right_sensor_name="right_kneel",
        asset_cfg=_grip_cfg(),
    )

    assert score[0] > 0.99
    assert score[1] == 0.0
    assert score[2] == 0.0


def test_feet_ground_reward_is_disabled_until_return_phase():
    phase = torch.tensor([0.4, 0.9])
    env = _Env(torch.zeros(2, 3), torch.zeros(2, 3), phase)
    env.scene.sensors = {
        "feet": _Sensor(torch.tensor([[1.0, 1.0], [1.0, 1.0]])),
    }

    score = microduck_mdp.feet_grounded_return_phased(env, sensor_name="feet")

    assert score[0] == 0.0
    assert score[1] == 1.0
