import math

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_grape_pick_env_cfg import (
    DESCENT_END,
    GRAPE_HALF_HEIGHT,
    GRAPE_LIFT_HEIGHT,
    GRAPE_POSITION_NOISE,
    HOLD_END,
    MicroduckGrapePickRlCfg,
    make_microduck_grape_pick_env_cfg,
)
from mjlab_microduck.robot.microduck_constants import get_grape_pick_robot_spec


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
    assert approach.func is microduck_mdp.mouth_grape_proximity_phased
    assert approach.params["std"] == 0.12
    assert lift.weight > 0.0
    assert lift.func is microduck_mdp.grape_lift_tracking_phased
    assert lift.params["target_height"] == GRAPE_LIFT_HEIGHT
    assert lift.params["grasp_std"] == 0.05
    assert set(cfg.metrics) >= {
        "grape_height",
        "target_grape_height",
        "mouth_grape_distance",
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
    assert scripted.close_end == HOLD_END


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
    assert mouth_joint >= 0
    assert mouth_body >= 0
    assert model.site_bodyid[tip] != mouth_body
    assert model.site_bodyid[lower_tip] == mouth_body
    assert math.isclose(model.jnt_range[mouth_joint, 0], math.radians(-5.0))
    assert math.isclose(model.jnt_range[mouth_joint, 1], math.radians(30.0))


def test_scripted_mouth_opens_descends_closes_and_stays_closed():
    close_midpoint = 0.5 * (DESCENT_END + HOLD_END)
    phases = torch.tensor(
        [0.0, DESCENT_END, close_midpoint, HOLD_END, 0.8, 0.99]
    )
    opening = microduck_mdp.ground_pick_mouth_opening(
        phases, DESCENT_END, HOLD_END
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
        robot_data.site_pos_w = mouth_pos[:, None, :]
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


def test_lift_reward_tracks_slewed_target_and_rejects_throwing():
    # Mid-rise target is 6.5 cm. Samples: held/on-target, held-but-early at
    # final height, and target-height grape far away from the mouth (a throw).
    phase = torch.tensor([0.6125, 0.6125, 0.6125])
    grape = torch.tensor([[0.0, 0.0, 0.065], [0.0, 0.0, 0.12], [0.0, 0.0, 0.065]])
    mouth = torch.tensor([[0.01, 0.0, 0.065], [0.01, 0.0, 0.12], [0.20, 0.0, 0.065]])
    out = microduck_mdp.grape_lift_tracking_phased(
        _Env(grape, mouth, phase), asset_cfg=_mouth_cfg()
    )

    assert out[0] > out[1]
    assert out[0] > 0.49
    assert out[2] < 1e-6


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

    reward = microduck_mdp.grape_lift_tracking_phased(env, **params)
    reconstructed = (
        values["phase_gate"] * values["height_score"] * values["grasp_score"]
    )
    assert torch.allclose(reward, reconstructed)


def test_approach_reward_uses_grape_surface_not_ground_height():
    phase = torch.tensor([0.40, 0.40])
    grape = torch.tensor([[0.0, 0.0, 0.01], [0.0, 0.0, 0.01]])
    mouth = torch.tensor([[0.01, 0.0, 0.01], [0.10, 0.0, 0.01]])
    out = microduck_mdp.mouth_grape_proximity_phased(
        _Env(grape, mouth, phase), asset_cfg=_mouth_cfg()
    )

    assert out[0] > 0.99
    assert out[1] < 0.01
