from pathlib import Path

import mujoco
import pytest

from mjlab_microduck.controller import NESController


ROBOT_DIR = Path(__file__).parents[1] / "src/mjlab_microduck/robot/microduck"


def test_physical_controller_compiles_with_four_independent_slider_keys():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_DIR / "controller_nes.xml"))
    expected = set(NESController.BUTTON_JOINTS)
    actual = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    }
    assert actual == expected
    for name in expected:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert tuple(model.jnt_range[joint_id]) == pytest.approx((-0.002, 0.0))
        assert model.jnt_stiffness[joint_id] == pytest.approx(1200.0)
        dof_id = model.jnt_dofadr[joint_id]
        assert model.dof_damping[dof_id] == pytest.approx(2.0)

    for name in (
        "dpad_left_surface",
        "dpad_right_surface",
        "button_a_surface",
        "button_b_surface",
    ):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert model.geom_priority[geom_id] == 2
        assert model.geom_size[geom_id, 2] == pytest.approx(0.003)


def test_full_scene_places_each_foot_over_its_controller():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_DIR / "scene_controller_nes.xml"))
    for body_name in ("dpad_platform", "ab_platform", "mario_monitor"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name) >= 0
    assert model.nq > 4


def test_home_stance_does_not_preload_any_button():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_DIR / "scene_controller_nes.xml"))
    data = mujoco.MjData(model)
    root_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
    )
    root_adr = model.jnt_qposadr[root_id]
    data.qpos[root_adr : root_adr + 7] = (0, 0, 0.135, 1, 0, 0, 0)
    home = {
        "left_hip_yaw": 0.0,
        "left_hip_roll": -0.0872664626,
        "left_hip_pitch": -0.457924,
        "left_knee": -0.004940,
        "left_ankle": 0.452984,
        "neck_pitch": 0.3490658504,
        "head_pitch": 0.3490658504,
        "head_yaw": 0.0,
        "head_roll": 0.0,
        "right_hip_yaw": 0.0,
        "right_hip_roll": 0.0872664626,
        "right_hip_pitch": 0.457924,
        "right_knee": 0.004940,
        "right_ankle": -0.452984,
    }
    for name, value in home.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
        )
        data.qpos[model.jnt_qposadr[joint_id]] = value
        data.ctrl[actuator_id] = value
    mujoco.mj_forward(model, data)
    for _ in range(50):
        mujoco.mj_step(model, data)

    for address in NESController.joint_qpos_addresses(model):
        assert -float(data.qpos[address]) < 0.0003


@pytest.mark.parametrize(
    "joint_name,body_name",
    [
        ("passive_dpad_left", "dpad_left_key"),
        ("passive_dpad_right", "dpad_right_key"),
        ("passive_button_a", "button_a_key"),
        ("passive_button_b", "button_b_key"),
    ],
)
def test_each_key_can_be_pressed_without_mechanically_moving_others(
    joint_name, body_name
):
    model = mujoco.MjModel.from_xml_path(str(ROBOT_DIR / "controller_nes.xml"))
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    data.xfrc_applied[body_id, 2] = -1.0
    for _ in range(1_000):
        mujoco.mj_step(model, data)

    travels = {}
    for name in NESController.BUTTON_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        travels[name] = -float(data.qpos[model.jnt_qposadr[joint_id]])
    assert travels[joint_name] > 0.0007
    assert all(
        travel < 0.00015
        for name, travel in travels.items()
        if name != joint_name
    )


@pytest.mark.parametrize(
    "travels,expected",
    [
        ((0.001, 0, 0, 0), {"LEFT"}),
        ((0, 0.001, 0, 0), {"RIGHT"}),
        ((0, 0, 0.001, 0), {"A"}),
        ((0, 0, 0, 0.001), {"B"}),
        ((0.001, 0, 0.001, 0), {"LEFT", "A"}),
    ],
)
def test_decoder_maps_each_independent_travel(travels, expected):
    state = NESController().update(*travels)
    assert {button.value for button in state.active} == expected


def test_each_key_has_activation_and_release_hysteresis():
    controller = NESController()
    assert controller.update(0, 0.0008, 0.0008, 0).right
    held = controller.update(0, 0.0005, 0.0005, 0)
    assert held.right and held.a
    released = controller.update(0, 0.0002, 0.0002, 0)
    assert not released.right and not released.a


def test_decoder_reads_negative_slider_qpos_as_positive_travel():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_DIR / "controller_nes.xml"))
    data = mujoco.MjData(model)
    addresses = NESController.joint_qpos_addresses(model)
    data.qpos[addresses[2]] = -0.001
    state = NESController().update_from_qpos(data.qpos, addresses)
    assert state.a
    assert not state.b
