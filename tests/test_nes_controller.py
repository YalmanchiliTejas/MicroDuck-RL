from math import radians
from pathlib import Path

import mujoco
import pytest

from mjlab_microduck.controller import NESController


ROBOT_DIR = Path(__file__).parents[1] / "src/mjlab_microduck/robot/microduck"


def test_physical_controller_compiles_with_four_passive_limited_axes():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_DIR / "controller_nes.xml"))
    expected = {
        "passive_dpad_x",
        "passive_dpad_y",
        "passive_ab_rocker",
        "passive_ab_press",
    }
    actual = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    }
    assert actual == expected
    for name in expected - {"passive_ab_press"}:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert tuple(model.jnt_range[joint_id]) == pytest.approx(
            (-radians(2), radians(2))
        )
        assert model.jnt_stiffness[joint_id] == pytest.approx(2.5)
        dof_id = model.jnt_dofadr[joint_id]
        assert model.dof_damping[dof_id] == pytest.approx(0.08)

    for name in ("dpad_surface", "ab_surface"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert model.geom_priority[geom_id] == 2
        assert model.geom_size[geom_id, 2] == pytest.approx(0.006)


def test_full_scene_places_each_foot_over_its_controller():
    model = mujoco.MjModel.from_xml_path(
        str(ROBOT_DIR / "scene_controller_nes.xml")
    )
    for body_name in ("dpad_platform", "ab_rocker_platform", "mario_monitor"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name) >= 0
    assert model.nq > 3


@pytest.mark.parametrize(
    ("x", "y", "ab", "press", "expected"),
    [
        (-1, 0, 0, 0.0, {"LEFT"}),
        (1, 0, 0, 0.0, {"RIGHT"}),
        (0, 1, 0, 0.0, {"UP"}),
        (0, -1, 0, 0.0, {"DOWN"}),
        (0, 0, 1, 0.0, {"A"}),
        (0, 0, -1, 0.0, {"B"}),
        (1, 0, 1, 0.0, {"RIGHT", "A"}),
        (1, 0, -1, 0.0, {"RIGHT", "B"}),
        (1, 0, 0, 0.002, {"RIGHT", "A", "B"}),
        (-1, 0, 1, 0.0, {"LEFT", "A"}),
    ],
)
def test_requested_single_and_combined_inputs(x, y, ab, press, expected):
    controller = NESController()
    state = controller.update(
        radians(2) * x, radians(2) * y, radians(2) * ab, press
    )
    assert {button.value for button in state.active} == expected


def test_dpad_and_ab_hysteresis_prevent_threshold_flicker():
    controller = NESController()
    assert controller.update(radians(0.7), 0, radians(0.7)).right
    held = controller.update(radians(0.3), 0, radians(0.3))
    assert held.right and held.a
    released = controller.update(radians(0.1), 0, radians(0.1))
    assert not released.right and not released.a


def test_ab_chord_press_has_its_own_hysteresis():
    controller = NESController()
    chord = controller.update(0, 0, 0, 0.0012)
    assert chord.a and chord.b
    held = controller.update(0, 0, 0, 0.0008)
    assert held.a and held.b
    released = controller.update(0, 0, 0, 0.0005)
    assert not released.a and not released.b
