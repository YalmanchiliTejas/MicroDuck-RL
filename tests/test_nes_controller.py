from math import radians
from pathlib import Path

import mujoco
import numpy as np
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
        lower = 0.0 if name == "passive_ab_rocker" else -radians(2)
        assert tuple(model.jnt_range[joint_id]) == pytest.approx(
            (lower, radians(2))
        )
        stiffness = 1.5 if name == "passive_dpad_x" else 3.0
        assert model.jnt_stiffness[joint_id] == pytest.approx(stiffness)
        dof_id = model.jnt_dofadr[joint_id]
        damping = 0.15 if name == "passive_dpad_x" else 0.25
        assert model.dof_damping[dof_id] == pytest.approx(damping)

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


def test_single_foot_loads_can_click_each_mario_axis_without_false_neutral():
    """A 4 N sole load at the calibrated pressure point must cross 0.6°."""

    model = mujoco.MjModel.from_xml_path(str(ROBOT_DIR / "controller_nes.xml"))
    left_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dpad_platform")
    right_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "ab_rocker_platform"
    )

    def settled_angles(
        left_xy: tuple[float, float], right_xy: tuple[float, float]
    ) -> tuple[float, float, float]:
        data = mujoco.MjData(model)
        for _ in range(500):
            data.qfrc_applied[:] = 0.0
            mujoco.mj_forward(model, data)
            for body_id, (x, y) in (
                (left_body, left_xy), (right_body, right_xy)
            ):
                point = data.xpos[body_id].copy() + np.array([x, y, 0.0])
                mujoco.mj_applyFT(
                    model,
                    data,
                    np.array([0.0, 0.0, -4.0]),
                    np.zeros(3),
                    point,
                    body_id,
                    data.qfrc_applied,
                )
            mujoco.mj_step(model, data)
        result = []
        for name in ("passive_dpad_x", "passive_dpad_y", "passive_ab_rocker"):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            result.append(float(data.qpos[model.jnt_qposadr[joint_id]]))
        return tuple(result)

    # Sole *sites* sit a few millimetres ahead of their actual pressure
    # centroids. Apply these forces at pressure centroids, not at site targets.
    neutral = settled_angles((0.0, 0.0), (0.0, 0.0))
    assert all(abs(value) < radians(0.2) for value in neutral)
    left = settled_angles((0.0, 0.018), (0.0, 0.0))
    right = settled_angles((0.0, -0.018), (0.0, 0.0))
    jump = settled_angles((0.0, 0.0), (0.012, 0.0))
    assert left[0] < -radians(0.6)
    assert right[0] > radians(0.6)
    assert jump[2] > radians(0.6)
    assert abs(jump[0]) < radians(0.2)


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
