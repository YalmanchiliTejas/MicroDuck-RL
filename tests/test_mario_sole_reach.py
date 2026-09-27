"""Physical feasibility with the robot's full collision meshes, not point loads.

These fixtures constrain sole XY/orientation and load Z with 0.30 kg per foot.
They test obstruction and switch isolation, not whole-body policy balance.
"""
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


ROBOT_DIR = Path(__file__).parents[1] / "src/mjlab_microduck/robot/microduck"


def sole_vertices(side):
    model = mujoco.MjModel.from_xml_path(str(ROBOT_DIR / "robot_groundcontact.xml"))
    geom = model.geom(f"{side}_foot_collision").id
    site = model.site(f"{side}_foot").id
    mesh = model.geom_dataid[geom]
    start = model.mesh_vertadr[mesh]
    vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
    inverse = np.empty(4)
    mujoco.mju_negQuat(inverse, model.site_quat[site])
    transformed = []
    for vertex in vertices:
        body, local = np.empty(3), np.empty(3)
        mujoco.mju_rotVecQuat(body, vertex, model.geom_quat[geom])
        mujoco.mju_rotVecQuat(
            local, body + model.geom_pos[geom] - model.site_pos[site], inverse
        )
        transformed.append(local)
    return " ".join(str(float(v)) for v in np.asarray(transformed).ravel())


def loaded_keys(left=0, right=0):
    # Key locations come from XML so a stale target will fail config tests.
    controller = mujoco.MjModel.from_xml_path(str(ROBOT_DIR / "controller_nes.xml"))
    positions = [controller.body(name).pos.copy()
                 for name in ("dpad_platform", "ab_platform")]
    for index, request in enumerate((left, right)):
        if request:
            name = ("dpad_left_key" if request < 0 else "dpad_right_key") if index == 0 else (
                "button_a_key" if request > 0 else "button_b_key")
            positions[index] += controller.body(name).pos
        positions[index][2] = 0.035
    meshes, bodies = [], []
    for index, side in enumerate(("left", "right")):
        meshes.append(f'<mesh name="{side}" vertex="{sole_vertices(side)}"/>')
        roll = np.deg2rad(-5 if side == "left" else 5)
        quat = f"{np.cos(roll / 2)} {np.sin(roll / 2)} 0 0"
        pos = " ".join(map(str, positions[index]))
        bodies.append(f'''<body name="{side}_probe" pos="{pos}">
          <joint name="passive_{side}_probe" type="slide" axis="0 0 1" damping="2"/>
          <inertial pos="0 0 0" mass="0.30" diaginertia="0.001 0.001 0.001"/>
          <geom type="mesh" mesh="{side}" quat="{quat}" friction="1.6 0.02 0.002"/>
        </body>''')
    xml = f'''<mujoco><include file="{ROBOT_DIR / 'controller_nes.xml'}"/>
      <option timestep="0.001"/>
      <asset>{''.join(meshes)}</asset><worldbody>{''.join(bodies)}</worldbody>
    </mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(2000):
        mujoco.mj_step(model, data)
    names = ("passive_dpad_left", "passive_dpad_right", "passive_button_a", "passive_button_b")
    return np.array([-data.qpos[model.jnt_qposadr[model.joint(name).id]] for name in names])


def loaded_rocker_pose(left=0, right=0):
    """Settle the measured planted rocker targets under realistic sole load."""

    controller = mujoco.MjModel.from_xml_path(str(ROBOT_DIR / "controller_nes.xml"))
    platform_names = ("dpad_platform", "ab_platform")
    positions = [controller.body(name).pos.copy() for name in platform_names]
    left_x = {-1: -0.018, 0: -0.0075, 1: 0.003}[left]
    right_x = {0: 0.0, 1: 0.012}[right]
    pitches = (left * 6.0, right * 6.0)
    positions[0][0] += left_x
    positions[1][0] += right_x
    for position in positions:
        position[2] = 0.035

    meshes, bodies = [], []
    for index, side in enumerate(("left", "right")):
        meshes.append(f'<mesh name="{side}" vertex="{sole_vertices(side)}"/>')
        roll = -5.0 if side == "left" else 5.0
        quat_xyzw = Rotation.from_euler(
            "xy", (roll, pitches[index]), degrees=True
        ).as_quat()
        quat = " ".join(map(str, np.roll(quat_xyzw, 1)))
        pos = " ".join(map(str, positions[index]))
        bodies.append(f'''<body name="{side}_probe" pos="{pos}" quat="{quat}">
          <joint name="passive_{side}_probe" type="slide" axis="0 0 1" damping="2"/>
          <inertial pos="0 0 0" mass="0.30" diaginertia="0.001 0.001 0.001"/>
          <geom type="mesh" mesh="{side}" friction="1.6 0.02 0.002"/>
        </body>''')
    xml = f'''<mujoco><include file="{ROBOT_DIR / 'controller_nes.xml'}"/>
      <option timestep="0.001"/>
      <asset>{''.join(meshes)}</asset><worldbody>{''.join(bodies)}</worldbody>
    </mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(2_000):
        mujoco.mj_step(model, data)
    names = (
        "passive_dpad_left", "passive_dpad_right",
        "passive_button_a", "passive_button_b",
    )
    return np.array([
        -data.qpos[model.jnt_qposadr[model.joint(name).id]] for name in names
    ])


@pytest.mark.parametrize("left,right,expected", [
    (0, 0, []), (-1, 0, [0]), (1, 0, [1]), (0, 1, [2]),
    (-1, 1, [0, 2]), (1, 1, [1, 2]),
])
def test_measured_rocker_poses_press_only_requested_keys(left, right, expected):
    travel = loaded_rocker_pose(left, right)
    for index in range(4):
        if index in expected:
            assert travel[index] >= 0.0007, travel * 1000
        else:
            assert travel[index] < 0.0003, travel * 1000


@pytest.mark.parametrize("left,right,expected", [
    (0, 0, []), (-1, 0, [0]), (1, 0, [1]), (0, 1, [2]), (0, -1, [3]),
    (-1, 1, [0, 2]), (1, 1, [1, 2]), (-1, -1, [0, 3]), (1, -1, [1, 3]),
])
def test_full_soles_can_press_exact_keys(left, right, expected):
    travel = loaded_keys(left, right)
    for index in range(4):
        if index in expected:
            assert travel[index] >= 0.0007, travel * 1000
        else:
            assert travel[index] < 0.0003, travel * 1000


@pytest.mark.parametrize(
    "key_name",
    ["dpad_left_key", "dpad_right_key"],
)
def test_articulated_left_leg_reaches_sagittal_dpad_with_level_sole(key_name):
    """Guard against proving reachability with a teleported sole only.

    The trunk stays fixed and upright, hip yaw/roll stay at HOME, and only the
    left leg's sagittal chain may move. Both D-pad targets must be attainable
    without changing the sole orientation.
    """

    model = mujoco.MjModel.from_xml_path(
        str(ROBOT_DIR / "scene_controller_nes.xml")
    )
    data = mujoco.MjData(model)
    root_adr = model.joint("trunk_base_freejoint").qposadr[0]
    data.qpos[root_adr : root_adr + 7] = (0, 0, 0.135, 1, 0, 0, 0)
    home = {
        "left_hip_yaw": 0.0,
        "left_hip_roll": np.deg2rad(-5.0),
        "left_hip_pitch": -0.457924,
        "left_knee": -0.004940,
        "left_ankle": 0.452984,
    }
    for name, value in home.items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)

    site_id = model.site("left_foot").id
    home_pos = data.site_xpos[site_id].copy()
    home_mat = data.site_xmat[site_id].reshape(3, 3).copy()
    platform_x = data.xpos[model.body("dpad_platform").id, 0]
    assert abs(home_pos[0] - platform_x) < 0.0005
    target = home_pos.copy()
    target[0] += model.body(key_name).pos[0]

    joint_names = ("left_hip_pitch", "left_knee", "left_ankle")
    qpos_adrs = [model.joint(name).qposadr[0] for name in joint_names]
    initial = np.array([data.qpos[adr] for adr in qpos_adrs])
    bounds = np.array([model.joint(name).range for name in joint_names]).T

    def residual(joint_pos):
        for adr, value in zip(qpos_adrs, joint_pos, strict=True):
            data.qpos[adr] = value
        mujoco.mj_forward(model, data)
        position_mm = (data.site_xpos[site_id] - target) * 1000.0
        site_mat = data.site_xmat[site_id].reshape(3, 3)
        orientation = Rotation.from_matrix(
            home_mat.T @ site_mat
        ).as_rotvec()
        return np.concatenate((position_mm, orientation * 100.0))

    result = least_squares(
        residual,
        initial,
        bounds=bounds,
        max_nfev=1_000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    residual(result.x)
    position_error_mm = np.linalg.norm(data.site_xpos[site_id] - target) * 1000
    orientation_error_deg = np.degrees(
        Rotation.from_matrix(
            home_mat.T @ data.site_xmat[site_id].reshape(3, 3)
        ).magnitude()
    )
    assert position_error_mm < 0.1
    assert orientation_error_deg < 0.1
