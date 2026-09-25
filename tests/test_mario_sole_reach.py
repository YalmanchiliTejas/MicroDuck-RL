"""Physical feasibility with the robot's full collision meshes, not point loads.

These fixtures constrain sole XY/orientation and load Z with 0.30 kg per foot.
They test obstruction and switch isolation, not whole-body policy balance.
"""
from pathlib import Path

import mujoco
import numpy as np
import pytest


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
