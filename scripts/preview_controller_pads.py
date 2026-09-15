#!/usr/bin/env python3
"""Render the Microduck controller-pad scene to a PNG.

This preview uses plain MuJoCo so it also works on machines without a GPU
capable of running the full batched Mjlab environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = (
    ROOT / "src/mjlab_microduck/robot/microduck/scene_controller_pads.xml"
)

HOME_JOINTS = {
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


def set_home_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    # The robot free joint is the only free joint in this scene.
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            qpos_adr = model.jnt_qposadr[joint_id]
            data.qpos[qpos_adr : qpos_adr + 7] = (0.0, 0.0, 0.12, 1.0, 0.0, 0.0, 0.0)
            break

    for name, value in HOME_JOINTS.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"joint missing from controller scene: {name}")
        data.qpos[model.jnt_qposadr[joint_id]] = value

    mujoco.mj_forward(model, data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "controller_pads_preview.png")
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=800)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    data = mujoco.MjData(model)
    set_home_pose(model, data)

    # Run a short settle, then render a close three-quarter view that shows all
    # three pads and the robot together.
    for _ in range(100):
        mujoco.mj_step(model, data)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.07, 0.0, 0.09)
    camera.distance = 0.55
    camera.azimuth = 145.0
    camera.elevation = -32.0

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    renderer.update_scene(data, camera=camera)
    image = renderer.render()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    Image.fromarray(image).save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
