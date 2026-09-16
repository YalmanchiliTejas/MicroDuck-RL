#!/usr/bin/env python3
"""Capture the live Mario monitor through Microduck's head camera.

Start ``microduck-super-mario`` first. It publishes the current NES RGB frame
to shared memory; this script uploads that frame to the in-world monitor before
rendering the named head camera.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
from PIL import Image

from mjlab_microduck.mario_monitor import (
    DEFAULT_FRAME_SHM,
    MarioFrameSubscriber,
    MarioHeadCameraRenderer,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / "src/mjlab_microduck/robot/microduck/scene_controller_pads.xml"


def _set_robot_spawn(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            adr = model.jnt_qposadr[joint_id]
            data.qpos[adr : adr + 7] = (0.0, 0.0, 0.12, 1.0, 0.0, 0.0, 0.0)
            break
    mujoco.mj_forward(model, data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--frame-shm", default=DEFAULT_FRAME_SHM)
    parser.add_argument("--output", type=Path, default=ROOT / "mario_head_camera.png")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--wait-seconds", type=float, default=5.0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    _set_robot_spawn(model, data)

    deadline = time.monotonic() + args.wait_seconds
    subscriber = MarioFrameSubscriber(args.frame_shm)
    while True:
        try:
            subscriber.connect()
            break
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Mario frame stream was not found; start microduck-super-mario first"
                ) from None
            time.sleep(0.05)

    try:
        with MarioHeadCameraRenderer(
            model, data, width=args.width, height=args.height
        ) as renderer:
            captured = renderer.render_latest(subscriber)
            if captured is None:
                raise RuntimeError("Mario frame was being updated; run the capture again")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(captured.rgb).save(args.output)
            print(args.output.resolve())
    finally:
        subscriber.close()


if __name__ == "__main__":
    main()
