#!/usr/bin/env python3
"""Render and inspect the physical two-foot NES controller scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw, ImageFont

from mjlab_microduck.controller import NESController


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "src/mjlab_microduck/robot/microduck/scene_controller_nes.xml"

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
    root_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
    )
    root_adr = model.jnt_qposadr[root_id]
    # Neutral controller top is 15 mm above the floor.
    data.qpos[root_adr : root_adr + 7] = (0.0, 0.0, 0.135, 1.0, 0.0, 0.0, 0.0)
    for name, value in HOME_JOINTS.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if joint_id < 0 or actuator_id < 0:
            raise RuntimeError(f"joint or actuator missing: {name}")
        data.qpos[model.jnt_qposadr[joint_id]] = value
        data.ctrl[actuator_id] = value
    mujoco.mj_forward(model, data)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ("Arial Bold.ttf", "Arial.ttf") if bold else ("Arial.ttf",)
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def annotate(image: Image.Image, telemetry: str) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = font(34, bold=True)
    label_font = font(21)
    mono_font = font(17)

    draw.rounded_rectangle((27, 24, 705, 110), radius=18, fill=(7, 12, 22, 220))
    draw.text((49, 40), "MicroDuck Physical NES Controller", font=title_font, fill="white")
    draw.text(
        (49, 79),
        "Left foot: 4-way D-pad  |  Right foot: A/B rocker",
        font=label_font,
        fill=(205, 218, 238),
    )

    # The physical axes and activation thresholds are explicit in the preview.
    panel = (image.width - 405, 24, image.width - 27, 370)
    draw.rounded_rectangle(panel, radius=18, fill=(7, 12, 22, 220))
    draw.text((panel[0] + 22, panel[1] + 16), "LIVE JOINT STATE", font=label_font, fill=(120, 205, 255))
    y = panel[1] + 52
    for line in telemetry.splitlines():
        draw.text((panel[0] + 22, y), line, font=mono_font, fill=(235, 240, 248))
        y += 23
    draw.text(
        (panel[0] + 22, panel[3] - 38),
        "ON  1.4 deg  |  OFF  0.8 deg",
        font=mono_font,
        fill=(255, 205, 95),
    )

    legend = [
        (32, 365, "D-PAD: UP / DOWN / LEFT / RIGHT", (64, 184, 255)),
        (410, 200, "A: forward", (242, 46, 51)),
        (622, 220, "B: backward", (242, 163, 31)),
        (854, 260, "A+B: firm press", (199, 77, 209)),
    ]
    y = image.height - 60
    for x, width, text, color in legend:
        draw.rounded_rectangle((x, y, x + width, y + 38), radius=10, fill=(7, 12, 22, 225))
        draw.ellipse((x + 10, y + 10, x + 27, y + 27), fill=(*color, 255))
        draw.text((x + 34, y + 7), text, font=label_font, fill="white")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/controller_nes_preview.png",
    )
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    set_home_pose(model, data)
    # Briefly settle the contacts for the layout preview. Long-horizon balance
    # must be tested with the BAM policy, not these simple XML position motors.
    for _ in range(50):
        mujoco.mj_step(model, data)

    controller = NESController()
    addresses = controller.joint_qpos_addresses(model)
    raw = tuple(float(data.qpos[address]) for address in addresses)
    values = (raw[0], raw[1], raw[2], -raw[3])
    state = controller.update(*values)
    telemetry = controller.debug_text(state, *values)
    print(telemetry)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.012, 0.0, 0.105)
    camera.distance = 0.52
    camera.azimuth = 142.0
    camera.elevation = -28.0

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    renderer.update_scene(data, camera=camera)
    image = Image.fromarray(renderer.render())
    annotate(image, telemetry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
