#!/usr/bin/env python3
"""Render a preview frame of the Microduck-controlled platform game."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from mjlab_microduck.controller_game import Button, PadGameBridge


ROOT = Path(__file__).resolve().parents[1]


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_preview(output: Path, width: int = 1200, height: int = 600) -> None:
    bridge = PadGameBridge()
    dt = 1.0 / 50.0
    pad_frame = None
    # Move toward the gap and tap jump. Stop while airborne for the preview.
    for index in range(58):
        t = index * dt
        pad_frame, state = bridge.step(
            {
                Button.RIGHT: 0.006,
                Button.JUMP: 0.006 if 0.85 <= t < 0.95 else 0.0,
            },
            dt,
        )
    assert pad_frame is not None

    image = Image.new("RGB", (width, height), (83, 170, 255))
    draw = ImageDraw.Draw(image)
    title_font = _font(36)
    hud_font = _font(24)

    # Sky decoration.
    for cx, cy, scale in ((170, 115, 1.0), (760, 90, 0.8), (1030, 155, 1.1)):
        white = (245, 250, 255)
        draw.ellipse((cx, cy, cx + 90 * scale, cy + 38 * scale), fill=white)
        draw.ellipse((cx + 30 * scale, cy - 25 * scale, cx + 105 * scale, cy + 40 * scale), fill=white)
        draw.ellipse((cx + 75 * scale, cy, cx + 145 * scale, cy + 38 * scale), fill=white)

    # Background hills.
    draw.polygon(((0, 420), (175, 245), (350, 420)), fill=(87, 190, 104))
    draw.polygon(((250, 420), (510, 215), (760, 420)), fill=(66, 166, 87))
    draw.polygon(((720, 420), (960, 250), (1200, 420)), fill=(84, 184, 98))

    cfg = bridge.game.config
    world_left, world_right = 0.0, 10.5
    ground_y = 455

    def sx(x: float) -> int:
        return int(55 + (x - world_left) / (world_right - world_left) * (width - 110))

    # Brick platforms, matching the actual ground segments used by physics.
    for start, end in cfg.ground_segments:
        x0, x1 = sx(start), sx(end)
        draw.rectangle((x0, ground_y, x1, height), fill=(174, 89, 47), outline=(92, 46, 30), width=4)
        brick_w, brick_h = 42, 30
        for row, y in enumerate(range(ground_y, height, brick_h)):
            offset = brick_w // 2 if row % 2 else 0
            for x in range(x0 - offset, x1, brick_w):
                draw.line((x, y, x, min(y + brick_h, height)), fill=(112, 55, 35), width=2)
            draw.line((x0, y, x1, y), fill=(112, 55, 35), width=2)

    # Goal pole and flag.
    goal_x = sx(cfg.goal_x)
    draw.rectangle((goal_x - 4, 235, goal_x + 4, ground_y), fill=(235, 239, 225))
    draw.ellipse((goal_x - 11, 222, goal_x + 11, 244), fill=(255, 218, 67))
    draw.polygon(((goal_x + 4, 250), (goal_x + 92, 275), (goal_x + 4, 302)), fill=(255, 83, 68))

    # Original red-capped platform character (simple vector art, no ROM asset).
    px = sx(state.x)
    py = int(ground_y - (state.y - cfg.player_radius) * 155 - 54)
    draw.ellipse((px - 24, py - 38, px + 24, py + 10), fill=(247, 196, 146), outline=(57, 48, 43), width=3)
    draw.rectangle((px - 30, py - 42, px + 20, py - 25), fill=(226, 50, 45))
    draw.rectangle((px - 33, py - 28, px + 32, py - 20), fill=(226, 50, 45))
    draw.rectangle((px - 24, py + 4, px + 24, py + 52), fill=(48, 82, 190), outline=(40, 45, 65), width=3)
    draw.rectangle((px - 29, py + 3, px + 29, py + 19), fill=(226, 50, 45))
    draw.ellipse((px + 7, py - 22, px + 12, py - 15), fill=(32, 32, 38))
    draw.rectangle((px - 28, py + 48, px - 2, py + 58), fill=(87, 48, 32))
    draw.rectangle((px + 3, py + 48, px + 31, py + 58), fill=(87, 48, 32))

    # HUD and physical-controller state.
    draw.rounded_rectangle((24, 20, width - 24, 80), radius=16, fill=(24, 33, 54, 225))
    draw.text((48, 30), "MICRODUCK PLATFORM", font=title_font, fill=(255, 255, 255))
    draw.text((820, 37), f"DISTANCE  {state.x:04.1f} / {cfg.goal_x:04.1f}", font=hud_font, fill=(255, 232, 91))

    buttons = (
        ("LEFT", pad_frame.controller.left, (70, 505, 230, 575)),
        ("RIGHT", pad_frame.controller.right, (255, 505, 435, 575)),
        ("JUMP", pad_frame.controller.jump, (460, 505, 630, 575)),
    )
    for label, active, box in buttons:
        fill = (56, 116, 245) if label != "JUMP" else (241, 69, 58)
        if not active:
            fill = tuple(channel // 2 for channel in fill)
        draw.rounded_rectangle(box, radius=14, fill=fill, outline=(245, 248, 255), width=3)
        text_box = draw.textbbox((0, 0), label, font=hud_font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        draw.text(
            ((box[0] + box[2] - text_w) / 2, (box[1] + box[3] - text_h) / 2 - 3),
            label,
            font=hud_font,
            fill=(255, 255, 255),
        )

    draw.multiline_text(
        (690, 510),
        "ONLY MEASURED PAD PRESSES\nMOVE THE GAME",
        font=_font(20),
        fill=(20, 38, 55),
        spacing=5,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "mario_game_preview.png")
    args = parser.parse_args()
    render_preview(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
