"""Generate paired, labeled Mario-like visual controls for the frozen FlyVis probe.

These are deliberately simple synthetic stimuli, not NES emulator frames. Every
class appears at every position, size, and background setting so that a
decoder cannot use those nuisance variables as object labels. Real recorded
frames remain the separate transfer test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


WIDTH, HEIGHT, FLOOR_Y = 256, 240, 208
OBJECTS = ("floor", "goomba", "pipe", "gap")


def render_mario_frames(
    kind: str, *, x: int, size: int, speed: int, style: int,
    frame_count: int = 20,
) -> list[Image.Image]:
    """Translate scenery left as Mario advances, with one controlled hazard."""

    if kind not in OBJECTS or not (0 <= style <= 2):
        raise ValueError("invalid object or background style")
    if frame_count < 2 or size <= 0 or speed <= 0 or not (80 <= x <= 190):
        raise ValueError("invalid movie geometry")
    sky = ((92, 148, 252), (110, 169, 241), (75, 150, 224))[style]
    ground = ((176, 92, 32), (149, 86, 38), (189, 107, 48))[style]
    frames = []
    for t in range(frame_count):
        offset = speed * t
        hazard_x = x - offset
        image = Image.new("RGB", (WIDTH, HEIGHT), sky)
        draw = ImageDraw.Draw(image)
        # Distinct backgrounds move with the same camera motion as the hazard.
        if style >= 1:
            for center in (54, 206):
                cx = center - offset // 2
                draw.ellipse((cx - 24, 112, cx + 24, 179), fill=(63, 185, 73))
        if style == 2:
            for center in (130, 242):
                cx = center - offset // 3
                draw.ellipse((cx - 17, 39, cx + 17, 55), fill=(235, 235, 245))
        draw.rectangle((0, FLOOR_Y, WIDTH - 1, HEIGHT - 1), fill=ground)
        for seam in range(-64 - offset % 32, WIDTH + 32, 32):
            draw.line((seam, FLOOR_Y + 15, seam, HEIGHT - 1), fill=(99, 55, 36), width=2)
        draw.line((0, FLOOR_Y, WIDTH, FLOOR_Y), fill=(235, 173, 91), width=3)
        if kind == "gap":
            draw.rectangle((hazard_x, FLOOR_Y, hazard_x + size - 1, HEIGHT - 1), fill=sky)
            draw.line((hazard_x - 1, FLOOR_Y, hazard_x - 1, HEIGHT), fill=(72, 46, 30), width=2)
            draw.line((hazard_x + size, FLOOR_Y, hazard_x + size, HEIGHT), fill=(72, 46, 30), width=2)
        elif kind == "pipe":
            top = FLOOR_Y - size
            draw.rectangle((hazard_x + 3, top + 7, hazard_x + 28, FLOOR_Y), fill=(20, 170, 34))
            draw.rectangle((hazard_x, top, hazard_x + 31, top + 8), fill=(31, 202, 45))
            draw.line((hazard_x + 5, top + 9, hazard_x + 5, FLOOR_Y), fill=(90, 230, 85), width=2)
        elif kind == "goomba":
            draw.ellipse((hazard_x, FLOOR_Y - 17, hazard_x + 18, FLOOR_Y), fill=(151, 87, 46))
            draw.rectangle((hazard_x + 3, FLOOR_Y - 5, hazard_x + 16, FLOOR_Y), fill=(95, 48, 27))
            draw.point((hazard_x + 6, FLOOR_Y - 10), fill=(255, 255, 255))
            draw.point((hazard_x + 12, FLOOR_Y - 10), fill=(255, 255, 255))
        # Mario stays in screen coordinates; his pixels are identical for all classes.
        draw.rectangle((44, FLOOR_Y - 25, 55, FLOOR_Y - 17), fill=(219, 47, 47))
        draw.rectangle((42, FLOOR_Y - 17, 57, FLOOR_Y - 8), fill=(43, 77, 174))
        draw.rectangle((44, FLOOR_Y - 8, 48, FLOOR_Y - 1), fill=(90, 45, 29))
        draw.rectangle((53, FLOOR_Y - 8, 57, FLOOR_Y - 1), fill=(90, 45, 29))
        frames.append(image)
    return frames


def render_edge_frames(
    *, direction: str, contrast: str, speed: int, x: int = 128,
    frame_count: int = 20,
) -> list[Image.Image]:
    """Controlled moving ON/OFF edge, analogous to FlyVis's edge assay."""

    if direction not in ("left", "right") or contrast not in ("on", "off"):
        raise ValueError("invalid edge direction or contrast")
    if speed <= 0 or frame_count < 2:
        raise ValueError("invalid edge movie")
    frames = []
    foreground, background = ((245, 15) if contrast == "on" else (15, 245))
    for t in range(frame_count):
        position = x + (speed * t if direction == "right" else -speed * t)
        frame = np.full((HEIGHT, WIDTH, 3), background, dtype=np.uint8)
        frame[:, max(0, min(WIDTH, position)):] = foreground
        frames.append(Image.fromarray(frame))
    return frames


def save_gif(frames: list[Image.Image], path: Path) -> None:
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=20, loop=0, optimize=False)


def build_dataset(output: Path, *, frame_count: int = 20) -> dict:
    """Write matched factorial hazard movies plus isolated moving-edge controls."""

    if frame_count < 2:
        raise ValueError("frame_count must be at least two")
    output.mkdir(parents=True, exist_ok=True)
    samples = []
    for style in range(3):
        for x in (125, 165):
            for size in (32, 48):
                for speed in (2,):
                    for kind in OBJECTS:
                        stem = f"style{style}-{kind}-x{x}-size{size}-speed{speed}"
                        save_gif(render_mario_frames(kind, x=x, size=size, speed=speed,
                                                     style=style, frame_count=frame_count),
                                 output / f"{stem}.gif")
                        sample = {"id": stem, "recording_id": f"synthetic-style-{style}",
                                  "object": kind, "gif": f"{stem}.gif",
                                  "start": 0, "stop": frame_count,
                                  "stimulus_x_px": x, "stimulus_speed_px_per_frame": speed,
                                  "synthetic": True}
                        if kind == "pipe":
                            sample["pipe_height_px"] = size
                        if kind == "gap":
                            sample["gap_width_px"] = size
                        samples.append(sample)
    edges = []
    for direction in ("left", "right"):
        for contrast in ("on", "off"):
            for speed in (1, 2, 3):
                stem = f"edge-{direction}-{contrast}-speed{speed}"
                save_gif(render_edge_frames(direction=direction, contrast=contrast,
                                            speed=speed, frame_count=frame_count),
                         output / f"{stem}.gif")
                edges.append({"id": stem, "gif": f"{stem}.gif", "direction": direction,
                              "contrast": contrast, "speed_px_per_frame": speed,
                              "start": 0, "stop": frame_count})
    manifest = {"note": "Synthetic calibration only; validate on held-out NES recordings.",
                "samples": samples}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "edges.json").write_text(json.dumps({"edges": edges}, indent=2) + "\n", encoding="utf-8")
    return {"manifest": str(output / "manifest.json"), "edges": str(output / "edges.json"),
            "hazard_samples": len(samples), "edge_samples": len(edges)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.output, frame_count=args.frames), indent=2))


if __name__ == "__main__":
    main()
