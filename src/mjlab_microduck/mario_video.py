"""Debug-overlay video recorder for the physical Mario controller task."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from mjlab.utils.wrappers import VideoRecorder
from PIL import Image, ImageDraw, ImageFont

from mjlab_microduck.tasks import mdp

BUTTON_NAMES = ("UP", "DOWN", "LEFT", "RIGHT", "A", "B")


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def mario_overlay_lines(
    requested: Sequence[float],
    activation: Sequence[float],
    progress: Sequence[float],
    metrics: dict[str, float],
) -> list[tuple[str, tuple[int, int, int]]]:
    """Build human-readable overlay lines and their status colors."""

    requested_names = [
        name for name, value in zip(BUTTON_NAMES, requested, strict=True) if value > 0.5
    ]
    request_text = "+".join(requested_names) if requested_names else "NEUTRAL"
    activation_text = "  ".join(
        f"{name}:{value:.2f}"
        for name, value in zip(BUTTON_NAMES, activation, strict=True)
    )
    progress_text = "  ".join(
        f"{name}:{value:.2f}"
        for name, value in zip(BUTTON_NAMES, progress, strict=True)
    )

    success = metrics.get("requested_button_success", 0.0) >= 0.5
    feet = metrics.get("feet_anchored", 0.0) >= 0.5
    pose = metrics.get("standing_pose_ready", 0.0) >= 0.5
    camera = metrics.get("camera_ready", 0.0) >= 0.5
    command_ready = metrics.get("command_ready", 1.0) >= 0.5
    green = (105, 235, 140)
    red = (255, 105, 105)
    amber = (255, 205, 95)
    white = (245, 245, 245)
    return [
        (f"REQUEST: {request_text}", (255, 222, 80)),
        (f"activation  {activation_text}", white),
        (f"progress    {progress_text}", (185, 215, 255)),
        (
            "  ".join(
                (
                    (
                        "SUCCESS:WAIT"
                        if not command_ready
                        else f"SUCCESS:{'PASS' if success else 'FAIL'}"
                    ),
                    f"FEET:{'PASS' if feet else 'FAIL'}",
                    f"POSE:{'PASS' if pose else 'FAIL'}",
                    f"CAMERA:{'PASS' if camera else 'FAIL'}",
                    f"CMD:{'READY' if command_ready else 'WAIT'}",
                )
            ),
            (
                amber
                if not command_ready
                else green if success and feet and pose and camera else red
            ),
        ),
    ]


def annotate_mario_frame(
    frame: np.ndarray,
    lines: Sequence[tuple[str, tuple[int, int, int]]],
) -> np.ndarray:
    """Draw a translucent, resolution-independent diagnostics panel."""

    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)
    image = Image.fromarray(array, mode="RGB").convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font_size = max(13, round(image.height / 40))
    font = _font(font_size)
    line_height = font_size + max(4, font_size // 3)
    panel_height = 16 + line_height * len(lines)
    draw.rounded_rectangle(
        (10, 10, image.width - 10, panel_height),
        radius=8,
        fill=(0, 0, 0, 185),
        outline=(255, 255, 255, 100),
        width=1,
    )
    for index, (text, color) in enumerate(lines):
        draw.text((20, 16 + index * line_height), text, font=font, fill=(*color, 255))
    return np.asarray(Image.alpha_composite(image, layer).convert("RGB"))


class MarioDebugVideoRecorder(VideoRecorder):
    """Video recorder that annotates the current command and physical state."""

    def _metric_values(self) -> dict[str, float]:
        return {
            name: float(values[0])
            for name, values in self._wrapped_env.metrics_manager.get_active_iterable_terms(0)
        }

    def _record_frame(self) -> None:
        if self._wrapped_env.render_mode != "rgb_array":
            return
        frame = self._wrapped_env.render()
        if frame is None:
            return
        rgb_frame = (
            frame[0] if isinstance(frame, np.ndarray) and frame.ndim == 4 else frame
        )
        with torch.no_grad():
            requested = mdp.mario_nes_requested_buttons(self._wrapped_env)[0]
            activation = mdp.mario_nes_activation(self._wrapped_env)[0]
            progress = mdp.mario_nes_progress(self._wrapped_env)[0]
        lines = mario_overlay_lines(
            requested.detach().cpu().tolist(),
            activation.detach().cpu().tolist(),
            progress.detach().cpu().tolist(),
            self._metric_values(),
        )
        self.current_video_frames.append(annotate_mario_frame(rgb_frame, lines))
