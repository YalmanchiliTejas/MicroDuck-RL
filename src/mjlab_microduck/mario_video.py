"""Debug-overlay video recorder for the physical Mario controller task."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from mjlab.utils.wrappers import VideoRecorder
from PIL import Image, ImageDraw, ImageFont

from mjlab_microduck.controller import NESController, NESControllerCalibration
from mjlab_microduck.tasks import mdp

GAME_BUTTON_INDICES = (2, 3, 4)
GAME_BUTTON_NAMES = ("LEFT", "RIGHT", "JUMP")


@dataclass(frozen=True, slots=True)
class MarioFrameDiagnostics:
    """Physical controller details that disambiguate travel from a real click."""

    # Positive depression of [UP, DOWN, LEFT, RIGHT, A, B], in metres.
    joint_state: Sequence[float]
    # Stateful hysteretic decoder in [UP, DOWN, LEFT, RIGHT, A, B] order.
    decoded: Sequence[bool]
    # Raw, pre-weight travel charged to LEFT, RIGHT, and JUMP respectively.
    unrequested_travel: Sequence[float]
    unrequested_raw_total: float
    unrequested_applied_total: float
    unrequested_weighted: float
    # Per-foot global net contact-force vectors, [left, right].
    foot_forces: Sequence[Sequence[float]]


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
    diagnostics: MarioFrameDiagnostics | None = None,
) -> list[tuple[str, tuple[int, int, int]]]:
    """Build human-readable overlay lines and their status colors."""

    requested_names = [
        name
        for index, name in zip(GAME_BUTTON_INDICES, GAME_BUTTON_NAMES, strict=True)
        if requested[index] > 0.5
    ]
    request_text = "+".join(requested_names) if requested_names else "NEUTRAL"
    activation_text = "  ".join(
        f"{name}:{activation[index]:.2f}"
        for index, name in zip(GAME_BUTTON_INDICES, GAME_BUTTON_NAMES, strict=True)
    )
    progress_text = "  ".join(
        f"{name}:{progress[index]:.2f}"
        for index, name in zip(GAME_BUTTON_INDICES, GAME_BUTTON_NAMES, strict=True)
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
    lines = [
        (f"REQUEST: {request_text}", (255, 222, 80)),
        (f"activation  {activation_text}", white),
        (f"progress    {progress_text}", (185, 215, 255)),
    ]
    if diagnostics is not None:
        up, down, left, right, button_a, button_b = diagnostics.joint_state
        decoded_names = ("UP", "DOWN", "LEFT", "RIGHT", "A", "B")
        decoded_text = "  ".join(
            f"{name}:{int(active)}"
            for name, active in zip(
                decoded_names, diagnostics.decoded, strict=True
            )
        )
        wrong_binary = any(
            diagnostics.decoded[index] and requested[index] <= 0.5
            for index in GAME_BUTTON_INDICES
        )
        left_wrong, right_wrong, jump_wrong = diagnostics.unrequested_travel
        force_text = []
        for name, force in zip(("L", "R"), diagnostics.foot_forces, strict=True):
            magnitude = math.sqrt(sum(component * component for component in force))
            force_text.append(f"{name}:{magnitude:.1f}/z{force[2]:+.1f}")
        lines.extend(
            (
                (
                    "travel(mm)  "
                    f"U:{up * 1e3:.2f} D:{down * 1e3:.2f}  "
                    f"L:{left * 1e3:.2f} R:{right * 1e3:.2f}  "
                    f"A:{button_a * 1e3:.2f} B:{button_b * 1e3:.2f}",
                    (205, 205, 255),
                ),
                (
                    f"decoder     {decoded_text}",
                    red if wrong_binary else green,
                ),
                (
                    "wrong travel "
                    f"L:{left_wrong:.2f}  R:{right_wrong:.2f}  "
                    f"J:{jump_wrong:.2f}  "
                    f"RAW:{diagnostics.unrequested_raw_total:.2f}  "
                    f"APPLIED:{diagnostics.unrequested_applied_total:.2f}  "
                    f"RW:{diagnostics.unrequested_weighted:+.2f}",
                    red
                    if diagnostics.unrequested_applied_total > 0.0
                    else amber if not command_ready else green,
                ),
                (f"foot force(N) {'  '.join(force_text)}", (210, 235, 210)),
            )
        )
    lines.append(
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
        )
    )
    return lines


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

    def _frame_diagnostics(
        self,
        requested: torch.Tensor,
        joint_state: torch.Tensor,
        metrics: dict[str, float],
    ) -> MarioFrameDiagnostics:
        if not hasattr(self, "_mario_nes_decoder"):
            reward_cfg = self._wrapped_env.reward_manager.get_term_cfg(
                "unrequested_button"
            )
            self._mario_nes_decoder = NESController(
                NESControllerCalibration(
                    activate_travel=float(reward_cfg.params["activate_angle"]),
                    release_travel=float(reward_cfg.params["release_angle"]),
                )
            )
        state = joint_state.detach().cpu().tolist()
        decoded_state = self._mario_nes_decoder.update(*state[2:6])
        decoded_dict = decoded_state.as_dict()
        decoded = tuple(
            decoded_dict[name] for name in ("UP", "DOWN", "LEFT", "RIGHT", "A", "B")
        )

        reward_cfg = self._wrapped_env.reward_manager.get_term_cfg(
            "unrequested_button"
        )
        activate_angle = float(reward_cfg.params["activate_angle"])
        release_angle = float(reward_cfg.params["release_angle"])
        enabled = reward_cfg.params.get("enabled_buttons")
        if enabled is None:
            enabled = (True,) * 6
        requested_values = requested.detach().cpu().tolist()
        left_travel, right_travel, jump_travel = state[2], state[3], state[4]
        span = max(activate_angle - release_angle, 1e-6)
        raw_game_travel = (
            max(left_travel - release_angle, 0.0) / span,
            max(right_travel - release_angle, 0.0) / span,
            max(jump_travel - release_angle, 0.0) / span,
        )
        contributions = tuple(
            travel * (1.0 - requested_values[index]) * float(enabled[index])
            for travel, index in zip(
                raw_game_travel, GAME_BUTTON_INDICES, strict=True
            )
        )
        raw_total = sum(contributions)
        command_ready = metrics.get("command_ready", 1.0) >= 0.5
        applied_total = raw_total if command_ready else 0.0

        sensor = self._wrapped_env.scene.sensors["feet_ground_contact"]
        force = sensor.data.force[0].detach().cpu()
        if force.ndim == 1:
            force = force.reshape(1, 3)
        if force.shape[0] != 2:
            raise RuntimeError(
                "Mario diagnostics expected left/right contact-force vectors"
            )
        foot_forces = tuple(tuple(float(value) for value in row) for row in force)
        return MarioFrameDiagnostics(
            joint_state=state,
            decoded=decoded,
            unrequested_travel=contributions,
            unrequested_raw_total=raw_total,
            unrequested_applied_total=applied_total,
            unrequested_weighted=applied_total * float(reward_cfg.weight),
            foot_forces=foot_forces,
        )

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
            joint_state = mdp.mario_nes_joint_state(self._wrapped_env)[0]
        metrics = self._metric_values()
        diagnostics = self._frame_diagnostics(requested, joint_state, metrics)
        lines = mario_overlay_lines(
            requested.detach().cpu().tolist(),
            activation.detach().cpu().tolist(),
            progress.detach().cpu().tolist(),
            metrics,
            diagnostics,
        )
        self.current_video_frames.append(annotate_mario_frame(rgb_frame, lines))
