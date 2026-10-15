"""Evaluate a frozen MaleCNS Mario policy without any learning."""

from __future__ import annotations

from mario_action_macros import MACRO_FRAMES, action_frames, append_executed_frames

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from male_cns import VISUAL_ABLATIONS, MaleCNS
from mario_dqn import ACTION_LEVELS, ActivityStack, FlybrainAction, FlybrainAgent
from mario_sidecar import nes_actions
from rollouts import REWARD_COMPONENTS, training_reward


def _device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _legacy_training_reward(components: dict[str, float], raw_reward: float) -> float:
    """Reproduce the reward used by the compromised schema-4 run."""

    if not components:
        return float(np.clip(raw_reward / 10.0, -5.0, 5.0))
    value = (
        components.get("progress", 0.0) / 10.0
        + components.get("time", 0.0) / 20.0
        + components.get("score", 0.0) / 5.0
        + components.get("coins", 0.0) / 5.0
        + components.get("powerup", 0.0) / 5.0
        + components.get("completion", 0.0) / 10.0
        + components.get("death", 0.0) / 10.0
    )
    return float(np.clip(value, -5.0, 5.0))


def _atomic_json(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    temporary.replace(path)


JUMP_ACTIONS = tuple(index for index, levels in enumerate(ACTION_LEVELS) if levels[2])
NON_JUMP_ACTIONS = tuple(index for index, levels in enumerate(ACTION_LEVELS) if not levels[2])


def _jump_metrics(scores: np.ndarray) -> dict[str, float]:
    """Report marginal and strongest competing action probabilities."""

    scores = np.asarray(scores, dtype=np.float64)
    return {
        "total_jump_probability": float(scores[list(JUMP_ACTIONS)].sum()),
        "best_jump_probability": float(scores[list(JUMP_ACTIONS)].max()),
        "best_non_jump_probability": float(scores[list(NON_JUMP_ACTIONS)].max()),
    }


def _ppo_input_sensitivity(agent, state: np.ndarray) -> dict[str, list[float]]:
    """Local gradient of total jump probability through the PPO readout."""

    from mario_ppo import _factorized_probabilities

    state_tensor = torch.as_tensor(
        state, dtype=torch.float32, device=agent.device
    ).unsqueeze(0).detach().requires_grad_(True)
    logits, _ = agent.network(state_tensor)
    probabilities = (
        _factorized_probabilities(logits)
        if agent.config.policy_mode == "factorized"
        else torch.softmax(logits, dim=1)
    )
    jump_probability = probabilities[:, list(JUMP_ACTIONS)].sum()
    gradient = torch.autograd.grad(jump_probability, state_tensor)[0][0]
    gradient = gradient[:, : agent.config.feature_dim]
    return {
        "jump_probability_gradient_rms_by_frame": [
            float(value) for value in gradient.square().mean(dim=1).sqrt().detach().cpu()
        ],
        "jump_probability_gradient_max_by_frame": [
            float(value) for value in gradient.abs().amax(dim=1).detach().cpu()
        ],
    }


def _ppo_recurrent_input_sensitivity(
    agent, tokens: np.ndarray, initial_hidden: np.ndarray
) -> dict[str, list[float]]:
    """Local jump-probability gradient through the current GRU frame tokens."""

    from mario_ppo import EPISODE_GRU_HIDDEN_DIM, _factorized_probabilities

    token_tensor = torch.as_tensor(
        tokens, dtype=torch.float32, device=agent.device
    ).unsqueeze(0).detach().requires_grad_(True)
    hidden_tensor = torch.as_tensor(
        initial_hidden, dtype=torch.float32, device=agent.device
    ).view(1, 1, EPISODE_GRU_HIDDEN_DIM)
    # cuDNN intentionally disables RNN backward while a module is in eval mode.
    # This diagnostic needs an input gradient, but it must not change policy
    # parameters or leave evaluation behavior enabled as training. A one-layer
    # GRU has no dropout, so toggling the execution mode does not change its
    # forward result.
    was_training = agent.network.training
    agent.network.train(True)
    try:
        logits, _, _ = agent.network.recurrent_tokens(token_tensor, hidden_tensor)
        probabilities = (
            _factorized_probabilities(logits)
            if agent.config.policy_mode == "factorized"
            else torch.softmax(logits, dim=1)
        )
        jump_probability = probabilities[:, list(JUMP_ACTIONS)].sum()
        gradient = torch.autograd.grad(jump_probability, token_tensor)[0][0]
    finally:
        agent.network.train(was_training)
    gradient = gradient[:, : agent.config.feature_dim]
    return {
        "jump_probability_gradient_rms_by_frame": [
            float(value) for value in gradient.square().mean(dim=1).sqrt().detach().cpu()
        ],
        "jump_probability_gradient_max_by_frame": [
            float(value) for value in gradient.abs().amax(dim=1).detach().cpu()
        ],
    }


def _replay_counterfactuals(
    agent,
    connectome: MaleCNS,
    initial_frame: np.ndarray,
    intervals: list[tuple[int, list[np.ndarray]]],
    decision_rows: list[dict],
    *,
    episode_seed: int,
) -> float:
    """Replay identical frames and actions with each injected channel silenced."""

    from mario_ppo import (
        ControllerTemporalStack, EpisodeGRUController, VisionFusionGRUController
    )

    max_baseline_error = 0.0
    original_ablation = connectome.visual_ablation
    try:
        for ablation in VISUAL_ABLATIONS:
            connectome.visual_ablation = ablation
            episode_recurrent = agent.config.temporal_encoder in (
                "episode_gru_v1", "vision_fusion_gru_v2"
            )
            if episode_recurrent:
                stack = (
                    VisionFusionGRUController(agent.config.feature_dim)
                    if agent.config.temporal_encoder == "vision_fusion_gru_v2"
                    else EpisodeGRUController(agent.config.feature_dim)
                )
                agent.reset_recurrent()
                tokens = stack.reset(
                    connectome.reset(initial_frame, seed=episode_seed)
                )
                agent.observe_recurrent(tokens)
                state = None
            else:
                stack = (
                    ControllerTemporalStack(
                        agent.config.feature_dim, agent.config.stack_depth
                    )
                    if agent.config.temporal_encoder == "controller_gru"
                    else ActivityStack(agent.config.stack_depth)
                )
                state = stack.reset(
                    connectome.reset(initial_frame, seed=episode_seed)
                )
            for row, (action, frames) in zip(decision_rows, intervals, strict=True):
                scores = np.asarray(
                    (
                        agent.policy_diagnostics_recurrent()
                        if episode_recurrent
                        else agent.policy_diagnostics(state)
                    )["action_probabilities"],
                    dtype=np.float64,
                )
                row.setdefault("counterfactual", {})[ablation] = {
                    "greedy_action": FlybrainAction(int(scores.argmax())).name.lower(),
                    **_jump_metrics(scores),
                }
                if ablation == "normal":
                    original = np.asarray(
                        list(row["action_probabilities"].values()), dtype=np.float64
                    )
                    max_baseline_error = max(
                        max_baseline_error, float(np.max(np.abs(scores - original)))
                    )
                activities = []
                for frame in frames:
                    activity = connectome.observe(frame)
                    if agent.config.temporal_encoder in (
                        "controller_gru", "episode_gru_v1", "vision_fusion_gru_v2"
                    ):
                        activities.append(activity)
                    else:
                        stack.append(activity)
                if episode_recurrent:
                    agent.observe_recurrent(
                        stack.append_interval(activities, action)
                    )
                elif agent.config.temporal_encoder == "controller_gru":
                    state = stack.append_interval(activities, action)
                else:
                    state = stack.state
    finally:
        connectome.visual_ablation = original_ablation
    for row in decision_rows:
        results = row["counterfactual"]
        baseline = results["normal"]
        for name, result in results.items():
            result["delta_jump_probability"] = (
                result["total_jump_probability"]
                - baseline["total_jump_probability"]
            )
            result["changed_greedy_action"] = (
                result["greedy_action"] != baseline["greedy_action"]
            )
    return max_baseline_error


def _counterfactual_summary(decision_rows: list[dict]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean_absolute_jump_probability_change": float(np.mean([
                abs(row["counterfactual"][name]["delta_jump_probability"])
                for row in decision_rows
            ])),
            "greedy_action_change_fraction": float(np.mean([
                row["counterfactual"][name]["changed_greedy_action"]
                for row in decision_rows
            ])),
        }
        for name in VISUAL_ABLATIONS
        if name != "normal"
    }


class ForensicRecorder:
    """Record one emulator episode as synchronized GIF and JSONL traces."""

    def __init__(
        self, directory: Path, episode: int, fps: int, *,
        retina_view: bool = False, flyvis_view: bool = False,
    ) -> None:
        self.directory = directory
        self.episode = episode
        self.fps = fps
        self.frames = []
        self.frame_rows: list[dict] = []
        self.decision_rows: list[dict] = []
        self.retina_view = retina_view
        self.retina_frames = []
        self.flyvis_view = flyvis_view
        self.flyvis_frames = []

    def add_frame(self, frame: np.ndarray, row: dict, overlay: list[str]) -> None:
        from PIL import Image, ImageDraw

        game = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
        line_height = 11
        overlay_height = line_height * len(overlay) + 4
        image = Image.new("RGB", (game.width, game.height + overlay_height), "black")
        image.paste(game, (0, 0))
        draw = ImageDraw.Draw(image)
        for line, text in enumerate(overlay):
            draw.text(
                (3, game.height + 2 + line * line_height),
                text,
                fill=(255, 255, 255),
            )
        self.frames.append(image)
        self.frame_rows.append(row)

    def add_decision(self, row: dict) -> None:
        self.decision_rows.append(row)

    def add_flyvis_frame(
        self, frame: np.ndarray, grids: np.ndarray, row: dict
    ) -> None:
        """Render the game beside signed spatial activity for all bridge types."""

        from PIL import Image, ImageDraw

        cell_types = (
            "R1", "L1", "L2", "T4a", "T4b", "T4c", "T4d",
            "T5a", "T5b", "T5c", "T5d",
        )
        values = np.asarray(grids, dtype=np.float32)
        if values.shape != (len(cell_types), 6, 8):
            raise ValueError("FlyVis display requires an 11x6x8 spatial grid")
        game = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
        width, height = game.size
        label_height = 28
        canvas = Image.new(
            "RGB", (4 * width, 3 * (height + label_height)), "black"
        )
        draw = ImageDraw.Draw(canvas)
        canvas.paste(game, (0, label_height))
        draw.text(
            (3, 2),
            f"game  decision={row['decision']} x={row['x']} "
            f"action={row['action']} death={int(row['death'])}",
            fill="white",
        )
        centers = {"R1": 1.15, "L1": -1.55, "L2": -1.55}
        scales = {"R1": 0.35, "L1": 0.45, "L2": 0.45}
        for panel_index, (cell_type, activity) in enumerate(
            zip(cell_types, values, strict=True), start=1
        ):
            column = panel_index % 4
            row_index = panel_index // 4
            x = column * width
            y = row_index * (height + label_height)
            center = centers.get(cell_type, 0.0)
            scale = scales.get(cell_type, 0.35)
            signed = np.tanh((activity - center) / scale)
            positive = np.clip(signed, 0.0, 1.0)
            negative = np.clip(-signed, 0.0, 1.0)
            rgb = np.stack(
                (255 * positive, 80 * (positive + negative), 255 * negative),
                axis=-1,
            ).astype(np.uint8)
            panel = Image.fromarray(rgb, "RGB").resize(
                (width, height), Image.Resampling.NEAREST
            )
            canvas.paste(panel, (x, y + label_height))
            draw.text(
                (x + 3, y + 2),
                f"{cell_type}  raw={activity.min():+.2f}..{activity.max():+.2f}",
                fill="white",
            )
            draw.text(
                (x + 3, y + 14),
                "blue=below rest  red=above rest",
                fill=(190, 195, 205),
            )
            draw.line(
                (
                    x + width // 2, y + label_height,
                    x + width // 2, y + label_height + height,
                ),
                fill=(90, 90, 90),
            )
        self.flyvis_frames.append(canvas)

    def add_retina_frame(
        self, frame: np.ndarray, diagnostic: dict, row: dict
    ) -> None:
        """One sampled frame per decision; source maps above pooled channel maps."""

        from PIL import Image, ImageDraw

        game = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
        width, height = game.size
        spatial = "ground_presence" in diagnostic
        columns = 5 if spatial else 4
        panels = [
            ("game", None, None),
            ("motion x4", "motion", (255, 255, 255)),
            ("dark growth x4", "dark_growth", (255, 255, 255)),
            ("vertical edges x4", "edges", (255, 255, 255)),
        ]
        if spatial:
            panels.append(("ground presence", "ground_presence", (80, 235, 130)))
        panels.extend((
            ("small-motion contributors x6", "small_motion", (80, 190, 255)),
            ("loom contributors x6", "loom", (225, 90, 255)),
            ("threat contributors x6", "threat", (255, 85, 85)),
            ("target contributors x6", "target", (255, 220, 75)),
        ))
        if spatial:
            panels.append(("vertical obstacle height", "obstacle_height", (255, 150, 55)))
        canvas = Image.new(
            "RGB", (columns * width, 2 * (height + 16) + (96 if spatial else 72)),
            "black",
        )
        draw = ImageDraw.Draw(canvas)
        for index, (label, key, color) in enumerate(panels):
            x = (index % columns) * width
            y = (index // columns) * (height + 16)
            draw.text((x + 3, y + 2), label, fill="white")
            if key is None:
                panel = game
            else:
                strength = np.clip(
                    np.asarray(diagnostic[key], dtype=np.float32)
                    * (4.0 if key in ("motion", "dark_growth", "edges") else 6.0),
                    0.0, 1.0,
                )
                rgb = np.stack([strength * component for component in color], axis=-1)
                panel = Image.fromarray(rgb.astype(np.uint8), "RGB")
            canvas.paste(panel, (x, y + 16))
            if key is not None:
                draw.line(
                    (x + width // 2, y + 16, x + width // 2, y + 16 + height),
                    fill=(95, 95, 95),
                )
        base_y = 2 * (height + 16) + 3
        visual = row["visual_drive"]
        draw.text(
            (3, base_y),
            f"episode={row['episode']} decision={row['decision']} "
            f"x={row['x']} y={row['y']} chosen={row['action']} "
            f"shift={diagnostic['camera_shift_pixels']}",
            fill="white",
        )
        draw.text(
            (3, base_y + 14),
            "post-action L/R drive " + "  ".join(
                f"{name}={visual[name + '_l']:.3f}/{visual[name + '_r']:.3f}"
                for name in ("small_motion", "loom", "threat", "target")
            ),
            fill="white",
        )
        jump = row["jump_metrics"]
        draw.text(
            (3, base_y + 28),
            f"pre-action P(jump)={jump['total_jump_probability']:.3f} "
            f"press={int(row['jump_pressed_edge'])} "
            f"release={int(row['jump_released_edge'])} "
            f"hold={row['jump_hold_decisions']} death={int(row['death'])}"
            if jump is not None else f"death={int(row['death'])}",
            fill="white",
        )
        draw.text(
            (3, base_y + 42),
            (
                "Channel panels show motion/edge contributors; green/orange panels "
                "show the added spatial inputs."
                if spatial else
                "Channel panels show top pixels pooled per LEFT/RIGHT half; "
                "target's constant 0.04 baseline is not pictured."
            ),
            fill="white",
        )
        if spatial:
            measures = row["retina_measurements"]
            draw.text(
                (3, base_y + 56),
                "gap near/mid/far=" + "/".join(
                    f"{measures[f'gap_{distance}']:.2f}"
                    for distance in ("near", "mid", "far")
                ) + "   obstacle near/mid/far=" + "/".join(
                    f"{measures[f'obstacle_{distance}']:.2f}"
                    for distance in ("near", "mid", "far")
                ) + f"   player screen x={measures['player_screen_x']}",
                fill="white",
            )
            draw.text(
                (3, base_y + 70),
                "Green=brick-like floor below; orange=pipe-like green vertical span. "
                "These are pixel filters, not object labels.",
                fill="white",
            )
        self.retina_frames.append(canvas)

    def _write_retina_timeline(self, path: Path) -> None:
        from PIL import Image, ImageDraw

        last_by_decision = {row["decision"]: row for row in self.frame_rows}
        rows = [last_by_decision[index] for index in sorted(last_by_decision)]
        if not rows:
            return
        width, left, right = 1100, 130, 25
        lane_height = 95
        image = Image.new("RGB", (width, 5 * lane_height + 30), (15, 17, 25))
        draw = ImageDraw.Draw(image)
        plot_width = width - left - right
        count = len(rows)
        def px(index: int) -> int:
            return left + round(index * plot_width / max(1, count - 1))

        for index, row in enumerate(rows):
            if row["jump_pressed_edge"]:
                draw.line((px(index), 16, px(index), 5 * lane_height + 15), fill=(72, 35, 35))
            if row["death"]:
                draw.line((px(index), 16, px(index), 5 * lane_height + 15), fill=(255, 80, 80), width=2)
        for lane, name in enumerate(("small_motion", "loom", "threat", "target")):
            top = 20 + lane * lane_height
            values = [
                float(row["visual_drive"][f"{name}_{side}"])
                for row in rows for side in ("l", "r")
            ]
            ceiling = max(0.01, max(values) * 1.05)
            draw.text((4, top), name, fill="white")
            draw.text((4, top + 16), f"0..{ceiling:.3f}", fill=(170, 175, 185))
            draw.line((left, top + 68, width - right, top + 68), fill=(85, 88, 95))
            for side, color in (("l", (70, 190, 255)), ("r", (255, 195, 70))):
                points = [
                    (px(i), top + 68 - round(56 * float(row["visual_drive"][f"{name}_{side}"]) / ceiling))
                    for i, row in enumerate(rows)
                ]
                if len(points) > 1:
                    draw.line(points, fill=color, width=2)
        top = 20 + 4 * lane_height
        max_x = max(1, max(int(row["x"]) for row in rows))
        draw.text((4, top), f"Mario x 0..{max_x}", fill="white")
        draw.line((left, top + 68, width - right, top + 68), fill=(85, 88, 95))
        points = [(px(i), top + 68 - round(56 * int(row["x"]) / max_x)) for i, row in enumerate(rows)]
        if len(points) > 1:
            draw.line(points, fill=(160, 235, 150), width=2)
        draw.text(
            (left, 5 * lane_height + 12),
            "cyan=left  amber=right  red vertical=jump press  bright red=death; horizontal=decision",
            fill="white",
        )
        image.save(path)

    def _write_spatial_timeline(self, path: Path) -> None:
        from PIL import Image, ImageDraw

        last_by_decision = {row["decision"]: row for row in self.frame_rows}
        rows = [last_by_decision[index] for index in sorted(last_by_decision)]
        if not rows or "gap_near" not in rows[0].get("retina_measurements", {}):
            return
        width, left, right = 1100, 155, 25
        image = Image.new("RGB", (width, 350), (15, 17, 25))
        draw = ImageDraw.Draw(image)

        def px(index: int) -> int:
            return left + round(index * (width - left - right) / max(1, len(rows) - 1))

        for index, row in enumerate(rows):
            color = (255, 85, 85) if row["death"] else (75, 40, 40)
            if row["death"] or row["jump_pressed_edge"]:
                draw.line((px(index), 15, px(index), 322), fill=color)
        for lane, name in enumerate(("gap", "obstacle")):
            top = 25 + lane * 105
            draw.text((5, top), f"{name} ahead (0..1)", fill="white")
            draw.line((left, top + 75, width - right, top + 75), fill=(85, 88, 95))
            for distance, color in (
                ("near", (255, 110, 95)),
                ("mid", (255, 205, 80)),
                ("far", (90, 200, 255)),
            ):
                points = [
                    (
                        px(index),
                        top + 75 - round(
                            65 * float(row["retina_measurements"][f"{name}_{distance}"])
                        ),
                    )
                    for index, row in enumerate(rows)
                ]
                if len(points) > 1:
                    draw.line(points, fill=color, width=2)
        top = 245
        maximum_x = max(1, max(int(row["x"]) for row in rows))
        draw.text((5, top), f"Mario x (0..{maximum_x})", fill="white")
        draw.line((left, top + 60, width - right, top + 60), fill=(85, 88, 95))
        points = [
            (px(index), top + 60 - round(50 * int(row["x"]) / maximum_x))
            for index, row in enumerate(rows)
        ]
        if len(points) > 1:
            draw.line(points, fill=(150, 235, 150), width=2)
        draw.text(
            (left, 328),
            "red=near  amber=mid  blue=far  dim vertical=jump press  bright vertical=death",
            fill="white",
        )
        image.save(path)

    def annotate_counterfactuals(self, *, valid: bool) -> None:
        """Add replay results to each GIF frame after the paired replay finishes."""

        from PIL import Image, ImageDraw

        by_decision = {row["decision"]: row for row in self.decision_rows}
        annotated = []
        for image, frame_row in zip(self.frames, self.frame_rows, strict=True):
            results = by_decision[frame_row["decision"]].get("counterfactual", {})
            if not results:
                annotated.append(image)
                continue
            lines = [
                f"cf {name}: Pj={results[name]['total_jump_probability']:.2f} "
                f"best={''.join(part[0].upper() for part in results[name]['greedy_action'].split('_'))}"
                for name in VISUAL_ABLATIONS
                if name in results
            ]
            if not valid:
                lines.append("REPLAY INVALID: baseline differs")
            extra_height = len(lines) * 11 + 4
            canvas = Image.new("RGB", (image.width, image.height + extra_height), "black")
            canvas.paste(image, (0, 0))
            draw = ImageDraw.Draw(canvas)
            for line_number, line in enumerate(lines):
                draw.text((3, image.height + 2 + line_number * 11), line, fill="white")
            annotated.append(canvas)
        self.frames = annotated

    def close(self) -> dict[str, str]:
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = f"episode-{self.episode:03d}"
        frame_trace = self.directory / f"{stem}-frames.jsonl"
        decision_trace = self.directory / f"{stem}-decisions.jsonl"
        animation = self.directory / f"{stem}.gif"
        _atomic_jsonl(frame_trace, self.frame_rows)
        _atomic_jsonl(decision_trace, self.decision_rows)
        if self.frames:
            temporary = animation.with_name(f".{animation.name}.tmp.gif")
            self.frames[0].save(
                temporary,
                save_all=True,
                append_images=self.frames[1:],
                duration=max(1, round(1000 / self.fps)),
                loop=0,
                optimize=False,
            )
            temporary.replace(animation)
        artifacts = {
            "video": str(animation.resolve()),
            "frame_trace": str(frame_trace.resolve()),
            "decision_trace": str(decision_trace.resolve()),
        }
        if self.retina_frames:
            retina_animation = self.directory / f"{stem}-retina.gif"
            temporary = retina_animation.with_name(f".{retina_animation.name}.tmp.gif")
            self.retina_frames[0].save(
                temporary, save_all=True, append_images=self.retina_frames[1:],
                duration=max(1, round(1000 / self.fps)), loop=0, optimize=False,
            )
            temporary.replace(retina_animation)
            timeline = self.directory / f"{stem}-retina-signals.png"
            self._write_retina_timeline(timeline)
            artifacts["retina_video"] = str(retina_animation.resolve())
            artifacts["retina_timeline"] = str(timeline.resolve())
            if "gap_near" in self.frame_rows[0].get("retina_measurements", {}):
                spatial_timeline = self.directory / f"{stem}-spatial-signals.png"
                self._write_spatial_timeline(spatial_timeline)
                artifacts["spatial_timeline"] = str(spatial_timeline.resolve())
        if self.flyvis_frames:
            flyvis_animation = self.directory / f"{stem}-flyvis.gif"
            temporary = flyvis_animation.with_name(
                f".{flyvis_animation.name}.tmp.gif"
            )
            self.flyvis_frames[0].save(
                temporary,
                save_all=True,
                append_images=self.flyvis_frames[1:],
                duration=max(1, round(1000 / self.fps)),
                loop=0,
                optimize=False,
            )
            temporary.replace(flyvis_animation)
            artifacts["flyvis_video"] = str(flyvis_animation.resolve())
        return artifacts


def run(args: argparse.Namespace) -> dict:
    import gym_super_mario_bros  # noqa: F401 -- registers environments
    import gymnasium as gym
    from nes_py.wrappers import JoypadSpace

    device = _device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    algorithm = str(payload.get("algorithm", "dqn"))
    if algorithm == "ppo":
        from mario_ppo import (
            ControllerTemporalStack, EpisodeGRUController, PPOAgent,
            VisionFusionGRUController,
        )

        agent = PPOAgent.load(args.checkpoint, device=device)
        agent.network.eval()
    else:
        agent = FlybrainAgent.load(
            args.checkpoint,
            device=device,
            allow_legacy_reward_contract=args.allow_legacy_reward_contract,
        )
        agent.online.eval()
    bounded_actions = bool(getattr(agent.config, "bounded_jump_actions", False))
    if bounded_actions and getattr(args, "counterfactual_channels", False):
        raise ValueError("counterfactual replay does not yet support bounded-action feedback; use full-rollout ablations")
    connectome = MaleCNS(
        data=args.male_cns_data,
        device=args.male_cns_device,
        seed=args.seed,
        dopamine_state=args.dopamine_state,
        dopamine_learning_rate=args.dopamine_learning_rate,
        visual_encoder=getattr(agent.config, "visual_encoder", "column_v1"),
        visual_ablation=getattr(args, "visual_ablation", "normal"),
        flyvis_model=getattr(args, "flyvis_model", None),
        flyvis_python=getattr(args, "flyvis_python", None),
        flyvis_worker=getattr(args, "flyvis_worker", None),
        spatial_readout=getattr(agent.config, "spatial_readout", False),
    )
    if connectome.feature_dim != agent.config.feature_dim:
        raise ValueError(
            f"checkpoint expects {agent.config.feature_dim} features, "
            f"MaleCNS provides {connectome.feature_dim}"
        )
    if getattr(args, "retina_view", False) and connectome.visual_encoder not in (
        "retina_lite_v2", "retina_spatial_v3"
    ):
        raise ValueError("retina view requires a retina_lite_v2 or retina_spatial_v3 checkpoint")
    if (
        getattr(args, "flyvis_view", False)
        and connectome.visual_encoder != "flyvis_spatial_v1"
    ):
        raise ValueError("FlyVis view requires a flyvis_spatial_v1 checkpoint")

    action_names = [action.name.lower() for action in FlybrainAction]
    # mario_sidecar.nes_actions and the checkpoint share the FlybrainAction order.
    if len(action_names) != agent.config.num_actions:
        raise ValueError("checkpoint action count does not match the Mario action space")

    env = JoypadSpace(gym.make(args.env, render_mode="rgb_array"), nes_actions())
    action_counts: Counter[int] = Counter()
    greedy_counts: Counter[int] = Counter()
    initial_action_counts: Counter[int] = Counter()
    fatal_terminal_action_counts: Counter[int] = Counter()
    q_sums = np.zeros(agent.config.num_actions, dtype=np.float64)
    q_observations = 0
    recurrent_health_sums: Counter[str] = Counter()
    q_margin_sum = 0.0
    state_rms_sum = 0.0
    state_delta_rms_sum = 0.0
    episodes: list[dict] = []
    first_actions: list[str] = []
    forensic_artifacts: list[dict] = []
    record_dir = getattr(args, "record_dir", None)
    record_episodes = int(getattr(args, "record_episodes", 0))
    record_fps = int(getattr(args, "record_fps", 60))
    retina_view = bool(getattr(args, "retina_view", False))
    flyvis_view = bool(getattr(args, "flyvis_view", False))
    visual_ablation = getattr(args, "visual_ablation", "normal")
    counterfactual_channels = bool(getattr(args, "counterfactual_channels", False))
    if counterfactual_channels and (algorithm != "ppo" or visual_ablation != "normal"):
        raise ValueError("paired channel replay requires normal PPO evaluation")

    try:
        for episode_index in range(args.episodes):
            observation, reset_info = env.reset(seed=args.seed + episode_index)
            episode_seed = args.seed + episode_index
            episode_recurrent = (
                algorithm == "ppo"
                and agent.config.temporal_encoder in (
                    "episode_gru_v1", "vision_fusion_gru_v2"
                )
            )
            if episode_recurrent:
                stack = (
                    VisionFusionGRUController(agent.config.feature_dim)
                    if agent.config.temporal_encoder == "vision_fusion_gru_v2"
                    else EpisodeGRUController(agent.config.feature_dim)
                )
                agent.reset_recurrent()
                current_tokens = stack.reset(
                    connectome.reset(observation, seed=episode_seed)
                )
                current_initial_hidden = agent.observe_recurrent(current_tokens)
                state = agent.recurrent_hidden()
            else:
                stack = (
                    ControllerTemporalStack(
                        agent.config.feature_dim, agent.config.stack_depth
                    )
                    if algorithm == "ppo"
                    and agent.config.temporal_encoder == "controller_gru"
                    else ActivityStack(agent.config.stack_depth)
                )
                state = stack.reset(
                    connectome.reset(observation, seed=episode_seed)
                )
                current_tokens = current_initial_hidden = None
            initial_frame = observation.copy() if counterfactual_channels and episode_index < record_episodes else None
            replay_intervals: list[tuple[int, list[np.ndarray]]] = []
            current_x = int(reset_info.get("x_pos", 0))
            components = {key: 0.0 for key in REWARD_COMPONENTS}
            raw_total = environment_training_total = legacy_total = 0.0
            max_x = 0
            stalled_frames = max_stalled_frames = 0
            terminated = truncated = False
            death = completion = False
            jump_press_edges = jump_release_edges = 0
            max_action_hold = max_jump_hold = 0
            initial_action: int | None = None
            terminal_action: int | None = None
            previous_jump = False
            previous_action: int | None = None
            jump_hold = action_hold = 0
            recorder = (
                ForensicRecorder(
                    record_dir, episode_index, record_fps,
                    retina_view=retina_view and visual_ablation == "normal",
                    flyvis_view=flyvis_view and visual_ablation == "normal",
                )
                if record_dir is not None and episode_index < record_episodes
                else None
            )

            for decision in range(1, args.max_decisions_per_episode + 1):
                visual_before = connectome.visual_stats()
                retina_before = connectome.retina_stats()
                policy_diagnostics: dict[str, object] = {}
                with torch.no_grad():
                    state_t = torch.as_tensor(state, device=agent.device).unsqueeze(0)
                    if algorithm == "ppo":
                        policy_diagnostics = (
                            agent.policy_diagnostics_recurrent()
                            if episode_recurrent
                            else agent.policy_diagnostics(state)
                        )
                        action_scores = policy_diagnostics["action_probabilities"]
                    else:
                        action_scores = agent.online(state_t)[0].detach().cpu().numpy()
                greedy_action = int(np.argmax(action_scores))
                if algorithm == "ppo":
                    if episode_recurrent:
                        action, log_probability, value = agent.act_recurrent(
                            deterministic=not args.sample_actions
                        )
                    else:
                        action, log_probability, value = agent.act(
                            state, deterministic=not args.sample_actions
                        )
                else:
                    action = agent.act(state, epsilon=args.epsilon)
                    log_probability = value = None
                jump_metrics = _jump_metrics(action_scores) if algorithm == "ppo" else None
                sensitivity = (
                    (
                        _ppo_recurrent_input_sensitivity(
                            agent, current_tokens, current_initial_hidden
                        )
                        if episode_recurrent
                        else _ppo_input_sensitivity(agent, state)
                    )
                    if algorithm == "ppo" and recorder is not None
                    else None
                )
                jump_pressed_edge = jump_released_edge = False
                if not bounded_actions:
                    jump_now = bool(ACTION_LEVELS[action][2])
                    jump_pressed_edge = jump_now and not previous_jump
                    jump_released_edge = previous_jump and not jump_now
                    jump_hold = jump_hold + 1 if jump_now else 0
                    action_hold = action_hold + 1 if action == previous_action else 1
                    jump_press_edges += int(jump_pressed_edge)
                    jump_release_edges += int(jump_released_edge)
                    max_action_hold = max(max_action_hold, action_hold)
                    max_jump_hold = max(max_jump_hold, jump_hold)
                    previous_jump = jump_now
                    previous_action = action
                q_sums += action_scores
                q_margin_sum += float(
                    np.partition(action_scores, -2)[-1]
                    - np.partition(action_scores, -2)[-2]
                )
                state_rms_sum += float(np.sqrt(np.mean(np.square(state))))
                q_observations += 1
                greedy_counts[greedy_action] += 1
                action_counts[action] += 1
                if initial_action is None:
                    initial_action = action
                    initial_action_counts[action] += 1
                terminal_action = action
                if len(first_actions) < args.first_actions:
                    first_actions.append(action_names[action])

                interval_raw = 0.0
                interval_components: dict[str, float] = {}
                interval_activities: list[np.ndarray] = []
                executed_actions: list[int] = []
                interval_press_edges = interval_release_edges = 0
                replay_frames: list[np.ndarray] = []
                x_before = current_x
                state_rms = float(np.sqrt(np.mean(np.square(state))))
                frame_actions = action_frames(action, bounded=bounded_actions, repeat=args.action_repeat)
                for frame_in_decision, executed_action in enumerate(frame_actions, start=1):
                    observation, reward, terminated, truncated, info = env.step(executed_action)
                    executed_actions.append(executed_action)
                    if bounded_actions:
                        jump_now = bool(ACTION_LEVELS[executed_action][2])
                        jump_pressed_edge = jump_now and not previous_jump
                        jump_released_edge = previous_jump and not jump_now
                        jump_hold = jump_hold + 1 if jump_now else 0
                        action_hold = action_hold + 1 if executed_action == previous_action else 1
                        jump_press_edges += int(jump_pressed_edge)
                        jump_release_edges += int(jump_released_edge)
                        interval_press_edges += int(jump_pressed_edge)
                        interval_release_edges += int(jump_released_edge)
                        max_action_hold = max(max_action_hold, action_hold)
                        max_jump_hold = max(max_jump_hold, jump_hold)
                        previous_jump, previous_action = jump_now, executed_action
                    interval_raw += float(reward)
                    current_x = int(info.get("x_pos", current_x))
                    stalled_frames = 0 if current_x > max_x else stalled_frames + 1
                    max_stalled_frames = max(max_stalled_frames, stalled_frames)
                    max_x = max(max_x, current_x)
                    for key, component_value in info.get(
                        "reward_components", {}
                    ).items():
                        if key in REWARD_COMPONENTS:
                            interval_components[key] = (
                                interval_components.get(key, 0.0)
                                + float(component_value)
                            )
                    retina_diagnostic = (
                        connectome.retina_diagnostic(observation)
                        if recorder is not None and recorder.retina_view
                        and (frame_in_decision == len(frame_actions) or terminated or truncated)
                        else None
                    )
                    activity = connectome.observe(
                        observation,
                        action_sequence=q_observations - 1,
                    )
                    if initial_frame is not None:
                        replay_frames.append(observation.copy())
                    if (
                        algorithm == "ppo"
                        and agent.config.temporal_encoder in (
                            "controller_gru", "episode_gru_v1", "vision_fusion_gru_v2"
                        )
                    ):
                        interval_activities.append(activity)
                    else:
                        stack.append(activity)
                    death |= bool(
                        info.get("death", False)
                        or info.get("is_dead", False)
                        or info.get("is_dying", False)
                    )
                    completion |= bool(
                        info.get("clear", False) or info.get("flag_get", False)
                    )
                    if recorder is not None:
                        frame_row = {
                            "episode": episode_index,
                            "decision": decision,
                            "frame_in_decision": frame_in_decision,
                            "action": action_names[action],
                            "executed_action": action_names[executed_action],
                            "hold_counter_units": "frames" if bounded_actions else "decisions",
                            "x": current_x,
                            "y": int(info.get("y_pos", 0)),
                            "reward": float(reward),
                            "interval_reward": interval_raw,
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            "death": bool(death),
                            "completion": bool(completion),
                            "visual_before_action": visual_before,
                            "retina_before_action": retina_before,
                            "visual_drive": connectome.visual_stats(),
                            "retina_measurements": connectome.retina_stats(),
                            "jump_metrics": jump_metrics,
                            "jump_pressed_edge": jump_pressed_edge,
                            "jump_released_edge": jump_released_edge,
                            "jump_hold_decisions": jump_hold,
                        }
                        recorder.add_frame(
                            observation,
                            frame_row,
                            [
                                (
                                    f"episode={episode_index} decision={decision} "
                                    f"frame={frame_in_decision}/{len(frame_actions)}"
                                ),
                                (
                                    f"action={action_names[executed_action]} x={current_x} "
                                    f"y={frame_row['y']} reward={interval_raw:+.1f}"
                                ),
                                (
                                    f"Pj={jump_metrics['total_jump_probability']:.2f} "
                                    f"best J/N={jump_metrics['best_jump_probability']:.2f}/"
                                    f"{jump_metrics['best_non_jump_probability']:.2f}"
                                    if jump_metrics is not None else "DQN policy"
                                ),
                                f"jump edge={int(jump_pressed_edge)}/{int(jump_released_edge)} hold={jump_hold}",
                                (
                                    f"pre R T/L/M={visual_before['threat_r']:.2f}/"
                                    f"{visual_before['loom_r']:.2f}/"
                                    f"{visual_before['small_motion_r']:.2f} "
                                    f"shift={retina_before.get('camera_shift_pixels', 0)}"
                                ),
                                (
                                    f"raw R motion/growth="
                                    f"{retina_before.get('lower_motion_r', 0):.2f}/"
                                    f"{retina_before.get('dark_growth_r', 0):.2f}"
                                ),
                                (
                                    "grad frames=" + ",".join(
                                        f"{item:.4f}" for item in
                                        sensitivity["jump_probability_gradient_rms_by_frame"]
                                    )
                                    if sensitivity is not None else ""
                                ),
                            ],
                        )
                        if retina_diagnostic is not None:
                            recorder.add_retina_frame(
                                observation, retina_diagnostic, frame_row
                            )
                        if (
                            recorder.flyvis_view
                            and (
                                frame_in_decision == len(frame_actions)
                                or terminated
                                or truncated
                            )
                        ):
                            flyvis_grids = connectome.flyvis_spatial_activity()
                            if flyvis_grids is None:
                                raise RuntimeError("FlyVis activity was not recorded")
                            recorder.add_flyvis_frame(
                                observation, flyvis_grids, frame_row
                            )
                    if terminated or truncated:
                        break
                if initial_frame is not None:
                    replay_intervals.append((action, replay_frames))

                for key, component_value in interval_components.items():
                    components[key] += component_value
                raw_total += interval_raw
                environment_training_total += training_reward(
                    interval_components, raw_reward=interval_raw
                )
                legacy_total += _legacy_training_reward(
                    interval_components, interval_raw
                )
                if episode_recurrent:
                    next_tokens = (
                        append_executed_frames(stack, interval_activities, executed_actions)
                        if bounded_actions else stack.append_interval(interval_activities, action)
                    )
                    if getattr(args, "reset_gru_every_decision", False):
                        agent.reset_recurrent()
                    next_initial_hidden = agent.observe_recurrent(next_tokens)
                    next_state = agent.recurrent_hidden()
                    controller_diagnostics = {
                        **dict(stack.last_diagnostics),
                        "hidden_rms": float(
                            np.sqrt(np.mean(np.square(next_state)))
                        ),
                        "hidden_max_abs": float(np.max(np.abs(next_state))),
                        "hidden_saturation_fraction": float(
                            np.mean(np.abs(next_state) > 0.95)
                        ),
                    }
                    if agent.config.temporal_encoder == "vision_fusion_gru_v2":
                        controller_diagnostics["fusion_vision_weight"] = float(
                            agent.network.fusion_vision_weight().item()
                        )
                    if agent.config.stabilized_vision_gru:
                        controller_diagnostics.update({
                            name: float(value.item())
                            for name, value in agent.network.temporal.gate_metrics.items()
                        })
                    for name in (
                        "hidden_rms", "hidden_saturation_fraction", "fusion_vision_weight",
                        "update_gate_mean", "update_gate_above_095_fraction",
                        "update_gate_below_005_fraction", "reset_gate_mean", "candidate_hidden_rms",
                    ):
                        if name in controller_diagnostics:
                            recurrent_health_sums[name] += controller_diagnostics[name]
                elif (
                    algorithm == "ppo"
                    and agent.config.temporal_encoder == "controller_gru"
                ):
                    next_state = stack.append_interval(interval_activities, action)
                    controller_diagnostics = dict(stack.last_diagnostics)
                else:
                    next_state = stack.state
                    controller_diagnostics = None
                state_delta_rms_sum += float(
                    np.sqrt(np.mean(np.square(next_state - state)))
                )
                if recorder is not None:
                    action_probabilities = [float(value) for value in action_scores]
                    probability_array = np.asarray(action_probabilities)
                    decision_row = {
                        "executed_actions": [action_names[a] for a in executed_actions],
                        "interval_jump_press_edges": interval_press_edges if bounded_actions else int(jump_pressed_edge),
                        "interval_jump_release_edges": interval_release_edges if bounded_actions else int(jump_released_edge),
                        "hold_counter_units": "frames" if bounded_actions else "decisions",
                        "episode": episode_index,
                        "decision": decision,
                        "x_before": x_before,
                        "x_after": current_x,
                        "max_x": max_x,
                        "action": action_names[action],
                        "greedy_action": action_names[greedy_action],
                        "selected_probability": (
                            float(action_scores[action]) if algorithm == "ppo" else None
                        ),
                        "log_probability": log_probability,
                        "critic_value": value,
                        "action_entropy": (
                            float(
                                -np.sum(
                                    probability_array
                                    * np.log(np.maximum(probability_array, 1.0e-12))
                                )
                            )
                            if algorithm == "ppo"
                            else None
                        ),
                        "action_probabilities": dict(
                            zip(action_names, action_probabilities, strict=True)
                        ),
                        "direction_probabilities": (
                            dict(
                                zip(
                                    ("neutral", "left", "right"),
                                    (
                                        float(item)
                                        for item in policy_diagnostics.get(
                                            "direction_probabilities", []
                                        )
                                    ),
                                    strict=True,
                                )
                            )
                            if "direction_probabilities" in policy_diagnostics
                            else None
                        ),
                        "jump_probabilities": (
                            dict(
                                zip(
                                    ("released", "pressed"),
                                    (
                                        float(item)
                                        for item in policy_diagnostics.get(
                                            "jump_probabilities", []
                                        )
                                    ),
                                    strict=True,
                                )
                            )
                            if "jump_probabilities" in policy_diagnostics
                            else None
                        ),
                        "run_probabilities": (
                            dict(
                                zip(
                                    ("off", "on"),
                                    (
                                        float(item)
                                        for item in policy_diagnostics.get(
                                            "run_probabilities", []
                                        )
                                    ),
                                    strict=True,
                                )
                            )
                            if "run_probabilities" in policy_diagnostics
                            else None
                        ),
                        "visual_drive": connectome.visual_stats(),
                        "visual_before_action": visual_before,
                        "retina_before_action": retina_before,
                        "retina_after_action": connectome.retina_stats(),
                        "jump_metrics": jump_metrics,
                        "jump_pressed_edge": bool(interval_press_edges) if bounded_actions else jump_pressed_edge,
                        "jump_released_edge": bool(interval_release_edges) if bounded_actions else jump_released_edge,
                        "jump_hold_decisions": jump_hold,
                        "action_hold_decisions": action_hold,
                        "ppo_input_sensitivity": sensitivity,
                        "state_rms": state_rms,
                        "state_delta_rms": float(
                            np.sqrt(np.mean(np.square(next_state - state)))
                        ),
                        "controller": controller_diagnostics,
                        "interval_reward": interval_raw,
                        "reward_components": interval_components,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "death": bool(death),
                        "completion": bool(completion),
                    }
                    recorder.add_decision(decision_row)
                state = next_state
                if episode_recurrent:
                    current_tokens = next_tokens
                    current_initial_hidden = next_initial_hidden
                if terminated or truncated:
                    break

            death |= components["death"] < 0.0
            completion |= components["completion"] > 0.0
            assert initial_action is not None and terminal_action is not None
            if death:
                fatal_terminal_action_counts[terminal_action] += 1
            row = {
                "episode": episode_index,
                "male_cns_seed": episode_seed,
                "decisions": decision,
                "raw_reward": raw_total,
                "legacy_training_reward": legacy_total,
                "environment_training_reward": environment_training_total,
                "max_x": max_x,
                "max_no_progress_frames": max_stalled_frames,
                "death": death,
                "completion": completion,
                "initial_action": action_names[initial_action],
                "terminal_action": action_names[terminal_action],
                "jump_press_edges": jump_press_edges,
                "jump_release_edges": jump_release_edges,
                "max_action_hold_decisions": max_action_hold,
                "max_jump_hold_decisions": max_jump_hold,
                "environment_terminated": bool(terminated),
                "environment_truncated": bool(truncated),
                "evaluator_truncated": not (terminated or truncated),
                "components": components,
            }
            episodes.append(row)
            if recorder is not None:
                if initial_frame is not None:
                    baseline_error = _replay_counterfactuals(
                        agent, connectome, initial_frame, replay_intervals,
                        recorder.decision_rows, episode_seed=episode_seed,
                    )
                    row["counterfactual_baseline_max_probability_error"] = baseline_error
                    row["counterfactual_replay_valid"] = baseline_error < 1.0e-4
                    row["counterfactual_summary"] = _counterfactual_summary(
                        recorder.decision_rows
                    )
                    if not row["counterfactual_replay_valid"]:
                        print(
                            f"WARNING: episode={episode_index} paired replay differs "
                            f"from original policy by {baseline_error:.6f}; "
                            "do not interpret its counterfactuals",
                            flush=True,
                        )
                    recorder.annotate_counterfactuals(
                        valid=row["counterfactual_replay_valid"]
                    )
                forensic_artifacts.append(
                    {"episode": episode_index, **recorder.close()}
                )
            print(
                f"episode={episode_index} decisions={decision} max_x={max_x} "
                f"death={int(death)} completion={int(completion)} "
                f"raw={raw_total:+.1f} legacy={legacy_total:+.3f} "
                f"environment={environment_training_total:+.3f}",
                flush=True,
            )
    finally:
        env.close()
        connectome.close()

    decision_count = max(1, sum(action_counts.values()))
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dopamine_state": (
            None if args.dopamine_state is None else str(args.dopamine_state.resolve())
        ),
        "learning_enabled": False,
        "dopamine_updates_enabled": False,
        "algorithm": algorithm,
        "visual_ablation": visual_ablation,
        "visual_encoder": getattr(agent.config, "visual_encoder", "column_v1"),
        "temporal_encoder": getattr(agent.config, "temporal_encoder", "conv4"),
        "stabilized_vision_gru": getattr(agent.config, "stabilized_vision_gru", False),
        "spatial_readout": getattr(agent.config, "spatial_readout", False),
        "bounded_jump_actions": bounded_actions,
        "frames_per_decision": MACRO_FRAMES if bounded_actions else args.action_repeat,
        "hold_counter_units": "frames" if bounded_actions else "decisions",
        "flyvis_frame_dt": getattr(connectome, "flyvis_frame_dt", None),
        "reset_gru_every_decision": bool(
            getattr(args, "reset_gru_every_decision", False)
        ),
        "temporal_depth": agent.config.stack_depth,
        "controller_feature_dim": getattr(
            agent.config, "controller_feature_dim", 0
        ),
        "mode": "sampled" if args.sample_actions else "mean",
        "epsilon": args.epsilon,
        "seed": args.seed,
        "male_cns_episode_seed_rule": "evaluation_seed + episode_index",
        "episodes": episodes,
        "forensic_artifacts": forensic_artifacts,
        "summary": {
            **{f"mean_{name}": value / decision_count for name, value in recurrent_health_sums.items()},
            "episode_count": len(episodes),
            "death_rate": float(np.mean([row["death"] for row in episodes])),
            "completion_rate": float(
                np.mean([row["completion"] for row in episodes])
            ),
            "mean_decisions": float(
                np.mean([row["decisions"] for row in episodes])
            ),
            "mean_max_x": float(np.mean([row["max_x"] for row in episodes])),
            "mean_max_no_progress_frames": float(np.mean([row["max_no_progress_frames"] for row in episodes])),
            "mean_raw_reward": float(
                np.mean([row["raw_reward"] for row in episodes])
            ),
            "mean_legacy_training_reward": float(
                np.mean([row["legacy_training_reward"] for row in episodes])
            ),
            "mean_environment_training_reward": float(
                np.mean([row["environment_training_reward"] for row in episodes])
            ),
            "mean_greedy_action_score_margin": q_margin_sum / max(1, q_observations),
            "mean_state_rms": state_rms_sum / max(1, q_observations),
            "mean_temporal_state_delta_rms": (
                state_delta_rms_sum / max(1, q_observations)
            ),
            "mean_jump_press_edges": float(
                np.mean([row["jump_press_edges"] for row in episodes])
            ),
            "mean_jump_release_edges": float(
                np.mean([row["jump_release_edges"] for row in episodes])
            ),
            "mean_max_action_hold_decisions": float(
                np.mean([row["max_action_hold_decisions"] for row in episodes])
            ),
            "mean_max_jump_hold_decisions": float(
                np.mean([row["max_jump_hold_decisions"] for row in episodes])
            ),
            "actions": {
                action_names[index]: {
                    "selected": action_counts[index],
                    "selected_fraction": action_counts[index] / decision_count,
                    "greedy_argmax": greedy_counts[index],
                    "mean_action_score": float(
                        q_sums[index] / max(1, q_observations)
                    ),
                }
                for index in range(agent.config.num_actions)
            },
            "initial_action_counts": {
                action_names[index]: initial_action_counts[index]
                for index in range(agent.config.num_actions)
            },
            "fatal_terminal_action_counts": {
                action_names[index]: fatal_terminal_action_counts[index]
                for index in range(agent.config.num_actions)
            },
            "first_actions": first_actions,
        },
    }
    if args.output is not None:
        _atomic_json(args.output, report)
        print(f"report={args.output}", flush=True)
    print(json.dumps(report["summary"], indent=2, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dopamine-state", type=Path)
    parser.add_argument("--male-cns-data", type=Path)
    parser.add_argument(
        "--flyvis-model",
        type=Path,
        help="pretrained FlyVis model directory required by flyvis_spatial_v1",
    )
    parser.add_argument("--flyvis-python", type=Path)
    parser.add_argument("--flyvis-worker", type=Path)
    parser.add_argument(
        "--male-cns-device", choices=("auto", "cpu", "cuda"), default="cpu"
    )
    parser.add_argument("--dopamine-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--env", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-decisions-per-episode", type=int, default=500)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument(
        "--sample-actions",
        action="store_true",
        help="sample a PPO policy instead of using its deterministic argmax",
    )
    parser.add_argument(
        "--reset-gru-every-decision",
        action="store_true",
        help="evaluation ablation: erase episode GRU memory before each action interval",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--first-actions", type=int, default=50)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--record-dir",
        type=Path,
        help="write annotated GIFs plus synchronized frame/decision JSONL traces",
    )
    parser.add_argument(
        "--record-episodes",
        type=int,
        default=0,
        help="record this many leading episodes (requires --record-dir)",
    )
    parser.add_argument("--record-fps", type=int, default=60)
    parser.add_argument(
        "--retina-view", action="store_true",
        help="record per-decision retina source/contributor GIF and channel timeline",
    )
    parser.add_argument(
        "--flyvis-view", action="store_true",
        help="record per-decision signed spatial FlyVis population maps",
    )
    parser.add_argument(
        "--visual-ablation", choices=VISUAL_ABLATIONS, default="normal",
        help="silence the named injected visual population throughout a frozen rollout",
    )
    parser.add_argument(
        "--counterfactual-channels", action="store_true",
        help="replay recorded frames through each masked MaleCNS channel using the original actions",
    )
    parser.add_argument(
        "--allow-legacy-reward-contract",
        action="store_true",
        help="evaluation only: load a schema-4 checkpoint without permitting training resume",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    args = parser.parse_args()
    if (
        args.episodes <= 0
        or args.max_decisions_per_episode <= 0
        or args.action_repeat <= 0
        or args.first_actions < 0
        or args.record_episodes < 0
        or args.record_fps <= 0
        or not 0.0 <= args.epsilon <= 1.0
    ):
        parser.error("episode/decision/action counts must be positive and epsilon in [0, 1]")
    if args.record_episodes and args.record_dir is None:
        parser.error("--record-episodes requires --record-dir")
    if args.counterfactual_channels and not args.record_episodes:
        parser.error("--counterfactual-channels requires recorded episodes")
    if args.retina_view and not args.record_episodes:
        parser.error("--retina-view requires recorded episodes")
    if args.flyvis_view and not args.record_episodes:
        parser.error("--flyvis-view requires recorded episodes")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    temporal_encoder = payload.get("config", {}).get("temporal_encoder", "conv4")
    if args.reset_gru_every_decision and temporal_encoder not in (
        "episode_gru_v1", "vision_fusion_gru_v2"
    ):
        parser.error("--reset-gru-every-decision requires an episode recurrent checkpoint")
    run(args)


if __name__ == "__main__":
    main()
