"""Real MaleCNS v1.0 connectome backend for Mario visual decisions.

Pixels stimulate identified visual neurons in the connectome. The resulting
descending-neuron spike trace is exposed to the DQN, while optional dopamine
plasticity modifies anatomically identified KC->MBON synapses.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np


VISUAL_CHANNELS = {
    "loom": ["LPLC2"],
    "threat": ["LC4"],
    "small_motion": ["LPLC1"],
    "target": ["LC10a"],
}
VISUAL_ENCODERS = (
    "column_v1",
    "retina_lite_v2",
    "retina_spatial_v3",
    "flyvis_spatial_v1",
)
VISUAL_ABLATIONS = (
    "normal", "threat", "loom", "small_motion", "target", "all", "eye",
    "all_visual", "flyvis_static", "flyvis_motion",
)


def _top_mean(values: np.ndarray, fraction: float = 0.1) -> float:
    """Mean the strongest local responses without diluting small sprites."""

    flat = np.asarray(values, dtype=np.float32).ravel()
    if not len(flat):
        return 0.0
    count = max(1, int(round(len(flat) * fraction)))
    return float(np.partition(flat, len(flat) - count)[-count:].mean())


def _top_pixels(values: np.ndarray, fraction: float) -> np.ndarray:
    """Retain the same strongest pixels selected by ``_top_mean`` for display."""

    flat = np.asarray(values, dtype=np.float32).ravel()
    selected = np.zeros_like(flat)
    if len(flat):
        count = max(1, int(round(len(flat) * fraction)))
        indices = np.argpartition(flat, len(flat) - count)[-count:]
        selected[indices] = flat[indices]
    return selected.reshape(values.shape)


def _align_previous_with_shift(
    gray: np.ndarray, previous: np.ndarray
) -> tuple[np.ndarray, int]:
    """Compensate small horizontal camera scroll before measuring local motion."""

    if gray.shape != previous.shape:
        return previous, 0
    height, width = gray.shape
    top = max(0, int(round(height * 0.12)))
    best_error = float("inf")
    best = previous
    best_shift = 0
    for shift in range(-4, 5):
        if shift < 0:
            current_view = gray[top:, : width + shift]
            previous_view = previous[top:, -shift:]
        elif shift > 0:
            current_view = gray[top:, shift:]
            previous_view = previous[top:, : width - shift]
        else:
            current_view = gray[top:]
            previous_view = previous[top:]
        error = float(np.mean(np.abs(current_view - previous_view)))
        if error < best_error:
            best_error = error
            best_shift = shift
            best = np.roll(previous, shift, axis=1)
            if shift < 0:
                best[:, shift:] = previous[:, -1:]
            elif shift > 0:
                best[:, :shift] = previous[:, :1]
    return best, best_shift


def _align_previous(gray: np.ndarray, previous: np.ndarray) -> np.ndarray:
    return _align_previous_with_shift(gray, previous)[0]


class MaleCNS:
    """Drive the published MaleCNS LIF model and expose descending activity."""

    def __init__(
        self,
        *,
        data: Path | None = None,
        device: str = "auto",
        seed: int = 123,
        trace_tau: float = 0.12,
        spike_file: Path | None = None,
        dopamine_state: Path | None = None,
        dopamine_learning_rate: float = 1.0e-5,
        visual_encoder: str = "column_v1",
        visual_ablation: str = "normal",
        flyvis_model: Path | None = None,
        flyvis_python: Path | None = None,
        flyvis_worker: Path | None = None,
    ) -> None:
        if visual_encoder not in VISUAL_ENCODERS:
            raise ValueError(f"visual encoder must be one of {VISUAL_ENCODERS}")
        self.visual_encoder = visual_encoder
        if visual_ablation not in VISUAL_ABLATIONS:
            raise ValueError(f"visual ablation must be one of {VISUAL_ABLATIONS}")
        self.visual_ablation = visual_ablation
        try:
            from flybrain import FlyBrain, Trace
        except ImportError as exc:  # pragma: no cover - exercised in cluster env
            raise RuntimeError(
                "The real MaleCNS backend is missing. Install the super_mario "
                "sidecar dependencies; do not substitute the old pixel-only DQN."
            ) from exc

        kwargs = {"device": device, "seed": seed}
        if data is not None:
            kwargs["data"] = data
        self.brain = FlyBrain(**kwargs)
        self.dopamine = None
        if dopamine_state is not None:
            from dopamine import DopaminePlasticity

            self.dopamine = DopaminePlasticity(
                self.brain,
                state_path=dopamine_state,
                learning_rate=dopamine_learning_rate,
            )
        self.descending = self.brain.cells(["descending_neuron"])
        if not len(self.descending):
            raise RuntimeError("MaleCNS metadata contains no descending neurons")
        self.trace = Trace(self.brain, idx=self.descending, tau=trace_tau)
        self.visual = {
            name: {
                side: self.brain.cells(cell_types, side=side)
                for side in ("L", "R")
            }
            for name, cell_types in VISUAL_CHANNELS.items()
        }
        missing = [
            f"{name}:{side}"
            for name, sides in self.visual.items()
            for side, indices in sides.items()
            if not len(indices)
        ]
        if missing:
            raise RuntimeError(f"MaleCNS visual cell types are missing: {missing}")
        self.previous_gray: np.ndarray | None = None
        self.last_visual_drive = {
            f"{name}_{side.lower()}": 0.0
            for name in VISUAL_CHANNELS
            for side in ("L", "R")
        }
        self.last_retina_measurements: dict[str, float | int] = {}
        self.flyvis_bridge = None
        if self.visual_encoder == "flyvis_spatial_v1":
            if flyvis_model is None:
                raise ValueError("flyvis_spatial_v1 requires a FlyVis model directory")
            if flyvis_python is None or flyvis_worker is None:
                raise ValueError(
                    "flyvis_spatial_v1 requires FlyVis Python and worker paths"
                )
            from flyvis_spatial_bridge import FlyVisSpatialBridge

            self.flyvis_bridge = FlyVisSpatialBridge(
                model_dir=flyvis_model,
                brain=self.brain,
                dt=float(self.brain.dt),
                python=flyvis_python,
                worker_script=flyvis_worker,
            )
        self.spike_file = Path(spike_file) if spike_file is not None else None
        if self.spike_file is not None:
            self.spike_file.parent.mkdir(parents=True, exist_ok=True)

    @property
    def feature_dim(self) -> int:
        return int(len(self.descending))

    @property
    def neuron_count(self) -> int:
        return int(self.brain.n)

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        rgb = np.asarray(frame)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError("Mario frame must have shape (height, width, 3 or 4)")
        rgb = np.ascontiguousarray(rgb[:, :, :3], dtype=np.float32) / 255.0
        return rgb @ np.asarray((0.299, 0.587, 0.114), dtype=np.float32)

    def _eye_drive(self, gray: np.ndarray, previous: np.ndarray) -> np.ndarray:
        """Project a horizontal panorama onto MaleCNS photoreceptor azimuths."""

        if self.visual_encoder in ("retina_lite_v2", "retina_spatial_v3"):
            height = gray.shape[0]
            vertical = np.linspace(0.35, 1.65, height, dtype=np.float32)
            vertical[: int(round(height * 0.12))] = 0.0
            vertical /= max(float(vertical.sum()), 1.0e-6)
            columns = (gray * vertical[:, None]).sum(axis=0)
            old_columns = (previous * vertical[:, None]).sum(axis=0)
        else:
            columns = gray.mean(axis=0)
            old_columns = previous.mean(axis=0)
        azimuth = np.asarray(self.brain.azimuth, dtype=np.float32)
        valid_eye = np.isfinite(azimuth)
        x = (azimuth[valid_eye] + 1.0) * 0.5
        source_x = np.linspace(0.0, 1.0, len(columns), dtype=np.float32)
        luminance = np.interp(x, source_x, columns)
        old_luminance = np.interp(x, source_x, old_columns)
        eye_drive = np.zeros_like(azimuth)
        eye_drive[valid_eye] = np.clip(
            0.35 * (1.0 - luminance)
            + 1.6 * np.abs(luminance - old_luminance),
            0,
            1,
        )
        return eye_drive.astype(np.float32)

    def _retina_lite_amounts(
        self, gray: np.ndarray, previous: np.ndarray
    ) -> dict[tuple[str, str], float]:
        """Fixed 2-D motion filters feeding biologically named visual channels."""

        aligned, camera_shift = _align_previous_with_shift(gray, previous)
        motion = np.abs(gray - aligned)
        dark = 1.0 - gray
        old_dark = 1.0 - aligned
        dark_growth = np.maximum(dark - old_dark, 0.0)
        gradient_x = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        height, width = gray.shape
        midpoint = width // 2
        bands = {
            "middle": slice(int(height * 0.30), int(height * 0.72)),
            "lower": slice(int(height * 0.58), int(height * 0.92)),
            "ground": slice(int(height * 0.76), int(height * 0.98)),
        }
        amounts: dict[tuple[str, str], float] = {}
        self.last_retina_measurements = {"camera_shift_pixels": camera_shift}
        for side, horizontal in (
            ("L", slice(0, midpoint)),
            ("R", slice(midpoint, width)),
        ):
            lower_motion = _top_mean(motion[bands["lower"], horizontal], 0.08)
            middle_motion = _top_mean(motion[bands["middle"], horizontal], 0.08)
            growth = _top_mean(dark_growth[bands["lower"], horizontal], 0.08)
            ground_motion = _top_mean(motion[bands["ground"], horizontal], 0.06)
            local_edges = _top_mean(gradient_x[bands["lower"], horizontal], 0.08)
            suffix = side.lower()
            self.last_retina_measurements.update(
                {
                    f"lower_motion_{suffix}": lower_motion,
                    f"middle_motion_{suffix}": middle_motion,
                    f"dark_growth_{suffix}": growth,
                    f"ground_motion_{suffix}": ground_motion,
                    f"local_edges_{suffix}": local_edges,
                }
            )
            amounts[("small_motion", side)] = min(
                0.8, 0.15 * lower_motion + 0.85 * middle_motion
            )
            amounts[("loom", side)] = min(
                0.8, 0.65 * growth + 0.15 * lower_motion
            )
            amounts[("threat", side)] = min(
                0.8, 0.55 * ground_motion + 0.25 * growth
            )
            amounts[("target", side)] = min(
                0.8, 0.04 + 0.18 * local_edges + 0.18 * middle_motion
            )
        return amounts

    @staticmethod
    def _spatial_retina(frame: np.ndarray) -> dict[str, np.ndarray | float | int]:
        """Mario-specific lower-field geometry, measured ahead of the sprite.

        Brick-colored pixels near the bottom indicate traversable floor; green
        vertical spans indicate pipe-like obstacles. These are deliberately
        simple palette filters, not semantic object recognition.
        """

        rgb = np.asarray(frame, dtype=np.uint8)[..., :3]
        height, width = rgb.shape[:2]
        red, green, blue = (rgb[..., channel].astype(np.int16) for channel in range(3))
        # Mario's bright-red clothing is distinct from the darker brick tiles.
        mario_red = (red > 235) & (green < 85) & (blue < 45)
        player_rows = slice(int(height * 0.20), int(height * 0.86))
        red_columns = mario_red[player_rows].sum(axis=0)
        window = np.convolve(red_columns, np.ones(12), mode="same")
        player_x = int(np.argmax(window)) if window.max() >= 4 else width // 2

        floor_rows = slice(int(height * 0.88), int(height * 0.97))
        brick = (red > 160) & (red > green + 30) & (blue < 200)
        ground = np.clip(brick[floor_rows].mean(axis=0) / 0.35, 0.0, 1.0).astype(np.float32)

        lower_rows = slice(int(height * 0.40), int(height * 0.87))
        green_solid = (green > 115) & (green > red + 30) & (green > blue + 25)
        vertical_green = green_solid[lower_rows].sum(axis=0)
        green_height = vertical_green / max(1.0, height * 0.30)
        obstacle = np.clip((green_height - 0.25) / 0.75, 0.0, 1.0).astype(np.float32)

        distances = (("near", 4, 20), ("mid", 20, 44), ("far", 44, 76))
        metrics: dict[str, float | int] = {"player_screen_x": player_x}
        for name, start, end in distances:
            section = slice(min(width, player_x + start), min(width, player_x + end))
            metrics[f"ground_{name}"] = (
                float(ground[section].mean()) if section.start < section.stop else 0.0
            )
            metrics[f"gap_{name}"] = (
                float(1.0 - ground[section].mean()) if section.start < section.stop else 0.0
            )
            metrics[f"obstacle_{name}"] = (
                float(obstacle[section].max()) if section.start < section.stop else 0.0
            )
        return {"ground_profile": ground, "obstacle_profile": obstacle, **metrics}

    def retina_diagnostic(self, frame: np.ndarray) -> dict[str, np.ndarray | int] | None:
        """Recompute display-only source and contributing-pixel maps, without state writes.

        Call before ``observe(frame)`` so ``previous_gray`` is the same frame used
        by the live encoder. The four channel maps show the exact top-pixel
        regions pooled on each side, before a channel ablation is applied.
        """

        if self.visual_encoder not in ("retina_lite_v2", "retina_spatial_v3"):
            return None
        gray = self._gray(frame)
        previous = gray if self.previous_gray is None else self.previous_gray
        aligned, shift = _align_previous_with_shift(gray, previous)
        motion = np.abs(gray - aligned)
        growth = np.maximum(aligned - gray, 0.0)
        edges = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        height, width = gray.shape
        bands = {
            "middle": slice(int(height * 0.30), int(height * 0.72)),
            "lower": slice(int(height * 0.58), int(height * 0.92)),
            "ground": slice(int(height * 0.76), int(height * 0.98)),
        }
        channels = {
            name: np.zeros_like(gray) for name in VISUAL_CHANNELS
        }
        for horizontal in (slice(0, width // 2), slice(width // 2, width)):
            def selected(source: np.ndarray, band: str, fraction: float) -> np.ndarray:
                result = np.zeros_like(gray)
                region = (bands[band], horizontal)
                result[region] = _top_pixels(source[region], fraction)
                return result

            lower_motion = selected(motion, "lower", 0.08)
            middle_motion = selected(motion, "middle", 0.08)
            ground_motion = selected(motion, "ground", 0.06)
            lower_growth = selected(growth, "lower", 0.08)
            lower_edges = selected(edges, "lower", 0.08)
            channels["small_motion"] += 0.15 * lower_motion + 0.85 * middle_motion
            channels["loom"] += 0.65 * lower_growth + 0.15 * lower_motion
            channels["threat"] += 0.55 * ground_motion + 0.25 * lower_growth
            channels["target"] += 0.18 * lower_edges + 0.18 * middle_motion
        diagnostic = {
            "motion": motion,
            "dark_growth": growth,
            "edges": edges,
            **channels,
            "camera_shift_pixels": shift,
        }
        if self.visual_encoder == "retina_spatial_v3":
            spatial = self._spatial_retina(frame)
            ground_map = np.zeros_like(gray)
            obstacle_map = np.zeros_like(gray)
            ground_map[int(height * 0.88):int(height * 0.97)] = spatial["ground_profile"]
            obstacle_map[int(height * 0.40):int(height * 0.87)] = spatial["obstacle_profile"]
            diagnostic["ground_presence"] = ground_map
            diagnostic["obstacle_height"] = obstacle_map
        return diagnostic

    def _encode(self, frame: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, float]]]:
        gray = self._gray(frame)
        previous = gray if self.previous_gray is None else self.previous_gray
        if self.visual_encoder == "flyvis_spatial_v1":
            if self.flyvis_bridge is None:  # defensive: constructor owns this contract
                raise RuntimeError("FlyVis spatial bridge was not initialized")
            eye_drive = np.zeros_like(self.brain.azimuth, dtype=np.float32)
            inject = self.flyvis_bridge.encode(frame, ablation=self.visual_ablation)
            self.last_retina_measurements = dict(self.flyvis_bridge.last_stats)
            self.last_visual_drive = {
                key: 0.0 for key in self.last_visual_drive
            }
            self.previous_gray = gray
            return eye_drive, inject
        eye_drive = self._eye_drive(gray, previous)
        spatial = None
        if self.visual_encoder == "retina_spatial_v3":
            spatial = self._spatial_retina(frame)
            azimuth = np.asarray(self.brain.azimuth, dtype=np.float32)
            valid_eye = np.isfinite(azimuth)
            screen_x = (azimuth[valid_eye] + 1.0) * 0.5 * (gray.shape[1] - 1)
            columns = np.arange(gray.shape[1], dtype=np.float32)
            forward = (
                (screen_x >= spatial["player_screen_x"] + 4)
                & (screen_x < spatial["player_screen_x"] + 76)
            )
            obstacle_drive = forward * np.interp(
                screen_x, columns, spatial["obstacle_profile"]
            )
            gap_drive = forward * (
                1.0 - np.interp(screen_x, columns, spatial["ground_profile"])
            )
            eye_drive[valid_eye] = np.clip(
                eye_drive[valid_eye] + 0.45 * obstacle_drive - 0.20 * gap_drive,
                0.0, 1.0,
            )
        if self.visual_ablation in ("eye", "all_visual"):
            eye_drive.fill(0.0)

        inject: list[tuple[np.ndarray, float]] = []
        if self.visual_encoder in ("retina_lite_v2", "retina_spatial_v3"):
            encoded = self._retina_lite_amounts(gray, previous)
            if spatial is not None:
                encoded[("threat", "R")] = min(
                    0.8, encoded[("threat", "R")]
                    + 0.32 * spatial["gap_near"]
                    + 0.16 * spatial["gap_mid"]
                    + 0.08 * spatial["gap_far"]
                )
                encoded[("loom", "R")] = min(
                    0.8, encoded[("loom", "R")]
                    + 0.28 * spatial["obstacle_near"]
                    + 0.14 * spatial["obstacle_mid"]
                    + 0.07 * spatial["obstacle_far"]
                )
                self.last_retina_measurements.update(
                    {key: value for key, value in spatial.items()
                     if key not in ("ground_profile", "obstacle_profile")}
                )
        else:
            self.last_retina_measurements = {}
            motion = np.abs(gray - previous)
            dark = 1.0 - gray
            midpoint = gray.shape[1] // 2
            encoded = {}
            for side, section in (
                ("L", slice(0, midpoint)),
                ("R", slice(midpoint, None)),
            ):
                side_motion = float(motion[:, section].mean())
                side_dark = float(dark[:, section].mean())
                growth = max(
                    0.0,
                    float(
                        dark[:, section].mean()
                        - (1.0 - previous[:, section]).mean()
                    ),
                )
                encoded[("loom", side)] = min(
                    0.8, growth * 8.0 + side_motion * 1.5
                )
                encoded[("threat", side)] = min(0.8, side_motion * 2.5)
                encoded[("small_motion", side)] = min(0.8, side_motion * 4.0)
                encoded[("target", side)] = min(0.8, 0.08 + side_dark * 0.35)
        for side in ("L", "R"):
            for name in VISUAL_CHANNELS:
                amount = encoded[(name, side)]
                if self.visual_ablation in (name, "all", "all_visual"):
                    amount = 0.0
                self.last_visual_drive[f"{name}_{side.lower()}"] = amount
                if amount > 0.0:
                    inject.append((self.visual[name][side], amount))
        self.previous_gray = gray
        return eye_drive.astype(np.float32), inject

    def visual_stats(self) -> dict[str, float]:
        return dict(self.last_visual_drive)

    def retina_stats(self) -> dict[str, float | int]:
        return dict(self.last_retina_measurements)

    def flyvis_spatial_activity(self) -> np.ndarray | None:
        """Return the most recent signed FlyVis maps for evaluation displays."""

        bridge = getattr(self, "flyvis_bridge", None)
        grids = None if bridge is None else bridge.last_grids
        return None if grids is None else grids.copy()

    def observe(self, frame: np.ndarray, *, action_sequence: int = -1) -> np.ndarray:
        eye_drive, inject = self._encode(frame)
        if self.dopamine is not None:
            inject.extend(self.dopamine.consume_injection())
        fired = self.brain.step(eye_drive=eye_drive, inject=inject)
        if self.dopamine is not None:
            self.dopamine.observe(fired)
        activity = self.trace.observe(fired).astype(np.float32, copy=False)
        if self.spike_file is not None:
            fired = np.asarray(fired, dtype=np.int64)
            mask = np.isin(fired, self.descending, assume_unique=False)
            rows = [{
                "time_s": time.time(),
                "brain_step": int(self.brain.steps),
                "population": "MaleCNS descending_neuron",
                "neuron_ids": fired[mask].tolist(),
                "all_spikes": int(len(fired)),
                "action_sequence": int(action_sequence),
            }]
            if self.dopamine is not None:
                for name, neurons in (("PAM", self.dopamine.pam), ("PPL1", self.dopamine.ppl1)):
                    dan_mask = np.isin(fired, neurons, assume_unique=False)
                    if np.any(dan_mask):
                        rows.append(
                            {
                                "time_s": rows[0]["time_s"],
                                "brain_step": int(self.brain.steps),
                                "population": f"MaleCNS {name} dopamine",
                                "neuron_ids": fired[dan_mask].tolist(),
                                "all_spikes": int(len(fired)),
                                "action_sequence": int(action_sequence),
                                "prediction_error": self.dopamine.last_signal,
                            }
                        )
            with self.spike_file.open("a", encoding="utf-8") as output:
                for row in rows:
                    output.write(json.dumps(row, separators=(",", ":")) + "\n")
        return activity.copy()

    def reinforce(self, prediction_error: float) -> float | None:
        """Apply one signed reward-prediction error to the mushroom body."""

        if self.dopamine is None:
            return None
        return self.dopamine.reinforce(prediction_error)

    def dopamine_stats(self) -> dict[str, float | int] | None:
        return None if self.dopamine is None else self.dopamine.stats()

    def save_plasticity(self) -> None:
        if self.dopamine is not None:
            self.dopamine.save()

    def close(self) -> None:
        """Release optional out-of-process visual encoder resources."""

        flyvis_bridge = getattr(self, "flyvis_bridge", None)
        if flyvis_bridge is not None:
            flyvis_bridge.close()
            self.flyvis_bridge = None

    def reset(self, frame: np.ndarray, *, seed: int | None = None) -> np.ndarray:
        if seed is None:
            self.brain.reset()
        else:
            self.brain.reset(seed=seed)
        self.trace.reset()
        self.previous_gray = None
        flyvis_bridge = getattr(self, "flyvis_bridge", None)
        if flyvis_bridge is not None:
            flyvis_bridge.reset()
        if self.dopamine is not None:
            self.dopamine.reset_episode()
        return self.observe(frame)
