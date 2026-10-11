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
VISUAL_ENCODERS = ("column_v1", "retina_lite_v2")


def _top_mean(values: np.ndarray, fraction: float = 0.1) -> float:
    """Mean the strongest local responses without diluting small sprites."""

    flat = np.asarray(values, dtype=np.float32).ravel()
    if not len(flat):
        return 0.0
    count = max(1, int(round(len(flat) * fraction)))
    return float(np.partition(flat, len(flat) - count)[-count:].mean())


def _align_previous(gray: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Compensate small horizontal camera scroll before measuring local motion."""

    if gray.shape != previous.shape:
        return previous
    height, width = gray.shape
    top = max(0, int(round(height * 0.12)))
    best_error = float("inf")
    best = previous
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
            best = np.roll(previous, shift, axis=1)
            if shift < 0:
                best[:, shift:] = previous[:, -1:]
            elif shift > 0:
                best[:, :shift] = previous[:, :1]
    return best


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
    ) -> None:
        if visual_encoder not in VISUAL_ENCODERS:
            raise ValueError(f"visual encoder must be one of {VISUAL_ENCODERS}")
        self.visual_encoder = visual_encoder
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

        if self.visual_encoder == "retina_lite_v2":
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

        aligned = _align_previous(gray, previous)
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
        for side, horizontal in (
            ("L", slice(0, midpoint)),
            ("R", slice(midpoint, width)),
        ):
            lower_motion = _top_mean(motion[bands["lower"], horizontal], 0.08)
            middle_motion = _top_mean(motion[bands["middle"], horizontal], 0.08)
            growth = _top_mean(dark_growth[bands["lower"], horizontal], 0.08)
            ground_motion = _top_mean(motion[bands["ground"], horizontal], 0.06)
            local_edges = _top_mean(gradient_x[bands["lower"], horizontal], 0.08)
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

    def _encode(self, frame: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, float]]]:
        gray = self._gray(frame)
        previous = gray if self.previous_gray is None else self.previous_gray
        eye_drive = self._eye_drive(gray, previous)

        inject: list[tuple[np.ndarray, float]] = []
        if self.visual_encoder == "retina_lite_v2":
            encoded = self._retina_lite_amounts(gray, previous)
        else:
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
                self.last_visual_drive[f"{name}_{side.lower()}"] = amount
                if amount > 0.0:
                    inject.append((self.visual[name][side], amount))
        self.previous_gray = gray
        return eye_drive.astype(np.float32), inject

    def visual_stats(self) -> dict[str, float]:
        return dict(self.last_visual_drive)

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

    def reset(self, frame: np.ndarray) -> np.ndarray:
        self.brain.reset()
        self.trace.reset()
        self.previous_gray = None
        if self.dopamine is not None:
            self.dopamine.reset_episode()
        return self.observe(frame)
