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
        dopamine_learning_rate: float = 0.02,
    ) -> None:
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

    def _encode(self, frame: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, float]]]:
        gray = self._gray(frame)
        previous = gray if self.previous_gray is None else self.previous_gray
        motion = np.abs(gray - previous)
        dark = 1.0 - gray

        # The compound-eye input is a one-dimensional azimuth panorama.
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

        inject: list[tuple[np.ndarray, float]] = []
        midpoint = gray.shape[1] // 2
        for side, section in (("L", slice(0, midpoint)), ("R", slice(midpoint, None))):
            side_motion = float(motion[:, section].mean())
            side_dark = float(dark[:, section].mean())
            growth = max(0.0, float(dark[:, section].mean() - (1.0 - previous[:, section]).mean()))
            amounts = {
                "loom": min(0.8, growth * 8.0 + side_motion * 1.5),
                "threat": min(0.8, side_motion * 2.5),
                "small_motion": min(0.8, side_motion * 4.0),
                "target": min(0.8, 0.08 + side_dark * 0.35),
            }
            inject.extend(
                (self.visual[name][side], amount)
                for name, amount in amounts.items()
                if amount > 0.0
            )
        self.previous_gray = gray
        return eye_drive.astype(np.float32), inject

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
