"""Reward-modulated plasticity for the published MaleCNS connectome.

The upstream ``flybrain`` simulator intentionally freezes every synapse.  This
module changes only anatomically identified KC->MBON synapses.  Recent Kenyon
cell firing supplies an eligibility trace; positive and negative reward
prediction errors select MBONs through their real PAM and PPL1 input strengths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class DopaminePlasticity:
    """Eligibility-trace, DAN-gated depression of real KC->MBON synapses."""

    SCHEMA = 1

    def __init__(
        self,
        brain,
        *,
        state_path: Path,
        learning_rate: float = 0.001,
        eligibility_tau_s: float = 2.0,
        min_scale: float = 0.2,
        recovery_rate: float = 2.0e-4,
    ) -> None:
        if getattr(brain, "device", "cpu") != "cpu":
            raise ValueError(
                "dopamine plasticity currently requires --male-cns-device cpu; "
                "flybrain's CUDA backend keeps a separate immutable CSR matrix"
            )
        if learning_rate <= 0 or eligibility_tau_s <= 0:
            raise ValueError("dopamine learning rate and eligibility tau must be positive")
        self.brain = brain
        self.state_path = Path(state_path)
        self.learning_rate = float(learning_rate)
        self.min_scale = float(min_scale)
        self.recovery_rate = float(recovery_rate)
        self.decay = float(np.exp(-float(brain.dt) / eligibility_tau_s))

        cell_type = np.asarray(brain.cell_type).astype(str)
        self.kc = np.flatnonzero(np.char.startswith(cell_type, "KC"))
        self.mbon = np.flatnonzero(np.char.startswith(cell_type, "MBON"))
        self.pam = np.flatnonzero(np.char.startswith(cell_type, "PAM"))
        self.ppl1 = np.flatnonzero(np.char.startswith(cell_type, "PPL1"))
        if not all(len(group) for group in (self.kc, self.mbon, self.pam, self.ppl1)):
            raise RuntimeError("MaleCNS is missing KC, MBON, PAM, or PPL1 neurons")

        self._kc_lookup = np.full(brain.n, -1, dtype=np.int32)
        self._kc_lookup[self.kc] = np.arange(len(self.kc), dtype=np.int32)
        self._mbon_lookup = np.full(brain.n, -1, dtype=np.int32)
        self._mbon_lookup[self.mbon] = np.arange(len(self.mbon), dtype=np.int32)

        positions: list[np.ndarray] = []
        edge_pre: list[np.ndarray] = []
        edge_post: list[np.ndarray] = []
        for local_pre, neuron in enumerate(self.kc):
            slots = np.arange(brain.indptr[neuron], brain.indptr[neuron + 1])
            local_post = self._mbon_lookup[brain.indices[slots]]
            valid = local_post >= 0
            if np.any(valid):
                positions.append(slots[valid])
                edge_pre.append(np.full(valid.sum(), local_pre, dtype=np.int32))
                edge_post.append(local_post[valid])
        self.edge_positions = np.concatenate(positions)
        self.edge_pre = np.concatenate(edge_pre)
        self.edge_post = np.concatenate(edge_post)
        self.base_weights = np.asarray(brain.weights[self.edge_positions], dtype=np.float32)
        self.scales = np.ones(len(self.edge_positions), dtype=np.float32)
        self.eligibility = np.zeros(len(self.kc), dtype=np.float32)
        self.reward_gate = self._dan_gate(self.pam)
        self.punishment_gate = self._dan_gate(self.ppl1)
        self.pending_signal = 0.0
        self.last_signal = 0.0
        self.updates = 0
        if self.state_path.exists():
            self.load()
        self._apply_weights()

    def _dan_gate(self, neurons: np.ndarray) -> np.ndarray:
        gate = np.zeros(len(self.mbon), dtype=np.float32)
        for neuron in neurons:
            slots = np.arange(self.brain.indptr[neuron], self.brain.indptr[neuron + 1])
            local_post = self._mbon_lookup[self.brain.indices[slots]]
            valid = local_post >= 0
            np.add.at(gate, local_post[valid], np.abs(self.brain.weights[slots[valid]]))
        maximum = float(gate.max(initial=0.0))
        if maximum > 0:
            gate = np.sqrt(gate / maximum)
        return gate

    def observe(self, fired: np.ndarray) -> None:
        self.eligibility *= self.decay
        local = self._kc_lookup[np.asarray(fired, dtype=np.int64)]
        local = local[local >= 0]
        if len(local):
            self.eligibility[np.unique(local)] = 1.0

    def consume_injection(self) -> list[tuple[np.ndarray, float]]:
        signal = self.pending_signal
        self.pending_signal = 0.0
        if signal > 0:
            return [(self.pam, min(1.0, signal))]
        if signal < 0:
            return [(self.ppl1, min(1.0, -signal))]
        return []

    def reinforce(self, prediction_error: float) -> float:
        signal = float(np.clip(prediction_error, -1.0, 1.0))
        self.last_signal = signal
        self.pending_signal = signal
        if self.recovery_rate:
            self.scales += self.recovery_rate * (1.0 - self.scales)
        gate = self.reward_gate if signal >= 0 else self.punishment_gate
        strength = self.eligibility[self.edge_pre] * gate[self.edge_post]
        self.scales *= np.exp(
            -self.learning_rate * abs(signal) * strength,
            dtype=np.float32,
        )
        np.maximum(self.scales, self.min_scale, out=self.scales)
        self._apply_weights()
        self.updates += 1
        return signal

    def _apply_weights(self) -> None:
        self.brain.weights[self.edge_positions] = self.base_weights * self.scales

    def reset_episode(self) -> None:
        self.eligibility.fill(0.0)
        self.pending_signal = 0.0

    def stats(self) -> dict[str, float | int]:
        return {
            "signal": self.last_signal,
            "updates": self.updates,
            "eligible_kc_fraction": float(np.mean(self.eligibility > 0.05)),
            "mean_kc_mbon_scale": float(self.scales.mean()),
            "min_kc_mbon_scale": float(self.scales.min()),
        }

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(f".{self.state_path.name}.tmp")
        with temporary.open("wb") as output:
            np.savez_compressed(
                output,
                schema=np.asarray(self.SCHEMA),
                kc_count=np.asarray(len(self.kc)),
                mbon_count=np.asarray(len(self.mbon)),
                edge_count=np.asarray(len(self.edge_positions)),
                scales=self.scales,
                updates=np.asarray(self.updates),
            )
        temporary.replace(self.state_path)

    def load(self) -> None:
        with np.load(self.state_path, allow_pickle=False) as archive:
            if int(archive["schema"]) != self.SCHEMA:
                raise ValueError("unsupported dopamine-plasticity state schema")
            expected = (len(self.kc), len(self.mbon), len(self.edge_positions))
            actual = (
                int(archive["kc_count"]),
                int(archive["mbon_count"]),
                int(archive["edge_count"]),
            )
            if actual != expected:
                raise ValueError("dopamine state does not match this MaleCNS connectome")
            scales = np.asarray(archive["scales"], dtype=np.float32)
            if scales.shape != self.scales.shape or not np.all(np.isfinite(scales)):
                raise ValueError("invalid dopamine-plasticity synapse scales")
            self.scales[:] = np.clip(scales, self.min_scale, 1.0)
            self.updates = int(archive["updates"])
