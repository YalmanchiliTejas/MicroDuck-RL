"""Decoder for the simulation-first four-key Mario controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .input_state import NESInputState


@dataclass(frozen=True, slots=True)
class NESControllerCalibration:
    activate_travel: float = 0.0007
    release_travel: float = 0.0003

    def __post_init__(self) -> None:
        if not 0.0 <= self.release_travel < self.activate_travel:
            raise ValueError("release travel must be below activation travel")


class NESController:
    """Hysteretic decoder for LEFT, RIGHT, A, and B vertical keys."""

    BUTTON_JOINTS = (
        "passive_dpad_left",
        "passive_dpad_right",
        "passive_button_a",
        "passive_button_b",
    )

    def __init__(self, calibration: NESControllerCalibration | None = None) -> None:
        self.calibration = calibration or NESControllerCalibration()
        self._active = [False] * 4

    def reset(self) -> None:
        self._active = [False] * 4

    def update(self, *travels: float) -> NESInputState:
        if len(travels) != 4:
            raise ValueError(f"expected four button travels, got {len(travels)}")
        for index, raw in enumerate(travels):
            travel = float(raw)
            threshold = (
                self.calibration.release_travel
                if self._active[index]
                else self.calibration.activate_travel
            )
            self._active[index] = travel >= threshold
        left, right, button_a, button_b = self._active
        return NESInputState(False, False, left, right, button_a, button_b)

    @classmethod
    def joint_qpos_addresses(cls, model: Any) -> tuple[int, ...]:
        """Resolve the four slider qpos addresses once per simulation."""

        import mujoco

        addresses = []
        for name in cls.BUTTON_JOINTS:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"controller joint not found: {name}")
            addresses.append(int(model.jnt_qposadr[joint_id]))
        return tuple(addresses)

    def update_from_qpos(
        self, qpos: Any, addresses: tuple[int, ...]
    ) -> NESInputState:
        return self.update(*(-float(qpos[address]) for address in addresses))

    def debug_text(self, state: NESInputState, *travels: float) -> str:
        if len(travels) != 4:
            raise ValueError(f"expected four button travels, got {len(travels)}")
        names = ("LEFT", "RIGHT", "A", "B")
        active_values = (state.left, state.right, state.a, state.b)
        lines = ["BUTTON TRAVEL:"]
        lines.extend(
            f"{name:<5} {travel * 1e3:4.2f} mm -> {int(active)}"
            for name, travel, active in zip(names, travels, active_values)
        )
        return "\n".join(lines)
