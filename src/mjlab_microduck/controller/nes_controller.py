"""Combined physical controller decoder; contains no emulator-specific code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .ab_rocker import ABRockerCalibration, ABRockerDecoder
from .dpad import DPadCalibration, DPadDecoder
from .input_state import NESInputState


@dataclass(frozen=True, slots=True)
class NESControllerCalibration:
    dpad: DPadCalibration = field(default_factory=DPadCalibration)
    ab: ABRockerCalibration = field(default_factory=ABRockerCalibration)


class NESController:
    """Expose a stable NES state from the three passive controller joints."""

    DPAD_X_JOINT = "passive_dpad_x"
    DPAD_Y_JOINT = "passive_dpad_y"
    AB_JOINT = "passive_ab_rocker"
    AB_PRESS_JOINT = "passive_ab_press"

    def __init__(self, calibration: NESControllerCalibration | None = None) -> None:
        self.calibration = calibration or NESControllerCalibration()
        self.dpad = DPadDecoder(self.calibration.dpad)
        self.ab = ABRockerDecoder(self.calibration.ab)

    def reset(self) -> None:
        self.dpad.reset()
        self.ab.reset()

    def update(
        self,
        dpad_x: float,
        dpad_y: float,
        ab_angle: float,
        ab_press_travel: float = 0.0,
    ) -> NESInputState:
        up, down, left, right = self.dpad.update(dpad_x, dpad_y)
        a, b = self.ab.update(ab_angle, ab_press_travel)
        return NESInputState(up=up, down=down, left=left, right=right, a=a, b=b)

    @classmethod
    def joint_qpos_addresses(cls, model: Any) -> tuple[int, int, int, int]:
        """Resolve joint names once for use in a per-step simulation loop."""

        import mujoco

        addresses = []
        for name in (
            cls.DPAD_X_JOINT,
            cls.DPAD_Y_JOINT,
            cls.AB_JOINT,
            cls.AB_PRESS_JOINT,
        ):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"controller joint not found: {name}")
            addresses.append(int(model.jnt_qposadr[joint_id]))
        return tuple(addresses)  # type: ignore[return-value]

    def update_from_qpos(
        self, qpos: Any, addresses: tuple[int, int, int, int]
    ) -> NESInputState:
        x_adr, y_adr, ab_adr, press_adr = addresses
        # The physical slide moves in negative Z, but decoder travel is positive.
        return self.update(
            qpos[x_adr], qpos[y_adr], qpos[ab_adr], -qpos[press_adr]
        )

    def debug_text(
        self,
        state: NESInputState,
        dpad_x: float,
        dpad_y: float,
        ab_angle: float,
        ab_press_travel: float = 0.0,
    ) -> str:
        dpad_active = "+".join(b.value for b in state.active if b.value in {"UP", "DOWN", "LEFT", "RIGHT"}) or "NEUTRAL"
        ab_active = "+".join(b.value for b in state.active if b.value in {"A", "B"}) or "NEUTRAL"
        values = state.as_dict()
        lines = [
            f"DPAD: x={dpad_x:+.3f} y={dpad_y:+.3f} rad -> {dpad_active}",
            f"AB: angle={ab_angle:+.3f} rad press={ab_press_travel * 1e3:.1f} mm -> {ab_active}",
            "",
            "NES:",
        ]
        lines.extend(f"{name:<5} = {int(active)}" for name, active in values.items())
        return "\n".join(lines)
