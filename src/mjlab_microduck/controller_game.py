"""Controller-pad decoding and a tiny side-scrolling game simulation.

This module intentionally has no MuJoCo, Torch, or renderer dependency.  The
MuJoCo task can feed it pad travel, while hardware can feed it calibrated force
or displacement values.  Both paths produce the same :class:`ControllerFrame`.

The game is deliberately a small, original Mario-like dynamics model rather
than an emulator or Nintendo asset.  It is useful for deterministic tests and
for training the controller loop; an emulator adapter can consume the same
controller frames later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import copysign
from typing import Mapping


class Button(str, Enum):
    """The minimum useful input set for a platform game."""

    LEFT = "left"
    RIGHT = "right"
    JUMP = "jump"


@dataclass(frozen=True, slots=True)
class ControllerFrame:
    """Button levels for one control frame."""

    left: bool = False
    right: bool = False
    jump: bool = False

    @classmethod
    def from_buttons(cls, buttons: set[Button]) -> "ControllerFrame":
        return cls(
            left=Button.LEFT in buttons,
            right=Button.RIGHT in buttons,
            jump=Button.JUMP in buttons,
        )

    @property
    def horizontal_axis(self) -> int:
        """Return -1, 0, or +1; opposite directions cancel safely."""

        return int(self.right) - int(self.left)


@dataclass(frozen=True, slots=True)
class PadCalibration:
    """Hysteresis thresholds for positive pad displacement in metres."""

    press_travel: float = 0.004
    release_travel: float = 0.002

    def __post_init__(self) -> None:
        if self.release_travel < 0.0:
            raise ValueError("release_travel must be non-negative")
        if self.press_travel <= self.release_travel:
            raise ValueError("press_travel must be greater than release_travel")


@dataclass(frozen=True, slots=True)
class PadFrame:
    """Decoded pad state, including one-frame edges."""

    controller: ControllerFrame
    just_pressed: frozenset[Button] = frozenset()
    just_released: frozenset[Button] = frozenset()


class PadBank:
    """Convert noisy pad travel into stable controller button states.

    ``travel`` is positive in the pressed direction. Missing pad readings are
    treated as zero so a disconnected sensor cannot leave a button stuck down.
    """

    def __init__(self, calibration: PadCalibration | None = None) -> None:
        self.calibration = calibration or PadCalibration()
        self._pressed: set[Button] = set()

    @property
    def pressed(self) -> frozenset[Button]:
        return frozenset(self._pressed)

    def reset(self) -> None:
        self._pressed.clear()

    def update(self, travel: Mapping[Button | str, float]) -> PadFrame:
        normalized = {
            button: float(travel.get(button, travel.get(button.value, 0.0)))
            for button in Button
        }
        previous = self._pressed.copy()
        for button, value in normalized.items():
            if button in self._pressed:
                if value <= self.calibration.release_travel:
                    self._pressed.remove(button)
            elif value >= self.calibration.press_travel:
                self._pressed.add(button)

        return PadFrame(
            controller=ControllerFrame.from_buttons(self._pressed),
            just_pressed=frozenset(self._pressed - previous),
            just_released=frozenset(previous - self._pressed),
        )


class GameStatus(str, Enum):
    RUNNING = "running"
    WON = "won"
    LOST = "lost"


@dataclass(frozen=True, slots=True)
class PlatformGameConfig:
    """Physics and level constants for the deterministic training game."""

    gravity: float = -18.0
    run_acceleration: float = 18.0
    ground_deceleration: float = 24.0
    max_run_speed: float = 3.0
    jump_speed: float = 6.2
    player_radius: float = 0.18
    start_x: float = 0.8
    goal_x: float = 9.5
    death_y: float = -1.5
    # Inclusive horizontal ground intervals. The default has one jumpable gap.
    ground_segments: tuple[tuple[float, float], ...] = ((0.0, 4.0), (5.0, 10.5))
    max_substep_s: float = 1.0 / 120.0

    def __post_init__(self) -> None:
        if self.max_substep_s <= 0.0:
            raise ValueError("max_substep_s must be positive")
        if not self.ground_segments:
            raise ValueError("ground_segments cannot be empty")
        if any(end <= start for start, end in self.ground_segments):
            raise ValueError("every ground segment must have start < end")


@dataclass(frozen=True, slots=True)
class PlatformGameState:
    x: float
    y: float
    vx: float
    vy: float
    grounded: bool
    status: GameStatus
    elapsed_s: float


class PlatformGame:
    """Small fixed-step side-scroller driven by :class:`ControllerFrame`."""

    def __init__(self, config: PlatformGameConfig | None = None) -> None:
        self.config = config or PlatformGameConfig()
        self._jump_was_down = False
        self.reset()

    @property
    def state(self) -> PlatformGameState:
        return PlatformGameState(
            x=self._x,
            y=self._y,
            vx=self._vx,
            vy=self._vy,
            grounded=self._grounded,
            status=self._status,
            elapsed_s=self._elapsed_s,
        )

    def reset(self) -> PlatformGameState:
        self._x = self.config.start_x
        self._y = self.config.player_radius
        self._vx = 0.0
        self._vy = 0.0
        self._grounded = True
        self._status = GameStatus.RUNNING
        self._elapsed_s = 0.0
        self._jump_was_down = False
        return self.state

    def _has_ground(self, x: float) -> bool:
        radius = self.config.player_radius
        return any(
            start + radius <= x <= end - radius
            for start, end in self.config.ground_segments
        )

    def next_gap_start(self) -> float | None:
        """Return the next ground-segment end ahead of the player."""

        radius = self.config.player_radius
        for start, end in self.config.ground_segments:
            if start + radius <= self._x <= end - radius:
                return end if end < self.config.goal_x else None
        return None

    def _approach_zero(self, value: float, amount: float) -> float:
        if abs(value) <= amount:
            return 0.0
        return value - copysign(amount, value)

    def _step_once(self, controller: ControllerFrame, dt: float, jump_edge: bool) -> None:
        axis = controller.horizontal_axis
        if axis:
            self._vx += axis * self.config.run_acceleration * dt
            self._vx = max(-self.config.max_run_speed, min(self.config.max_run_speed, self._vx))
        elif self._grounded:
            self._vx = self._approach_zero(self._vx, self.config.ground_deceleration * dt)

        if jump_edge and self._grounded:
            self._vy = self.config.jump_speed
            self._grounded = False

        old_y = self._y
        self._vy += self.config.gravity * dt
        self._x += self._vx * dt
        self._y += self._vy * dt

        floor_y = self.config.player_radius
        crossed_floor = old_y >= floor_y and self._y <= floor_y
        if crossed_floor and self._vy <= 0.0 and self._has_ground(self._x):
            self._y = floor_y
            self._vy = 0.0
            self._grounded = True
        elif not self._has_ground(self._x) or self._y > floor_y:
            self._grounded = False

        self._x = max(self.config.player_radius, self._x)
        if self._x >= self.config.goal_x:
            self._status = GameStatus.WON
        elif self._y < self.config.death_y:
            self._status = GameStatus.LOST

    def step(self, controller: ControllerFrame, dt: float) -> PlatformGameState:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if self._status is not GameStatus.RUNNING:
            return self.state

        jump_edge = controller.jump and not self._jump_was_down
        remaining = dt
        first = True
        while remaining > 1e-12:
            substep = min(remaining, self.config.max_substep_s)
            self._step_once(controller, substep, jump_edge and first)
            self._elapsed_s += substep
            remaining -= substep
            first = False
            if self._status is not GameStatus.RUNNING:
                break
        self._jump_was_down = controller.jump
        return self.state


@dataclass(slots=True)
class PadGameBridge:
    """End-to-end bridge used by either a simulator or physical pad reader."""

    pads: PadBank = field(default_factory=PadBank)
    game: PlatformGame = field(default_factory=PlatformGame)

    def reset(self) -> PlatformGameState:
        self.pads.reset()
        return self.game.reset()

    def step(
        self, pad_travel: Mapping[Button | str, float], dt: float
    ) -> tuple[PadFrame, PlatformGameState]:
        pad_frame = self.pads.update(pad_travel)
        return pad_frame, self.game.step(pad_frame.controller, dt)


@dataclass(frozen=True, slots=True)
class PlatformGamePlanner:
    """Minimal high-level planner that runs right and jumps before gaps."""

    jump_lead_m: float = 0.9

    def command(self, game: PlatformGame) -> ControllerFrame:
        state = game.state
        if state.status is not GameStatus.RUNNING:
            return ControllerFrame()
        gap_start = game.next_gap_start()
        jump = bool(
            state.grounded
            and gap_start is not None
            and 0.0 <= gap_start - state.x <= self.jump_lead_m
        )
        return ControllerFrame(right=True, jump=jump)


@dataclass(slots=True)
class MarioDuckLoop:
    """Coordinate high-level requests with physically measured pad input.

    ``requested`` is suitable for the policy's 3D twist slot. ``step`` accepts
    measured travel and advances the game using only decoded physical presses.
    The planner request never directly changes game state.
    """

    bridge: PadGameBridge = field(default_factory=PadGameBridge)
    planner: PlatformGamePlanner = field(default_factory=PlatformGamePlanner)

    @property
    def requested(self) -> ControllerFrame:
        return self.planner.command(self.bridge.game)

    @property
    def command_vector(self) -> tuple[float, float, float]:
        request = self.requested
        return (float(request.left), float(request.right), float(request.jump))

    def reset(self) -> PlatformGameState:
        return self.bridge.reset()

    def step(
        self, pad_travel: Mapping[Button | str, float], dt: float
    ) -> tuple[ControllerFrame, PadFrame, PlatformGameState]:
        request = self.requested
        pads, state = self.bridge.step(pad_travel, dt)
        return request, pads, state
