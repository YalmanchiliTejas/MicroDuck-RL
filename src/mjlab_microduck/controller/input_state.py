"""Value objects shared by the physical controller and future adapters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NESButton(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    A = "A"
    B = "B"


@dataclass(frozen=True, slots=True)
class NESInputState:
    """Six independent NES button levels for one simulation step."""

    up: bool = False
    down: bool = False
    left: bool = False
    right: bool = False
    a: bool = False
    b: bool = False

    def as_dict(self) -> dict[str, bool]:
        return {
            "UP": self.up,
            "DOWN": self.down,
            "LEFT": self.left,
            "RIGHT": self.right,
            "A": self.a,
            "B": self.b,
        }

    @property
    def active(self) -> tuple[NESButton, ...]:
        values = self.as_dict()
        return tuple(button for button in NESButton if values[button.value])
