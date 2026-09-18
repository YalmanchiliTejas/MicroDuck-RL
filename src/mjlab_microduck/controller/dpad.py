"""Hysteretic decoder for the left-foot two-axis D-pad platform."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DPadCalibration:
    """D-pad limits and thresholds, in radians.

    At the default 2 degree hard stop, the 0.7/0.4 ratios activate at 1.4
    degrees and release at 0.8 degrees. Across a 53 mm half-length plate this
    is approximately 1.3 mm and 0.7 mm of edge displacement.
    """

    max_tilt: float = 0.034906585  # 2 degrees
    activate_ratio: float = 0.7
    release_ratio: float = 0.4

    def __post_init__(self) -> None:
        if self.max_tilt <= 0.0:
            raise ValueError("max_tilt must be positive")
        if not 0.0 <= self.release_ratio < self.activate_ratio <= 1.0:
            raise ValueError("require 0 <= release_ratio < activate_ratio <= 1")

    @property
    def activate_threshold(self) -> float:
        return self.max_tilt * self.activate_ratio

    @property
    def release_threshold(self) -> float:
        return self.max_tilt * self.release_ratio


class DPadDecoder:
    """Decode signed platform angles without vibration-induced flicker.

    ``x`` is lateral: positive is RIGHT and negative is LEFT. ``y`` is
    fore/aft: positive is UP and negative is DOWN. Each axis can select at
    most one direction, while either direction remains independent of A/B.
    """

    def __init__(self, calibration: DPadCalibration | None = None) -> None:
        self.calibration = calibration or DPadCalibration()
        self._x = 0
        self._y = 0

    def reset(self) -> None:
        self._x = 0
        self._y = 0

    def _update_axis(self, value: float, previous: int) -> int:
        activate = self.calibration.activate_threshold
        release = self.calibration.release_threshold
        if previous > 0 and value >= release:
            return 1
        if previous < 0 and value <= -release:
            return -1
        if value >= activate:
            return 1
        if value <= -activate:
            return -1
        return 0

    def update(self, x: float, y: float) -> tuple[bool, bool, bool, bool]:
        self._x = self._update_axis(float(x), self._x)
        self._y = self._update_axis(float(y), self._y)
        return self._y > 0, self._y < 0, self._x < 0, self._x > 0
