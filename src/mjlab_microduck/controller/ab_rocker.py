"""Hysteretic decoder for the right-foot A/B rocker."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ABRockerCalibration:
    max_tilt: float = 0.034906585  # 2 degrees
    activate_ratio: float = 0.3
    release_ratio: float = 0.1
    chord_press_travel: float = 0.0007
    chord_release_travel: float = 0.0005

    def __post_init__(self) -> None:
        if self.max_tilt <= 0.0:
            raise ValueError("max_tilt must be positive")
        if not 0.0 <= self.release_ratio < self.activate_ratio <= 1.0:
            raise ValueError("require 0 <= release_ratio < activate_ratio <= 1")
        if not 0.0 <= self.chord_release_travel < self.chord_press_travel:
            raise ValueError("chord release travel must be below press travel")

    @property
    def activate_threshold(self) -> float:
        return self.max_tilt * self.activate_ratio

    @property
    def release_threshold(self) -> float:
        return self.max_tilt * self.release_ratio


class ABRockerDecoder:
    """Positive fore/aft tilt selects A; negative tilt selects B."""

    def __init__(self, calibration: ABRockerCalibration | None = None) -> None:
        self.calibration = calibration or ABRockerCalibration()
        self._direction = 0
        self._chord = False

    def reset(self) -> None:
        self._direction = 0
        self._chord = False

    def update(self, angle: float, press_travel: float = 0.0) -> tuple[bool, bool]:
        value = float(angle)
        activate = self.calibration.activate_threshold
        release = self.calibration.release_threshold
        if self._direction > 0 and value >= release:
            self._direction = 1
        elif self._direction < 0 and value <= -release:
            self._direction = -1
        elif value >= activate:
            self._direction = 1
        elif value <= -activate:
            self._direction = -1
        else:
            self._direction = 0

        travel = float(press_travel)
        if self._chord:
            self._chord = travel > self.calibration.chord_release_travel
        else:
            self._chord = travel >= self.calibration.chord_press_travel
        return self._chord or self._direction > 0, self._chord or self._direction < 0
