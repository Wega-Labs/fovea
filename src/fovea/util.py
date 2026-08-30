"""Shared helpers for the Fovea engine."""

from dataclasses import dataclass


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True, slots=True)
class ScreenPoint:
    """Normalized screen coordinates. Origin is top-left. Range is 0..1."""

    x: float
    y: float
