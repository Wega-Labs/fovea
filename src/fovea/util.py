"""Shared helpers for the Fovea engine."""

import math
from dataclasses import dataclass


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def percentile(values: list[float], quantile: float) -> float | None:
    """Return the linearly interpolated ``quantile`` of ``values`` (``None`` when empty)."""
    if not values:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be within [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True, slots=True)
class ScreenPoint:
    """Normalized screen coordinates. Origin is top-left. Range is 0..1."""

    x: float
    y: float
