"""One Euro Filter: smooth jitter, keep intentional saccades."""

from __future__ import annotations

import math


class OneEuroFilter:
    """Casiez et al. One Euro Filter (low latency + jitter reduction)."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, dcutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x: float | None = None
        self._dx: float = 0.0

    def reset(self) -> None:
        self._x = None
        self._dx = 0.0

    def filter(self, value: float, dt: float) -> float:
        if dt <= 0:
            dt = 1.0 / 30.0
        if self._x is None:
            self._x = value
            return value
        edx = (value - self._x) / dt
        a_d = _alpha(self.dcutoff, dt)
        self._dx = a_d * edx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = _alpha(cutoff, dt)
        self._x = a * value + (1.0 - a) * self._x
        return self._x


class OneEuroPoint:
    def __init__(self, min_cutoff: float, beta: float) -> None:
        self.x = OneEuroFilter(min_cutoff, beta)
        self.y = OneEuroFilter(min_cutoff, beta)

    def reset(self) -> None:
        self.x.reset()
        self.y.reset()

    def filter(self, px: float, py: float, dt: float) -> tuple[float, float]:
        return self.x.filter(px, dt), self.y.filter(py, dt)


def ema(previous: float | None, current: float, alpha: float) -> float:
    if previous is None:
        return current
    a = max(0.0, min(1.0, alpha))
    return a * current + (1.0 - a) * previous


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / dt)
