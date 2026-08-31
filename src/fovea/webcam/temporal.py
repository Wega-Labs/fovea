"""Camera-independent fixation and blink state machines."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from fovea.events import Blink, Eye, Fixation

_FIXATION_EMIT_INTERVAL_NS = 100_000_000


@dataclass
class FixationDetector:
    """I-DT detector over smoothed display-normalized gaze points."""

    stability_ms: float
    radius: float
    _points: deque[tuple[int, float, float, float]] = field(
        default_factory=deque,
        init=False,
        repr=False,
    )
    _active_since_ns: int | None = field(default=None, init=False, repr=False)
    _last_emit_ns: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.stability_ms < 0:
            raise ValueError("stability_ms must be zero or greater")
        if self.radius < 0:
            raise ValueError("radius must be zero or greater")

    def update(
        self,
        x: float,
        y: float,
        confidence: float,
        timestamp_ns: int,
    ) -> Fixation | None:
        self._points.append((timestamp_ns, x, y, confidence))
        window_ns = max(0, round(self.stability_ms * 1_000_000))
        cutoff = timestamp_ns - window_ns
        while len(self._points) > 1 and self._points[0][0] < cutoff:
            self._points.popleft()

        span_ns = timestamp_ns - self._points[0][0]
        if span_ns < window_ns or not self._is_stable():
            if self._active_since_ns is not None:
                latest = self._points[-1]
                self.reset()
                self._points.append(latest)
            return None

        active_since_ns = self._active_since_ns
        if active_since_ns is None:
            active_since_ns = self._points[0][0]
            self._active_since_ns = active_since_ns
        last_emit_ns = self._last_emit_ns
        if last_emit_ns is not None and timestamp_ns - last_emit_ns < _FIXATION_EMIT_INTERVAL_NS:
            return None

        self._last_emit_ns = timestamp_ns
        points = np.asarray([(point[1], point[2]) for point in self._points], dtype=np.float64)
        confidences = [point[3] for point in self._points]
        return Fixation(
            x=float(np.mean(points[:, 0])),
            y=float(np.mean(points[:, 1])),
            duration_ms=(timestamp_ns - active_since_ns) / 1_000_000.0,
            confidence=float(np.mean(confidences)),
            timestamp_ns=timestamp_ns,
        )

    def _is_stable(self) -> bool:
        if not self._points:
            return False
        x_values = [point[1] for point in self._points]
        y_values = [point[2] for point in self._points]
        return (
            max(x_values) - min(x_values) <= 2.0 * self.radius
            and max(y_values) - min(y_values) <= 2.0 * self.radius
        )

    def reset(self) -> None:
        self._points.clear()
        self._active_since_ns = None
        self._last_emit_ns = None


@dataclass
class BlinkDetector:
    """Measure a blink from its closing transition until the eyes reopen."""

    _started_ns: int | None = field(default=None, init=False, repr=False)
    _confidence: float = field(default=0.0, init=False, repr=False)

    def update(
        self,
        blinking: bool,
        confidence: float,
        timestamp_ns: int,
    ) -> Blink | None:
        if blinking:
            if self._started_ns is None:
                self._started_ns = timestamp_ns
                self._confidence = confidence
            else:
                self._confidence = max(self._confidence, confidence)
            return None
        started_ns = self._started_ns
        if started_ns is None:
            return None
        event = Blink(
            eye=Eye.BOTH,
            duration_ms=max(0.0, (timestamp_ns - started_ns) / 1_000_000.0),
            confidence=self._confidence,
            timestamp_ns=timestamp_ns,
        )
        self.reset()
        return event

    def reset(self) -> None:
        self._started_ns = None
        self._confidence = 0.0
