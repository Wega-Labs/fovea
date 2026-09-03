"""Camera-independent fixation, blink, saccade, pursuit, and wink state machines."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from itertools import pairwise
from statistics import median

import numpy as np

from fovea.events import Blink, DoubleBlink, Eye, Fixation, LongBlink, Saccade, Wink

_FIXATION_EMIT_INTERVAL_NS = 100_000_000
_ADAPTIVE_MIN_SAMPLES = 5

type BlinkTriggerEvent = LongBlink | DoubleBlink


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
        # Keep the newest sample at or before the cutoff so the retained span
        # covers the full window; pruning everything older than the cutoff
        # would leave a span that only reaches ``stability_ms`` when a sample
        # lands exactly on it, which real timestamps never do.
        while len(self._points) > 1 and self._points[1][0] <= cutoff:
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


@dataclass(frozen=True, slots=True)
class VelocityUpdate:
    """Per-sample velocity classification produced by :class:`SaccadeDetector`.

    ``moving`` is true for any ordered, finite sample at or above the pursuit
    velocity (saccading samples included). ``pursuit`` is the stricter public
    flag that requires a sustained, directionally coherent sub-saccadic band.
    ``saccade`` carries the completed event on the landing sample only.
    """

    moving: bool
    saccading: bool
    pursuit: bool
    saccade: Saccade | None = None


_STILL = VelocityUpdate(moving=False, saccading=False, pursuit=False)


@dataclass
class SaccadeDetector:
    """I-VT saccade detector with a smooth-pursuit qualifier.

    Velocity is measured between consecutive accepted samples in
    display-normalized units per second. The first sample at or above
    ``velocity_threshold`` starts a saccade whose origin is the previous
    sub-threshold sample; the first sub-threshold sample afterwards lands it
    and carries the completed :class:`Saccade`. Samples whose speed stays in
    ``[pursuit_velocity, velocity_threshold)`` form a pursuit band; ``pursuit``
    is reported once the band has lasted ``pursuit_ms`` and the trailing
    ``pursuit_ms`` window is directionally coherent (net displacement over
    path length at or above ``pursuit_coherence``).

    Samples whose timestamp does not advance are ignored completely, and a
    non-finite coordinate resets the detector without emitting.
    """

    velocity_threshold: float
    pursuit_velocity: float
    pursuit_ms: float
    pursuit_coherence: float
    _last: tuple[int, float, float] | None = field(default=None, init=False, repr=False)
    _origin: tuple[int, float, float] | None = field(default=None, init=False, repr=False)
    _band_start_ns: int | None = field(default=None, init=False, repr=False)
    _band: deque[tuple[int, float, float]] = field(
        default_factory=deque,
        init=False,
        repr=False,
    )
    _state: VelocityUpdate = field(default=_STILL, init=False, repr=False)

    def __post_init__(self) -> None:
        values = (
            self.velocity_threshold,
            self.pursuit_velocity,
            self.pursuit_ms,
            self.pursuit_coherence,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("saccade settings must be finite")
        if not 0.0 < self.pursuit_velocity < self.velocity_threshold:
            raise ValueError("pursuit_velocity must be positive and below velocity_threshold")
        if self.pursuit_ms < 0.0:
            raise ValueError("pursuit_ms must be zero or greater")
        if not 0.0 <= self.pursuit_coherence <= 1.0:
            raise ValueError("pursuit_coherence must be between 0.0 and 1.0")

    def update(self, x: float, y: float, timestamp_ns: int) -> VelocityUpdate:
        if not (math.isfinite(x) and math.isfinite(y)):
            self.reset()
            return _STILL
        last = self._last
        if last is None:
            self._last = (timestamp_ns, x, y)
            self._state = _STILL
            return _STILL
        last_ns, last_x, last_y = last
        if timestamp_ns <= last_ns:
            return VelocityUpdate(self._state.moving, self._state.saccading, self._state.pursuit)

        velocity = math.hypot(x - last_x, y - last_y) / ((timestamp_ns - last_ns) / 1e9)
        sample = (timestamp_ns, x, y)
        self._last = sample
        saccade: Saccade | None = None
        if velocity >= self.velocity_threshold:
            if self._origin is None:
                self._origin = last
            self._clear_band()
            state = VelocityUpdate(moving=True, saccading=True, pursuit=False)
        else:
            origin = self._origin
            if origin is not None:
                self._origin = None
                onset_ns, from_x, from_y = origin
                saccade = Saccade(
                    from_x=from_x,
                    from_y=from_y,
                    to_x=x,
                    to_y=y,
                    amplitude=math.hypot(x - from_x, y - from_y),
                    duration_ms=(timestamp_ns - onset_ns) / 1_000_000.0,
                    timestamp_ns=timestamp_ns,
                )
            if velocity < self.pursuit_velocity:
                self._clear_band()
                state = _STILL
            elif saccade is not None:
                self._clear_band()
                state = VelocityUpdate(moving=True, saccading=False, pursuit=False)
            else:
                pursuit = self._track_band(last, sample)
                state = VelocityUpdate(moving=True, saccading=False, pursuit=pursuit)
        self._state = state
        return VelocityUpdate(state.moving, state.saccading, state.pursuit, saccade)

    def reset(self) -> None:
        self._last = None
        self._origin = None
        self._clear_band()
        self._state = _STILL

    def _clear_band(self) -> None:
        self._band.clear()
        self._band_start_ns = None

    def _track_band(
        self,
        previous: tuple[int, float, float],
        sample: tuple[int, float, float],
    ) -> bool:
        if self._band_start_ns is None:
            self._band_start_ns = previous[0]
            self._band.append(previous)
        self._band.append(sample)
        window_ns = max(0, round(self.pursuit_ms * 1_000_000))
        cutoff = sample[0] - window_ns
        while len(self._band) > 2 and self._band[1][0] <= cutoff:
            self._band.popleft()
        if sample[0] - self._band_start_ns < window_ns:
            return False
        return self._coherence() >= self.pursuit_coherence

    def _coherence(self) -> float:
        points = list(self._band)
        path = sum(
            math.hypot(later[1] - earlier[1], later[2] - earlier[2])
            for earlier, later in pairwise(points)
        )
        if path <= 0.0:
            return 1.0
        net = math.hypot(points[-1][1] - points[0][1], points[-1][2] - points[0][2])
        return net / path


@dataclass
class WinkDetector:
    """Detect a deliberate single-eye closure while the other eye stays open.

    ``update`` receives each eye's closure as ``True``/``False`` or ``None``
    when that eye's landmarks are invalid. A candidate wink starts when exactly
    one valid eye is closed and the other valid eye is open. It is cancelled
    when the other eye also closes (natural-blink onset asymmetry), when either
    eye becomes invalid, or when the closure exceeds ``max_ms``. After a
    cancellation, a both-closed frame, or an invalid frame, no new candidate is
    accepted until both eyes have been seen open and valid again, so the
    trailing eye of a natural blink cannot start a wink. On reopen a
    :class:`Wink` is emitted if the closure lasted at least ``min_ms``.
    """

    min_ms: float
    max_ms: float
    _eye: Eye | None = field(default=None, init=False, repr=False)
    _started_ns: int | None = field(default=None, init=False, repr=False)
    _confidence: float = field(default=0.0, init=False, repr=False)
    _blocked: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not (math.isfinite(self.min_ms) and math.isfinite(self.max_ms)):
            raise ValueError("wink durations must be finite")
        if self.min_ms < 0.0:
            raise ValueError("min_ms must be zero or greater")
        if self.max_ms < self.min_ms:
            raise ValueError("max_ms must be at least min_ms")

    def update(
        self,
        left_closed: bool | None,
        right_closed: bool | None,
        confidence: float,
        timestamp_ns: int,
    ) -> Wink | None:
        both_valid = left_closed is not None and right_closed is not None
        both_open = both_valid and not left_closed and not right_closed
        if self._eye is None:
            if both_open:
                self._blocked = False
                return None
            if not both_valid or (left_closed and right_closed):
                self._blocked = True
                return None
            if self._blocked:
                return None
            self._eye = Eye.LEFT if left_closed else Eye.RIGHT
            self._started_ns = timestamp_ns
            self._confidence = confidence
            return None

        started_ns = self._started_ns
        assert started_ns is not None
        eye_closed = left_closed if self._eye is Eye.LEFT else right_closed
        other_closed = right_closed if self._eye is Eye.LEFT else left_closed
        if both_valid and eye_closed is True and other_closed is False:
            self._confidence = max(self._confidence, confidence)
            if (timestamp_ns - started_ns) / 1_000_000.0 > self.max_ms:
                self._cancel()
            return None
        if both_open:
            duration_ms = (timestamp_ns - started_ns) / 1_000_000.0
            event_confidence = max(self._confidence, confidence)
            eye = self._eye
            self.reset()
            if self.min_ms <= duration_ms <= self.max_ms:
                return Wink(
                    eye=Eye.LEFT if eye is Eye.LEFT else Eye.RIGHT,
                    duration_ms=duration_ms,
                    confidence=event_confidence,
                    timestamp_ns=timestamp_ns,
                )
            return None
        self._cancel()
        return None

    def reset(self) -> None:
        self._eye = None
        self._started_ns = None
        self._confidence = 0.0
        self._blocked = False

    def _cancel(self) -> None:
        self.reset()
        self._blocked = True


@dataclass
class BlinkTriggerDetector:
    """Classify measured blinks into long-blink and double-blink triggers.

    The long-blink threshold starts at ``long_blink_ms`` and, once at least
    five natural blinks have been observed, rises to
    ``max(long_blink_ms, long_blink_factor * median)`` of the most recent
    ``natural_blink_window`` natural durations. Adaptation can only raise the
    threshold, so ``long_blink_ms`` is a conservative cold-start floor. Long
    blinks are excluded from the learned distribution. Two natural blinks
    whose gap (second closing minus first reopening) is within
    ``double_blink_ms`` yield one :class:`DoubleBlink`; a long blink clears any
    pending natural blink so natural-long-natural never pairs.

    The learned distribution lives for the life of the detector instance,
    which the frame processor constructs once per ``events()`` session.
    ``reset`` clears only the pending sequence state.
    """

    long_blink_ms: float
    long_blink_factor: float
    natural_blink_window: int
    double_blink_ms: float
    _durations: deque[float] = field(default_factory=deque, init=False, repr=False)
    _pending: Blink | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        values = (self.long_blink_ms, self.long_blink_factor, self.double_blink_ms)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("blink trigger settings must be finite")
        if self.long_blink_ms < 0.0 or self.double_blink_ms < 0.0:
            raise ValueError("blink trigger durations must be zero or greater")
        if self.long_blink_factor < 1.0:
            raise ValueError("long_blink_factor must be at least 1.0")
        if self.natural_blink_window < _ADAPTIVE_MIN_SAMPLES:
            raise ValueError(f"natural_blink_window must be at least {_ADAPTIVE_MIN_SAMPLES}")
        self._durations = deque(maxlen=self.natural_blink_window)

    @property
    def long_blink_threshold_ms(self) -> float:
        """The current long-blink threshold, adapted to observed natural blinks."""
        if len(self._durations) < _ADAPTIVE_MIN_SAMPLES:
            return self.long_blink_ms
        return max(self.long_blink_ms, self.long_blink_factor * median(self._durations))

    def update(self, blink: Blink) -> tuple[BlinkTriggerEvent, ...]:
        if blink.duration_ms >= self.long_blink_threshold_ms:
            self._pending = None
            return (LongBlink(duration_ms=blink.duration_ms, timestamp_ns=blink.timestamp_ns),)
        self._durations.append(blink.duration_ms)
        pending = self._pending
        if pending is not None:
            start_ns = blink.timestamp_ns - round(blink.duration_ms * 1_000_000)
            gap_ms = (start_ns - pending.timestamp_ns) / 1_000_000.0
            if 0.0 <= gap_ms <= self.double_blink_ms:
                self._pending = None
                return (DoubleBlink(timestamp_ns=blink.timestamp_ns),)
        self._pending = blink
        return ()

    def reset(self) -> None:
        """Drop the pending blink sequence while preserving the learned durations."""
        self._pending = None
