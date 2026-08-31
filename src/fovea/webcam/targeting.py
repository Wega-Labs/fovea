"""Camera-independent target snapping, hysteresis, and dwell state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from fovea.events import Dwell, DwellProgress, TargetEnter, TargetLeave
from fovea.util import clamp01

_DWELL_PROGRESS_INTERVAL_NS = 66_666_667

type TargetIntentEvent = TargetEnter | TargetLeave | DwellProgress | Dwell


@dataclass(frozen=True, slots=True)
class TargetRect:
    """A host-registered rectangle in display-normalized coordinates."""

    id: str
    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.w, self.h)
        if not self.id.strip():
            raise ValueError("target ids must be non-empty")
        if any(isinstance(value, bool) for value in values):
            raise ValueError("target rectangle coordinates must be numbers")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("target rectangle coordinates must be finite")
        if self.x < 0.0 or self.y < 0.0 or self.w <= 0.0 or self.h <= 0.0:
            raise ValueError("target rectangles require non-negative x/y and positive w/h")
        if self.x + self.w > 1.0 or self.y + self.h > 1.0:
            raise ValueError("target rectangles must fit within display_normalized space")

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (
            self.x - margin <= x <= self.x + self.w + margin
            and self.y - margin <= y <= self.y + self.h + margin
        )

    def distance_to(self, x: float, y: float) -> float:
        dx = max(self.x - x, 0.0, x - (self.x + self.w))
        dy = max(self.y - y, 0.0, y - (self.y + self.h))
        return math.hypot(dx, dy)


def validate_targets(targets: tuple[TargetRect, ...]) -> tuple[TargetRect, ...]:
    ids = [target.id for target in targets]
    if len(ids) != len(set(ids)):
        raise ValueError("target ids must be unique")
    return targets


@dataclass(frozen=True, slots=True)
class TargetMatch:
    target_id: str
    snapped_x: float
    snapped_y: float


@dataclass(frozen=True, slots=True)
class TargetUpdate:
    match: TargetMatch | None
    events: tuple[TargetIntentEvent, ...]


@dataclass
class TargetTracker:
    """Resolve noisy gaze points to host targets and maintain dwell state."""

    dwell_ms: float
    hysteresis: float
    expand: float
    snap_radius: float
    targets: tuple[TargetRect, ...] = ()
    _active_id: str | None = field(default=None, init=False, repr=False)
    _entered_ns: int | None = field(default=None, init=False, repr=False)
    _last_progress_ns: int | None = field(default=None, init=False, repr=False)
    _paused_at_ns: int | None = field(default=None, init=False, repr=False)
    _dwell_emitted: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        settings = (self.dwell_ms, self.hysteresis, self.expand, self.snap_radius)
        if not all(math.isfinite(value) for value in settings):
            raise ValueError("target tracking settings must be finite")
        if self.dwell_ms < 0.0:
            raise ValueError("dwell_ms must be zero or greater")
        if min(self.hysteresis, self.expand, self.snap_radius) < 0.0:
            raise ValueError("target margins must be zero or greater")
        self.targets = validate_targets(tuple(self.targets))

    def replace_targets(
        self,
        targets: tuple[TargetRect, ...],
        timestamp_ns: int,
    ) -> tuple[TargetIntentEvent, ...]:
        replacement = validate_targets(tuple(targets))
        active = self._active_target()
        self.targets = replacement
        if active is None:
            if self._active_id is not None:
                self._reset_active()
            return ()
        unchanged = next((target for target in replacement if target.id == active.id), None)
        if unchanged == active:
            return ()
        event = TargetLeave(id=active.id, timestamp_ns=timestamp_ns)
        self._reset_active()
        return (event,)

    def update(
        self,
        x: float,
        y: float,
        confidence: float,
        timestamp_ns: int,
    ) -> TargetUpdate:
        self._resume(timestamp_ns)
        confidence = clamp01(confidence)
        confidence_margin = self.expand * (1.0 - confidence)
        events: list[TargetIntentEvent] = []
        active = self._active_target()

        if active is not None and not active.contains(
            x,
            y,
            margin=self.hysteresis + confidence_margin,
        ):
            events.append(TargetLeave(id=active.id, timestamp_ns=timestamp_ns))
            self._reset_active()
            active = None

        if active is None:
            active = self._select_target(x, y, confidence_margin)
            if active is not None:
                self._active_id = active.id
                self._entered_ns = timestamp_ns
                self._last_progress_ns = None
                self._dwell_emitted = False
                events.append(TargetEnter(id=active.id, timestamp_ns=timestamp_ns))

        if active is None:
            return TargetUpdate(match=None, events=tuple(events))

        events.extend(self._dwell_events(active.id, timestamp_ns))
        center_x, center_y = active.center
        return TargetUpdate(
            match=TargetMatch(active.id, center_x, center_y),
            events=tuple(events),
        )

    def freeze(self, timestamp_ns: int) -> TargetMatch | None:
        """Pause dwell timing while retaining the active target selection."""
        if self._active_id is not None and self._paused_at_ns is None:
            self._paused_at_ns = timestamp_ns
        return self.current_match()

    def current_match(self) -> TargetMatch | None:
        active = self._active_target()
        if active is None:
            return None
        center_x, center_y = active.center
        return TargetMatch(active.id, center_x, center_y)

    def _resume(self, timestamp_ns: int) -> None:
        paused_at_ns = self._paused_at_ns
        if paused_at_ns is None:
            return
        paused_ns = max(0, timestamp_ns - paused_at_ns)
        if self._entered_ns is not None:
            self._entered_ns += paused_ns
        if self._last_progress_ns is not None:
            self._last_progress_ns += paused_ns
        self._paused_at_ns = None

    def _select_target(
        self,
        x: float,
        y: float,
        confidence_margin: float,
    ) -> TargetRect | None:
        contained = [target for target in self.targets if target.contains(x, y, confidence_margin)]
        if contained:
            return min(contained, key=lambda target: _center_distance(target, x, y))
        nearby = [target for target in self.targets if target.distance_to(x, y) <= self.snap_radius]
        if not nearby:
            return None
        return min(
            nearby,
            key=lambda target: (target.distance_to(x, y), _center_distance(target, x, y)),
        )

    def _dwell_events(self, target_id: str, timestamp_ns: int) -> list[TargetIntentEvent]:
        entered_ns = self._entered_ns
        if entered_ns is None or self._dwell_emitted:
            return []
        dwell_ns = round(self.dwell_ms * 1_000_000)
        elapsed_ns = max(0, timestamp_ns - entered_ns)
        progress = 1.0 if dwell_ns == 0 else min(1.0, elapsed_ns / dwell_ns)
        due = (
            self._last_progress_ns is None
            or timestamp_ns - self._last_progress_ns >= _DWELL_PROGRESS_INTERVAL_NS
            or progress >= 1.0
        )
        events: list[TargetIntentEvent] = []
        if due:
            events.append(
                DwellProgress(
                    id=target_id,
                    progress=progress,
                    timestamp_ns=timestamp_ns,
                )
            )
            self._last_progress_ns = timestamp_ns
        if progress >= 1.0:
            events.append(Dwell(id=target_id, timestamp_ns=timestamp_ns))
            self._dwell_emitted = True
        return events

    def _active_target(self) -> TargetRect | None:
        active_id = self._active_id
        if active_id is None:
            return None
        return next((target for target in self.targets if target.id == active_id), None)

    def _reset_active(self) -> None:
        self._active_id = None
        self._entered_ns = None
        self._last_progress_ns = None
        self._paused_at_ns = None
        self._dwell_emitted = False


def _center_distance(target: TargetRect, x: float, y: float) -> float:
    center_x, center_y = target.center
    return math.hypot(center_x - x, center_y - y)
