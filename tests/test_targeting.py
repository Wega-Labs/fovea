from __future__ import annotations

from itertools import pairwise

import pytest

from fovea.events import Dwell, DwellProgress, TargetEnter, TargetLeave
from fovea.webcam.targeting import TargetRect, TargetTracker


def _ns(milliseconds: float) -> int:
    return round(milliseconds * 1_000_000)


def _tracker(*targets: TargetRect) -> TargetTracker:
    return TargetTracker(
        dwell_ms=500.0,
        hysteresis=0.05,
        expand=0.02,
        snap_radius=0.06,
        targets=targets,
    )


def test_adjacent_targets_use_hysteresis_before_switching() -> None:
    tracker = _tracker(
        TargetRect("left", 0.10, 0.40, 0.30, 0.20),
        TargetRect("right", 0.45, 0.40, 0.30, 0.20),
    )
    cases = [
        (0, 0.20, "left", (TargetEnter, DwellProgress)),
        (70, 0.44, "left", (DwellProgress,)),
        (140, 0.46, "right", (TargetLeave, TargetEnter, DwellProgress)),
        (210, 0.78, "right", (DwellProgress,)),
        (280, 0.90, None, (TargetLeave,)),
    ]
    for milliseconds, x, target_id, event_types in cases:
        update = tracker.update(x, 0.50, 1.0, _ns(milliseconds))
        assert (None if update.match is None else update.match.target_id) == target_id
        assert tuple(type(event) for event in update.events) == event_types


def test_nearby_point_snaps_to_target_center() -> None:
    tracker = _tracker(TargetRect("button", 0.40, 0.40, 0.20, 0.20))
    update = tracker.update(0.37, 0.50, 1.0, _ns(0))
    assert update.match is not None
    assert update.match.target_id == "button"
    assert update.match.snapped_x == pytest.approx(0.5)
    assert update.match.snapped_y == pytest.approx(0.5)


def test_low_confidence_expands_target_acquisition() -> None:
    target = TargetRect("button", 0.40, 0.40, 0.20, 0.20)
    strict = TargetTracker(500.0, 0.0, 0.10, 0.0, (target,))
    expanded = TargetTracker(500.0, 0.0, 0.10, 0.0, (target,))
    assert strict.update(0.351, 0.50, 1.0, _ns(0)).match is None
    assert expanded.update(0.351, 0.50, 0.5, _ns(0)).match is not None


def test_uncertain_tracking_pauses_dwell_clock() -> None:
    tracker = _tracker(TargetRect("button", 0.40, 0.40, 0.20, 0.20))
    tracker.update(0.5, 0.5, 1.0, _ns(0))
    tracker.update(0.5, 0.5, 1.0, _ns(300))
    assert tracker.freeze(_ns(300)) is not None

    resumed = tracker.update(0.5, 0.5, 1.0, _ns(800))
    assert not any(isinstance(event, Dwell) for event in resumed.events)
    completed = tracker.update(0.5, 0.5, 1.0, _ns(1000))
    assert sum(isinstance(event, Dwell) for event in completed.events) == 1


def test_jitter_inside_one_target_emits_exactly_one_dwell() -> None:
    tracker = _tracker(TargetRect("button", 0.40, 0.40, 0.20, 0.20))
    events = []
    for milliseconds in range(0, 1001, 50):
        jitter = 0.015 if milliseconds % 100 else -0.015
        update = tracker.update(0.5 + jitter, 0.5 - jitter, 0.9, _ns(milliseconds))
        events.extend(update.events)

    dwells = [event for event in events if isinstance(event, Dwell)]
    progress = [event for event in events if isinstance(event, DwellProgress)]
    assert len(dwells) == 1
    assert dwells[0].id == "button"
    assert all(
        later.timestamp_ns - earlier.timestamp_ns >= _ns(1000 / 15)
        for earlier, later in pairwise(progress)
    )


def test_replace_all_preserves_identical_active_target_and_leaves_changed_target() -> None:
    target = TargetRect("button", 0.40, 0.40, 0.20, 0.20)
    tracker = _tracker(target)
    tracker.update(0.5, 0.5, 1.0, _ns(0))
    assert tracker.replace_targets((target,), _ns(100)) == ()

    changed = TargetRect("button", 0.45, 0.40, 0.20, 0.20)
    events = tracker.replace_targets((changed,), _ns(200))
    assert len(events) == 1
    assert isinstance(events[0], TargetLeave)


def test_target_ids_must_be_unique() -> None:
    target = TargetRect("duplicate", 0.10, 0.10, 0.20, 0.20)
    with pytest.raises(ValueError, match="unique"):
        _tracker(target, target)
