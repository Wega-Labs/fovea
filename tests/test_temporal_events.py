from __future__ import annotations

from itertools import pairwise

import pytest

from fovea.webcam.temporal import BlinkDetector, FixationDetector


def _ns(milliseconds: float) -> int:
    return round(milliseconds * 1_000_000)


def test_stable_hold_emits_growing_fixations_at_no_more_than_ten_hz() -> None:
    detector = FixationDetector(stability_ms=300.0, radius=0.04)
    events = []
    for milliseconds in range(0, 801, 20):
        jitter = 0.005 if milliseconds % 40 else -0.005
        event = detector.update(0.5 + jitter, 0.5 - jitter, 0.8, _ns(milliseconds))
        if event is not None:
            events.append(event)

    assert events
    assert events[0].duration_ms >= 300.0
    assert events[-1].duration_ms > events[0].duration_ms
    assert all(
        later.timestamp_ns - earlier.timestamp_ns >= _ns(100.0)
        for earlier, later in pairwise(events)
    )
    assert all(event.x == pytest.approx(0.5, abs=0.01) for event in events)


def test_saccade_breaks_fixation_and_requires_a_new_stability_window() -> None:
    detector = FixationDetector(stability_ms=200.0, radius=0.03)
    assert detector.update(0.5, 0.5, 1.0, _ns(0)) is None
    assert detector.update(0.5, 0.5, 1.0, _ns(200)) is not None
    assert detector.update(0.8, 0.5, 1.0, _ns(250)) is None
    assert detector.update(0.8, 0.5, 1.0, _ns(400)) is None
    restarted = detector.update(0.8, 0.5, 1.0, _ns(450))
    assert restarted is not None
    assert restarted.duration_ms == pytest.approx(200.0)


def test_tracking_loss_resets_fixation() -> None:
    detector = FixationDetector(stability_ms=100.0, radius=0.04)
    detector.update(0.5, 0.5, 1.0, _ns(0))
    assert detector.update(0.5, 0.5, 1.0, _ns(100)) is not None
    detector.reset()
    assert detector.update(0.5, 0.5, 1.0, _ns(200)) is None


def test_blink_is_emitted_when_eyes_reopen_with_measured_duration() -> None:
    detector = BlinkDetector()
    assert detector.update(True, 0.7, _ns(1000)) is None
    assert detector.update(True, 0.8, _ns(1060)) is None
    event = detector.update(False, 0.9, _ns(1125))
    assert event is not None
    assert event.duration_ms == pytest.approx(125.0)
    assert event.confidence == pytest.approx(0.8)
    assert detector.update(False, 1.0, _ns(1200)) is None


def test_cancelled_blink_does_not_emit() -> None:
    detector = BlinkDetector()
    detector.update(True, 0.8, _ns(0))
    detector.reset()
    assert detector.update(False, 0.8, _ns(100)) is None
