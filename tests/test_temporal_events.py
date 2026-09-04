from __future__ import annotations

import math
from itertools import pairwise

import pytest

from fovea.events import Blink, DoubleBlink, Eye, LongBlink, Wink
from fovea.webcam.features import extract_features
from fovea.webcam.temporal import (
    BlinkDetector,
    BlinkTriggerDetector,
    FixationDetector,
    SaccadeDetector,
    VelocityUpdate,
    WinkDetector,
)
from tests.synth import synthetic_landmarks


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


def test_fixation_emits_with_unaligned_frame_timestamps() -> None:
    """Real capture never lands a sample exactly on the stability cutoff."""
    detector = FixationDetector(stability_ms=300.0, radius=0.04)
    frame_ns = 33_333_333
    events = [
        detector.update(0.5, 0.5, 1.0, 1_000_000_000 + index * frame_ns) for index in range(20)
    ]
    emitted = [event for event in events if event is not None]
    assert emitted
    assert emitted[0].duration_ms >= 300.0
    assert events[10] is not None
    assert all(event is None for event in events[:9])


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


# --- saccades and pursuit -----------------------------------------------------

_HOLD = [(milliseconds, 0.2, 0.2) for milliseconds in range(0, 101, 20)]
_FLIGHT = [(120, 0.45, 0.4), (140, 0.7, 0.6)]
_LANDING = [(milliseconds, 0.7, 0.6) for milliseconds in range(160, 301, 20)]


def _saccades(**overrides: float) -> SaccadeDetector:
    settings: dict[str, float] = {
        "velocity_threshold": 1.5,
        "pursuit_velocity": 0.3,
        "pursuit_ms": 100.0,
        "pursuit_coherence": 0.7,
    }
    settings.update(overrides)
    return SaccadeDetector(**settings)


def _drive(
    detector: SaccadeDetector,
    samples: list[tuple[float, float, float]],
) -> list[VelocityUpdate]:
    return [detector.update(x, y, _ns(milliseconds)) for milliseconds, x, y in samples]


def test_hold_jump_hold_emits_exactly_one_saccade_with_the_landing_sample() -> None:
    detector = _saccades()
    updates = _drive(detector, _HOLD + _FLIGHT + _LANDING)
    saccades = [update.saccade for update in updates if update.saccade is not None]
    assert len(saccades) == 1
    saccade = saccades[0]
    assert saccade.from_x == pytest.approx(0.2)
    assert saccade.from_y == pytest.approx(0.2)
    assert saccade.to_x == pytest.approx(0.7)
    assert saccade.to_y == pytest.approx(0.6)
    assert saccade.amplitude == pytest.approx(math.hypot(0.5, 0.4))
    assert saccade.duration_ms == pytest.approx(60.0)
    assert saccade.timestamp_ns == _ns(160)

    landing_index = len(_HOLD) + len(_FLIGHT)
    assert updates[landing_index].saccade is saccade
    flight = updates[len(_HOLD) : landing_index]
    assert [update.saccading for update in flight] == [True, True]
    assert all(update.moving and not update.pursuit for update in flight)
    assert not updates[landing_index].saccading
    assert not any(update.pursuit for update in updates)
    assert not any(update.moving for update in updates[: len(_HOLD)])


def test_slow_drift_below_pursuit_velocity_is_neither_moving_nor_a_saccade() -> None:
    detector = _saccades()
    samples = [
        (milliseconds, 0.2 + 0.004 * (milliseconds // 20), 0.5)
        for milliseconds in range(0, 401, 20)
    ]
    updates = _drive(detector, samples)
    assert not any(update.moving for update in updates)
    assert not any(update.pursuit for update in updates)
    assert all(update.saccade is None for update in updates)


def test_reset_mid_flight_drops_the_partial_saccade() -> None:
    detector = _saccades()
    _drive(detector, _HOLD + _FLIGHT[:1])
    detector.reset()
    updates = _drive(detector, _FLIGHT[1:] + _LANDING)
    assert all(update.saccade is None for update in updates)
    assert not any(update.saccading for update in updates)


def test_non_advancing_timestamps_are_ignored_mid_flight() -> None:
    detector = _saccades()
    _drive(detector, _HOLD)
    in_flight = detector.update(0.45, 0.4, _ns(120))
    assert in_flight.saccading

    duplicate = detector.update(0.45, 0.4, _ns(120))
    stale = detector.update(0.2, 0.2, _ns(110))
    assert duplicate.saccading and duplicate.saccade is None
    assert stale.saccading and stale.saccade is None

    landed = detector.update(0.45, 0.4, _ns(140))
    assert landed.saccade is not None
    assert landed.saccade.from_x == pytest.approx(0.2)
    assert landed.saccade.to_x == pytest.approx(0.45)
    assert landed.saccade.duration_ms == pytest.approx(40.0)


def test_non_finite_sample_resets_without_emitting() -> None:
    detector = _saccades()
    _drive(detector, _HOLD + _FLIGHT[:1])
    assert detector.update(math.nan, 0.5, _ns(140)) == VelocityUpdate(False, False, False)
    landed = _drive(detector, [(160, 0.7, 0.6), (180, 0.7, 0.6)])
    assert all(update.saccade is None and not update.saccading for update in landed)


@pytest.mark.parametrize(
    "overrides",
    [
        {"pursuit_velocity": 1.5},
        {"pursuit_velocity": 0.0},
        {"pursuit_velocity": 2.0},
        {"pursuit_ms": -1.0},
        {"pursuit_coherence": 1.5},
        {"pursuit_coherence": -0.1},
        {"velocity_threshold": math.nan},
        {"velocity_threshold": math.inf},
    ],
)
def test_invalid_saccade_settings_are_rejected(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        _saccades(**overrides)


def test_velocity_exactly_at_threshold_starts_a_saccade() -> None:
    detector = _saccades()
    detector.update(0.25, 0.5, _ns(0))
    # 0.1875 display units over exactly 0.125 s is exactly 1.5 units/s.
    at_threshold = detector.update(0.4375, 0.5, _ns(125))
    assert at_threshold.saccading
    landed = detector.update(0.4375, 0.5, _ns(250))
    assert landed.saccade is not None
    assert landed.saccade.amplitude == pytest.approx(0.1875)
    assert landed.saccade.duration_ms == pytest.approx(250.0)


def test_velocity_just_below_threshold_is_moving_but_not_a_saccade() -> None:
    detector = _saccades()
    detector.update(0.25, 0.5, _ns(0))
    below = detector.update(0.4375, 0.5, _ns(126))
    assert below.moving
    assert not below.saccading
    # One straight in-band step longer than pursuit_ms already qualifies as pursuit.
    assert below.pursuit
    assert detector.update(0.4375, 0.5, _ns(252)).saccade is None


def test_coherent_in_band_motion_qualifies_as_pursuit_after_pursuit_ms() -> None:
    detector = _saccades()
    samples = [(0.0, 0.2, 0.5)] + [(20.0 * k, 0.2 + 0.012 * k, 0.5) for k in range(1, 16)]
    updates = _drive(detector, samples)
    assert all(update.moving and not update.saccading for update in updates[1:])
    assert [update.pursuit for update in updates[1:6]] == [False, False, False, False, True]
    assert all(update.pursuit for update in updates[5:])
    assert all(update.saccade is None for update in updates)


def test_in_band_oscillation_is_moving_but_never_pursuit() -> None:
    detector = _saccades()
    samples = [(20.0 * k, 0.5 + (0.012 if k % 2 else 0.0), 0.5) for k in range(30)]
    updates = _drive(detector, samples)
    assert all(update.moving for update in updates[1:])
    assert not any(update.pursuit for update in updates)


def test_jittery_hold_is_not_moving() -> None:
    detector = _saccades()
    samples = [(20.0 * k, 0.5 + (0.002 if k % 2 else -0.002), 0.5) for k in range(20)]
    updates = _drive(detector, samples)
    assert not any(update.moving for update in updates)
    assert not any(update.pursuit for update in updates)


def test_saccade_samples_are_moving_but_not_pursuit() -> None:
    detector = _saccades()
    updates = _drive(detector, _HOLD + _FLIGHT)
    flight = updates[len(_HOLD) :]
    assert all(update.moving and update.saccading and not update.pursuit for update in flight)


def test_saccade_clears_the_pursuit_band() -> None:
    detector = _saccades()
    ramp = [(20.0 * k, 0.2 + 0.012 * k, 0.5) for k in range(8)]
    assert _drive(detector, ramp)[-1].pursuit
    jump = detector.update(0.8, 0.5, _ns(160))
    assert jump.saccading and not jump.pursuit
    landing = detector.update(0.812, 0.5, _ns(180))
    assert landing.saccade is not None
    assert landing.moving and not landing.pursuit
    later = _drive(detector, [(180.0 + 20.0 * k, 0.812 + 0.012 * k, 0.5) for k in range(1, 6)])
    assert [update.pursuit for update in later] == [False, False, False, False, True]


# --- blink triggers -----------------------------------------------------------


def _blink(duration_ms: float, end_ms: float) -> Blink:
    return Blink(Eye.BOTH, duration_ms, 0.8, _ns(end_ms))


def _triggers(**overrides: float | int) -> BlinkTriggerDetector:
    settings: dict[str, float | int] = {
        "long_blink_ms": 600.0,
        "long_blink_factor": 2.0,
        "natural_blink_window": 30,
        "double_blink_ms": 500.0,
    }
    settings.update(overrides)
    return BlinkTriggerDetector(**settings)  # type: ignore[arg-type]


def test_natural_blink_emits_no_trigger() -> None:
    assert _triggers().update(_blink(120, 1000)) == ()


def test_blink_at_or_over_the_threshold_is_a_long_blink() -> None:
    assert _triggers().update(_blink(700, 1000)) == (LongBlink(700.0, _ns(1000)),)
    assert _triggers().update(_blink(600, 1000)) == (LongBlink(600.0, _ns(1000)),)
    assert _triggers().update(_blink(599, 1000)) == ()


def test_two_natural_blinks_within_the_gap_emit_one_double_blink() -> None:
    detector = _triggers()
    assert detector.update(_blink(120, 1000)) == ()
    assert detector.update(_blink(120, 1420)) == (DoubleBlink(_ns(1420)),)


def test_triple_yields_one_double_blink_and_four_yield_two() -> None:
    detector = _triggers()
    results = [detector.update(_blink(120, end_ms)) for end_ms in (1000, 1420, 1840, 2260)]
    assert results == [(), (DoubleBlink(_ns(1420)),), (), (DoubleBlink(_ns(2260)),)]


def test_natural_long_natural_never_pairs() -> None:
    detector = _triggers()
    assert detector.update(_blink(120, 1000)) == ()
    assert detector.update(_blink(700, 1800)) == (LongBlink(700.0, _ns(1800)),)
    assert detector.update(_blink(120, 2000)) == ()


def test_negative_gap_does_not_pair() -> None:
    detector = _triggers()
    detector.update(_blink(120, 1000))
    assert detector.update(_blink(200, 1100)) == ()


def test_gap_exactly_double_blink_ms_pairs_and_one_more_millisecond_does_not() -> None:
    detector = _triggers()
    detector.update(_blink(120, 1000))
    assert detector.update(_blink(120, 1620)) == (DoubleBlink(_ns(1620)),)

    detector = _triggers()
    detector.update(_blink(120, 1000))
    assert detector.update(_blink(120, 1621)) == ()


def test_long_blink_threshold_adapts_upward_and_excludes_long_blinks() -> None:
    detector = _triggers()
    assert detector.long_blink_threshold_ms == 600.0
    for index in range(5):
        assert detector.update(_blink(350, 5000 * (index + 1))) == ()
    assert detector.long_blink_threshold_ms == pytest.approx(700.0)
    assert detector.update(_blink(650, 30_000)) == ()
    assert detector.update(_blink(800, 35_000)) == (LongBlink(800.0, _ns(35_000)),)
    assert detector.long_blink_threshold_ms == pytest.approx(700.0)


def test_long_blink_threshold_never_drops_below_the_floor() -> None:
    detector = _triggers()
    for index in range(5):
        detector.update(_blink(100, 5000 * (index + 1)))
    assert detector.long_blink_threshold_ms == 600.0


def test_reset_clears_the_pending_blink_but_keeps_learned_durations() -> None:
    detector = _triggers()
    for index in range(5):
        detector.update(_blink(350, 5000 * (index + 1)))
    assert detector.update(_blink(120, 40_000)) == ()
    detector.reset()
    assert detector.long_blink_threshold_ms == pytest.approx(700.0)
    assert detector.update(_blink(120, 40_300)) == ()


@pytest.mark.parametrize(
    "overrides",
    [
        {"natural_blink_window": 4},
        {"long_blink_factor": 0.5},
        {"long_blink_ms": -1.0},
        {"double_blink_ms": -1.0},
        {"double_blink_ms": math.nan},
    ],
)
def test_invalid_blink_trigger_settings_are_rejected(overrides: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        _triggers(**overrides)


# --- winks --------------------------------------------------------------------


def _winks(min_ms: float = 120.0, max_ms: float = 700.0) -> WinkDetector:
    return WinkDetector(min_ms=min_ms, max_ms=max_ms)


def _drive_wink(
    detector: WinkDetector,
    frames: list[tuple[float, bool | None, bool | None, float]],
) -> list[Wink]:
    events = [
        detector.update(left, right, confidence, _ns(milliseconds))
        for milliseconds, left, right, confidence in frames
    ]
    return [event for event in events if event is not None]


def _closure(
    eye: Eye,
    *,
    start_ms: float = 20.0,
    end_ms: float = 220.0,
    step_ms: float = 20.0,
    confidence: float = 0.9,
) -> list[tuple[float, bool | None, bool | None, float]]:
    frames: list[tuple[float, bool | None, bool | None, float]] = [(0.0, False, False, confidence)]
    milliseconds = start_ms
    while milliseconds < end_ms:
        frames.append((milliseconds, eye is Eye.LEFT, eye is Eye.RIGHT, confidence))
        milliseconds += step_ms
    frames.append((end_ms, False, False, confidence))
    return frames


@pytest.mark.parametrize("eye", [Eye.LEFT, Eye.RIGHT])
def test_single_eye_closure_emits_a_wink_for_that_eye(eye: Eye) -> None:
    events = _drive_wink(_winks(), _closure(eye))
    assert events == [Wink(eye, 200.0, 0.9, _ns(220))]  # type: ignore[arg-type]


def test_wink_confidence_includes_the_reopening_frame() -> None:
    frames = _closure(Eye.LEFT, confidence=0.0)
    frames[-1] = (220.0, False, False, 0.8)
    events = _drive_wink(_winks(), frames)
    assert len(events) == 1
    assert events[0].confidence == pytest.approx(0.8)


def test_natural_blink_onset_and_offset_asymmetry_is_not_a_wink() -> None:
    frames: list[tuple[float, bool | None, bool | None, float]] = [
        (0, False, False, 0.9),
        (20, True, False, 0.9),
        (40, True, True, 0.9),
        (60, True, True, 0.9),
        (80, True, True, 0.9),
        (100, True, False, 0.9),
        (120, False, False, 0.9),
        (400, False, False, 0.9),
    ]
    assert _drive_wink(_winks(), frames) == []


def test_invalid_eye_prevents_and_cancels_winks() -> None:
    frames = [
        (milliseconds, closed, None if closed else False, 0.9)
        for milliseconds, closed, _right, _confidence in _closure(Eye.LEFT)
    ]
    assert _drive_wink(_winks(), frames) == []

    cancelled = _closure(Eye.LEFT)
    cancelled[3] = (60.0, True, None, 0.9)
    assert _drive_wink(_winks(), cancelled) == []


def test_closure_longer_than_max_ms_is_not_a_wink_but_the_next_one_is() -> None:
    frames = _closure(Eye.LEFT, end_ms=780.0)
    frames += _closure(Eye.LEFT, start_ms=820.0, end_ms=1040.0)[1:]
    frames.insert(len(_closure(Eye.LEFT, end_ms=780.0)), (800.0, False, False, 0.9))
    events = _drive_wink(_winks(), frames)
    assert [event.timestamp_ns for event in events] == [_ns(1040)]
    assert events[0].duration_ms == pytest.approx(220.0)


def test_closure_exactly_min_ms_is_a_wink_and_one_millisecond_less_is_not() -> None:
    assert _drive_wink(_winks(), _closure(Eye.RIGHT, end_ms=140.0)) == [
        Wink(Eye.RIGHT, 120.0, 0.9, _ns(140))
    ]
    assert _drive_wink(_winks(), _closure(Eye.RIGHT, end_ms=139.0)) == []


def test_both_eyes_closing_within_min_ms_is_not_a_wink() -> None:
    frames: list[tuple[float, bool | None, bool | None, float]] = [
        (0, False, False, 0.9),
        (20, False, True, 0.9),
        (40, False, True, 0.9),
        (60, True, True, 0.9),
        (200, True, True, 0.9),
        (220, False, False, 0.9),
    ]
    assert _drive_wink(_winks(), frames) == []


@pytest.mark.parametrize(("min_ms", "max_ms"), [(200.0, 100.0), (-1.0, 100.0), (math.nan, 1.0)])
def test_invalid_wink_settings_are_rejected(min_ms: float, max_ms: float) -> None:
    with pytest.raises(ValueError):
        _winks(min_ms, max_ms)


def _natural_blink_state(milliseconds: float) -> tuple[bool, bool]:
    """Left/right closure for a 60 s stream of 20 natural blinks with asymmetry."""
    for index in range(20):
        start = 1000.0 + index * 2900.0
        onset = 20.0 + 10.0 * (index % 3)
        offset = 40.0 - 10.0 * (index % 3)
        both_from = start + onset
        both_until = both_from + 100.0
        if not start <= milliseconds < both_until + offset:
            continue
        leading_left = index % 2 == 0
        if milliseconds < both_from or milliseconds >= both_until:
            return (leading_left, not leading_left)
        return (True, True)
    return (False, False)


def test_sixty_seconds_of_natural_blinks_yield_no_winks() -> None:
    detector = _winks()
    winks = []
    asymmetric_frames = 0
    for milliseconds in range(0, 60_000, 10):
        left, right = _natural_blink_state(float(milliseconds))
        asymmetric_frames += left != right
        wink = detector.update(left, right, 0.8, _ns(milliseconds))
        if wink is not None:
            winks.append(wink)
    assert asymmetric_frames >= 80
    assert winks == []


def test_feature_path_natural_blinks_and_dropouts_yield_no_winks() -> None:
    """Sixty seconds of landmarks at 30 fps through extract_features and WinkDetector."""
    faces = {
        "open": synthetic_landmarks(),
        "left": synthetic_landmarks(left_closed=True),
        "right": synthetic_landmarks(right_closed=True),
        "both": synthetic_landmarks(blink=True),
        "left_invalid": synthetic_landmarks(left_valid=False),
        "right_invalid": synthetic_landmarks(right_valid=False),
    }
    schedule = ["open"] * 1800
    for index in range(20):
        start = 30 + index * 88
        leading = "left" if index % 2 == 0 else "right"
        schedule[start] = leading
        for frame in range(start + 1, start + 4):
            schedule[frame] = "both"
        schedule[start + 4] = leading
    for index in range(8):
        start = 70 + index * 220
        for frame in range(start, start + 3):
            schedule[frame] = "left_invalid" if index % 2 == 0 else "right_invalid"

    detector = _winks()
    winks = []
    asymmetric_frames = blink_frames = invalid_frames = 0
    for index, state in enumerate(schedule):
        features = extract_features(faces[state], 640.0, 480.0, 0.16, 0.12, 35.0)
        left = features.left.ear < 0.16 if features.left.valid else None
        right = features.right.ear < 0.16 if features.right.valid else None
        asymmetric_frames += left is not None and right is not None and left != right
        blink_frames += features.blink
        invalid_frames += left is None or right is None
        timestamp_ns = round(1_000_000_000 + index * (1_000_000_000 / 30))
        wink = detector.update(left, right, 0.8, timestamp_ns)
        if wink is not None:
            winks.append(wink)

    assert asymmetric_frames >= 40
    assert blink_frames >= 60
    assert invalid_frames >= 24
    assert winks == []
