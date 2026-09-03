"""Tests for the shared per-frame processor and event vocabulary v2 wiring."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fovea.events import (
    Blink,
    CalibrationDone,
    Diagnostics,
    DoubleBlink,
    Eye,
    Fixation,
    FoveaEvent,
    GazePoint,
    LongBlink,
    Saccade,
    TargetLeave,
    TrackingState,
    TrackingStatus,
    Wink,
)
from fovea.webcam.calibration import CalibrationIdentity, fit_ridge, save_model
from fovea.webcam.engine import GazeEngine, GazeSettings
from fovea.webcam.event_source import WebcamEventSource
from fovea.webcam.features import LEFT_IRIS_RING
from fovea.webcam.frame_processor import GazeFrameProcessor
from fovea.webcam.targeting import TargetRect
from tests.synth import SyntheticLandmark, synthetic_landmarks
from tests.test_gaze_pipeline import _features

_FRAME_NS = 33_333_333
_T0 = 1_000_000_000


def _timestamp(index: int) -> int:
    return _T0 + index * _FRAME_NS


def _processor(
    tmp_path: Path,
    *,
    targets: Sequence[TargetRect] = (),
    **overrides: Any,
) -> GazeFrameProcessor:
    settings = GazeSettings(calibration_path=str(tmp_path / "missing.json"), **overrides)
    return GazeFrameProcessor(GazeEngine(settings), settings, targets=targets)


def _drive(
    processor: GazeFrameProcessor,
    frames: Sequence[Sequence[Any]],
    *,
    start_index: int = 0,
    diagnostics_at: frozenset[int] = frozenset(),
) -> list[tuple[FoveaEvent, ...]]:
    per_frame = []
    for index, landmarks in enumerate(frames, start=start_index):
        per_frame.append(
            processor.events_for_frame(
                landmarks,
                640.0,
                480.0,
                1 / 30,
                30.0,
                _timestamp(index),
                None,
                diagnostics_due=index in diagnostics_at,
            )
        )
    return per_frame


def _names(events: Sequence[FoveaEvent]) -> list[str]:
    return [type(event).__name__ for event in events]


def _flatten(per_frame: Sequence[Sequence[FoveaEvent]]) -> list[FoveaEvent]:
    return [event for frame_events in per_frame for event in frame_events]


def _quick_calibration(tmp_path: Path) -> GazeFrameProcessor:
    processor = _processor(tmp_path, samples_per_point=1, min_good_samples=1, settle_frames=0)
    processor.engine.start_calibration()
    return processor


# --- Step 0: one processor for live capture and replay -------------------------


def test_frame_event_order_matches_the_live_contract(tmp_path: Path) -> None:
    processor = _quick_calibration(tmp_path)
    open_face = synthetic_landmarks()
    wizard = _drive(processor, [open_face] * 10, diagnostics_at=frozenset({9}))
    assert _names(wizard[0]) == ["CalibrationCue", "TrackingState"]
    assert _names(wizard[9]) == ["TrackingState", "Diagnostics", "CalibrationDone"]

    blink = synthetic_landmarks(blink=True)
    live = _drive(
        processor,
        [open_face] * 2 + [blink] * 3 + [open_face] * 12,
        start_index=10,
        diagnostics_at=frozenset({15}),
    )
    assert _names(live[0]) == ["TrackingState", "GazePoint"]
    assert _names(live[2]) == ["TrackingState"]
    assert _names(live[5]) == ["TrackingState", "Diagnostics", "Blink", "GazePoint"]
    fixation_frames = [frame_events for frame_events in live if "Fixation" in _names(frame_events)]
    assert fixation_frames
    assert all(_names(frame_events)[-1] == "Fixation" for frame_events in fixation_frames)


def test_lost_frame_emits_lost_state_and_optional_diagnostics(tmp_path: Path) -> None:
    processor = _processor(tmp_path)
    assert processor.events_for_lost_frame(24.0, _timestamp(0)) == (
        TrackingState(status=TrackingStatus.LOST, confidence=0.0, timestamp_ns=_timestamp(0)),
    )
    events = processor.events_for_lost_frame(24.0, _timestamp(1), diagnostics_due=True)
    assert _names(events) == ["TrackingState", "Diagnostics"]
    assert events[1] == Diagnostics(24.0, 0.0, 0.0, 0.0, 0.0, _timestamp(1))


def test_targets_replaced_between_frames_keep_queued_leave_and_current_match(
    tmp_path: Path,
) -> None:
    button = TargetRect("button", 0.4, 0.4, 0.2, 0.2)
    processor = _processor(tmp_path, targets=(button,))
    first = _drive(processor, [synthetic_landmarks()])[0]
    assert _names(first) == ["TrackingState", "GazePoint", "TargetEnter", "DwellProgress"]
    gaze = next(event for event in first if isinstance(event, GazePoint))
    assert gaze.target_id == "button"
    assert gaze.snapped_x == pytest.approx(0.5)

    assert processor.replace_targets((button,), _timestamp(0) + 1) == ()
    moved = TargetRect("button", 0.45, 0.4, 0.2, 0.2)
    assert processor.replace_targets((moved,), _timestamp(0) + 2) == (
        TargetLeave("button", _timestamp(0) + 2),
    )
    second = _drive(processor, [synthetic_landmarks()], start_index=1)[0]
    assert _names(second) == ["TrackingState", "GazePoint", "TargetEnter", "DwellProgress"]
    assert next(event for event in second if isinstance(event, GazePoint)).snapped_x == (
        pytest.approx(0.55)
    )


class _FakeCamera:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        return None

    def connect(self) -> None:
        return None

    def read(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def disconnect(self) -> None:
        return None


class _FakeEstimator:
    landmarks: Sequence[Any] = ()

    def __init__(self, **_kwargs: object) -> None:
        return None

    def process(self, _frame: object) -> object:
        return type("Obs", (), {"landmarks": _FakeEstimator.landmarks, "blendshapes": {}})()

    def close(self) -> None:
        return None


def _save_centered_model(path: Path) -> None:
    rows = [
        _features(0.5, 0.5).vector(),
        _features(0.3, 0.3).vector(),
        _features(0.7, 0.7).vector(),
    ]
    model = fit_ridge(
        rows,
        [(0.5, 0.5), (0.2, 0.2), (0.8, 0.8)],
        {"n": 3},
        {"n": "GOOD"},
        identity=CalibrationIdentity(None, 1280, 720, 0, 640, 480),
    )
    save_model(model, path)


def test_webcam_source_delegates_target_replacement_to_the_processor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calibration_path = tmp_path / "gaze_calibration.json"
    _save_centered_model(calibration_path)
    _FakeEstimator.landmarks = synthetic_landmarks()
    monkeypatch.setattr("fovea.webcam.event_source.Webcam", _FakeCamera)
    monkeypatch.setattr("fovea.webcam.event_source.FaceLandmarkEstimator", _FakeEstimator)

    source = WebcamEventSource(
        GazeSettings(calibration_path=str(calibration_path)),
        max_frames=3,
        show_calibration=False,
    )
    source.set_targets((TargetRect("button", 0.4, 0.4, 0.2, 0.2),))
    iterator = source.events()
    first_frame = [next(iterator) for _ in range(4)]
    assert _names(first_frame) == ["TrackingState", "GazePoint", "TargetEnter", "DwellProgress"]
    assert isinstance(first_frame[1], GazePoint)
    assert first_frame[1].target_id == "button"

    source.set_targets((TargetRect("button", 0.45, 0.4, 0.2, 0.2),))
    rest = list(iterator)
    assert isinstance(rest[0], TargetLeave)
    assert _names(rest[1:5]) == ["TrackingState", "GazePoint", "TargetEnter", "DwellProgress"]
    assert source._processor is None


def test_webcam_source_keeps_diagnostics_after_tracking_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calibration_path = tmp_path / "gaze_calibration.json"
    _save_centered_model(calibration_path)
    _FakeEstimator.landmarks = synthetic_landmarks(face_width=0.08)
    monkeypatch.setattr("fovea.webcam.event_source.Webcam", _FakeCamera)
    monkeypatch.setattr("fovea.webcam.event_source.FaceLandmarkEstimator", _FakeEstimator)

    source = WebcamEventSource(
        GazeSettings(calibration_path=str(calibration_path)),
        max_frames=2,
        show_calibration=False,
        diagnostics=True,
    )
    events = list(source.events())
    assert _names(events[:3]) == ["TrackingState", "Diagnostics", "GazePoint"]
    assert _names(events[3:]) == ["TrackingState", "GazePoint"]
    assert all(not event.pursuit for event in events if isinstance(event, GazePoint))


# --- event vocabulary v2 --------------------------------------------------------


def test_wink_is_labelled_by_landmark_topology_and_closure_frames_never_saccade(
    tmp_path: Path,
) -> None:
    processor = _processor(tmp_path, smoothing_alpha=1.0)
    open_face = synthetic_landmarks()
    closed = list(synthetic_landmarks(left_closed=True))
    for index in LEFT_IRIS_RING:
        # A closed eye corrupts its iris landmarks; make the raw gaze jump hard.
        closed[index] = SyntheticLandmark(closed[index].x + 0.04, closed[index].y)
    per_frame = _drive(processor, [open_face] * 12 + [closed] * 8 + [open_face] * 4)
    events = _flatten(per_frame)

    winks = [event for event in events if isinstance(event, Wink)]
    assert len(winks) == 1
    assert winks[0].eye is Eye.LEFT
    assert winks[0].duration_ms == pytest.approx(8 * _FRAME_NS / 1e6)
    assert winks[0].timestamp_ns == _timestamp(20)

    gaze = [event for event in events if isinstance(event, GazePoint)]
    closure_speeds = [
        abs(later.x - earlier.x) / (_FRAME_NS / 1e9) for earlier, later in pairwise(gaze[11:20])
    ]
    assert max(closure_speeds) > 1.5
    assert not [
        event
        for event in events
        if isinstance(event, Saccade) and event.timestamp_ns <= _timestamp(20)
    ]
    assert not any(event.pursuit for event in gaze[:20])
    assert not [event for event in events if isinstance(event, Blink | LongBlink | DoubleBlink)]


def test_saccade_is_reported_before_the_landing_gaze_point(tmp_path: Path) -> None:
    processor = _processor(tmp_path, smoothing_alpha=1.0)
    per_frame = _drive(
        processor,
        [synthetic_landmarks()] * 14 + [synthetic_landmarks(gaze_dx=0.25)] * 16,
    )
    saccades = [event for event in _flatten(per_frame) if isinstance(event, Saccade)]
    assert len(saccades) == 1
    saccade = saccades[0]
    assert saccade.from_x == pytest.approx(0.5, abs=0.01)
    assert saccade.to_x > saccade.from_x + 0.2
    assert saccade.amplitude == pytest.approx(abs(saccade.to_x - saccade.from_x), abs=1e-6)
    landing = next(frame_events for frame_events in per_frame if saccade in frame_events)
    assert _names(landing)[:3] == ["TrackingState", "Saccade", "GazePoint"]
    assert not any("Fixation" in _names(frame_events) for frame_events in per_frame[14:18])


def test_fixation_never_co_emits_with_moving_or_pursuit_samples(tmp_path: Path) -> None:
    processor = _processor(tmp_path)
    hold = [synthetic_landmarks()] * 14
    ramp = [synthetic_landmarks(gaze_dx=0.012 * step) for step in range(1, 20)]
    settle = [ramp[-1]] * 16
    per_frame = _drive(processor, hold + ramp + settle)

    assert any(isinstance(event, Fixation) for event in _flatten(per_frame[:14]))
    gaze = [
        next(event for event in frame_events if isinstance(event, GazePoint))
        for frame_events in per_frame
    ]
    speeds = [0.0] + [
        float(np.hypot(later.x - earlier.x, later.y - earlier.y)) / (_FRAME_NS / 1e9)
        for earlier, later in pairwise(gaze)
    ]
    first_moving = next(index for index, speed in enumerate(speeds) if speed >= 0.3)
    pursuit_frames = [index for index, point in enumerate(gaze) if point.pursuit]
    assert pursuit_frames
    assert min(pursuit_frames) >= first_moving + 3
    last_moving = max(index for index, speed in enumerate(speeds) if speed >= 0.3)
    assert not any(
        isinstance(event, Fixation) for event in _flatten(per_frame[first_moving : last_moving + 1])
    )
    assert any(isinstance(event, Fixation) for event in _flatten(per_frame[last_moving + 1 :]))


def test_wizard_frames_and_the_boundary_blink_never_trigger(tmp_path: Path) -> None:
    processor = _quick_calibration(tmp_path)
    open_face = synthetic_landmarks()
    blink = synthetic_landmarks(blink=True)
    frames = (
        [open_face] * 9  # targets 0-8
        + [blink] * 3  # closure starts inside the wizard (samples rejected)
        + [open_face]  # frame 12: reopens, samples target 9, completes the wizard
        + [open_face]  # frame 13: first live frame with both eyes open
        + [blink] * 3  # frames 14-16
        + [open_face]  # frame 17: first live blink (would pair with the wizard blink)
        + [blink] * 3  # frames 18-20
        + [open_face]  # frame 21: pairs with the frame-17 blink
    )
    per_frame = _drive(processor, frames)
    events = _flatten(per_frame)

    assert any(isinstance(event, CalibrationDone) for event in per_frame[12])
    assert processor.engine.wizard is None
    blinks = [event for event in events if isinstance(event, Blink)]
    assert [blink_event.timestamp_ns for blink_event in blinks] == [
        _timestamp(12),
        _timestamp(17),
        _timestamp(21),
    ]
    doubles = [event for event in events if isinstance(event, DoubleBlink)]
    assert doubles == [DoubleBlink(_timestamp(21))]
    assert not [event for event in events if isinstance(event, LongBlink | Wink)]


def test_unilateral_closure_held_across_the_wizard_exit_is_not_a_wink(tmp_path: Path) -> None:
    processor = _quick_calibration(tmp_path)
    open_face = synthetic_landmarks()
    left_closed = synthetic_landmarks(left_closed=True)
    frames = (
        [open_face] * 9  # targets 0-8
        + [left_closed]  # frame 9: samples target 9 (not a blink) and completes the wizard
        + [left_closed] * 6  # frames 10-15: still closed after the wizard
        + [open_face]  # frame 16: both eyes open, triggers become live afterwards
        + [left_closed] * 8  # frames 17-24: a genuine wink
        + [open_face]  # frame 25
    )
    per_frame = _drive(processor, frames)
    events = _flatten(per_frame)

    assert any(isinstance(event, CalibrationDone) for event in per_frame[9])
    winks = [event for event in events if isinstance(event, Wink)]
    assert [wink.timestamp_ns for wink in winks] == [_timestamp(25)]
    assert winks[0].eye is Eye.LEFT
    assert winks[0].duration_ms == pytest.approx(8 * _FRAME_NS / 1e6)


def test_tracking_loss_resets_trigger_sequences(tmp_path: Path) -> None:
    processor = _processor(tmp_path)
    open_face = synthetic_landmarks()
    blink = synthetic_landmarks(blink=True)
    frames = (
        [open_face] * 3
        + [blink] * 3
        + [open_face]
        + [None]
        + [open_face]
        + [blink] * 3
        + [open_face]
    )
    per_frame = _drive(processor, frames)  # type: ignore[arg-type]
    events = _flatten(per_frame)
    assert len([event for event in events if isinstance(event, Blink)]) == 2
    assert not [event for event in events if isinstance(event, DoubleBlink)]
