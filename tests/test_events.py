from dataclasses import FrozenInstanceError

import pytest

from fovea import Diagnostics, Eye, GazePoint, TrackingState, TrackingStatus, Wink


def test_gaze_point_is_immutable() -> None:
    point = GazePoint(x=0.25, y=0.75, confidence=0.9, timestamp_ns=1)

    with pytest.raises(FrozenInstanceError):
        point.x = 0.5  # type: ignore[misc]


def test_gaze_point_target_fields_default_to_none() -> None:
    point = GazePoint(x=0.25, y=0.75, confidence=0.9, timestamp_ns=1)
    assert point.target_id is None
    assert point.snapped_x is None
    assert point.snapped_y is None


def test_gaze_point_pursuit_defaults_to_false() -> None:
    assert GazePoint(x=0.25, y=0.75, confidence=0.9, timestamp_ns=1).pursuit is False


def test_wink_rejects_both_eyes() -> None:
    with pytest.raises(ValueError, match="left or right"):
        Wink(eye=Eye.BOTH, duration_ms=150.0, confidence=0.8, timestamp_ns=1)  # type: ignore[arg-type]


def test_gaze_point_latency_defaults_to_unknown() -> None:
    point = GazePoint(x=0.25, y=0.75, confidence=0.9, timestamp_ns=1)
    assert point.latency_ms is None


def test_diagnostics_latency_fields_default_to_unknown_and_zero_drops() -> None:
    diagnostics = Diagnostics(30.0, 8.2, 0.17, -18.0, 2.0, 1)
    assert diagnostics.latency_p50_ms is None
    assert diagnostics.latency_p95_ms is None
    assert diagnostics.dropped_frames == 0


def test_tracking_detail_defaults_to_empty() -> None:
    state = TrackingState(
        status=TrackingStatus.ACTIVE,
        confidence=1.0,
        timestamp_ns=1,
    )
    assert state.detail == ""


def test_diagnostics_is_an_immutable_public_event() -> None:
    diagnostics = Diagnostics(30.0, 8.2, 0.17, -18.0, 2.0, 1)
    with pytest.raises(FrozenInstanceError):
        diagnostics.fps = 20.0  # type: ignore[misc]


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_confidence_must_be_normalized(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        TrackingState(
            status=TrackingStatus.ACTIVE,
            confidence=confidence,
            timestamp_ns=1,
        )
