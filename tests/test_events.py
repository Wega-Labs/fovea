from dataclasses import FrozenInstanceError

import pytest

from fovea import GazePoint, TrackingState, TrackingStatus


def test_gaze_point_is_immutable() -> None:
    point = GazePoint(x=0.25, y=0.75, confidence=0.9, timestamp_ns=1)

    with pytest.raises(FrozenInstanceError):
        point.x = 0.5  # type: ignore[misc]


def test_tracking_detail_defaults_to_empty() -> None:
    state = TrackingState(
        status=TrackingStatus.ACTIVE,
        confidence=1.0,
        timestamp_ns=1,
    )
    assert state.detail == ""


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_confidence_must_be_normalized(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        TrackingState(
            status=TrackingStatus.ACTIVE,
            confidence=confidence,
            timestamp_ns=1,
        )
