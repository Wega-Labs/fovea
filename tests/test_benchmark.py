from __future__ import annotations

import pytest

from fovea.benchmark import (
    BenchmarkConfig,
    distribution,
    summarize_accuracy,
    summarize_jitter,
    summarize_yaw,
)
from fovea.events import (
    Diagnostics,
    GazePoint,
    GazeTestDone,
    GazeTestPoint,
    TrackingState,
    TrackingStatus,
)


def test_distribution_uses_interpolated_p95() -> None:
    summary = distribution([0.0, 1.0, 2.0, 3.0, 4.0])
    assert summary == {
        "n": 5,
        "mean": 2.0,
        "median": 2.0,
        "p95": pytest.approx(3.8),
        "max": 4.0,
    }


def test_accuracy_reports_normalized_and_physical_angular_error() -> None:
    point = GazeTestPoint(0.5, 0.5, 0.56, 0.5, 0.06)
    report = GazeTestDone(1, 0.06, 0.06, 0.06, (point,), 1)
    summary = summarize_accuracy(
        report,
        distance_cm=60.0,
        screen_width_cm=30.0,
        screen_height_cm=20.0,
    )
    normalized = summary["normalized_error"]
    angular = summary["angular_error_degrees"]
    assert isinstance(normalized, dict)
    assert isinstance(angular, dict)
    assert normalized["median"] == pytest.approx(0.06)
    assert angular["median"] == pytest.approx(1.7184, abs=0.001)
    assert summary["passes_public_target"] is True


def test_jitter_is_measured_around_the_median_point() -> None:
    points = [
        GazePoint(0.49, 0.50, 0.9, 1),
        GazePoint(0.50, 0.49, 0.9, 2),
        GazePoint(0.51, 0.50, 0.9, 3),
        GazePoint(0.50, 0.51, 0.9, 4),
    ]
    summary = summarize_jitter(points)
    assert summary["center"] == {"x": 0.5, "y": 0.5}
    assert summary["span_x"] == pytest.approx(0.02)
    assert summary["span_y"] == pytest.approx(0.02)


def test_yaw_summary_reports_tracking_robustness() -> None:
    diagnostics = [
        Diagnostics(30.0, 8.0, 0.2, -18.0, 0.0, 1),
        Diagnostics(30.0, 9.0, 0.2, -21.0, 0.0, 2),
    ]
    tracking = [
        TrackingState(TrackingStatus.ACTIVE, 0.9, 1),
        TrackingState(TrackingStatus.UNCERTAIN, 0.5, 2),
    ]
    summary = summarize_yaw(diagnostics, tracking, -20.0)
    assert summary["reached_target"] is True
    assert summary["active_tracking_rate"] == pytest.approx(0.5)


def test_benchmark_config_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="screen dimensions"):
        BenchmarkConfig(0.0, 20.0, 640, 480, "camera", "office", "none")
