from __future__ import annotations

import itertools
from collections.abc import Iterator

import pytest

from fovea.benchmark import (
    BenchmarkConfig,
    distribution,
    percentile,
    run_live_benchmark,
    summarize_accuracy,
    summarize_jitter,
    summarize_yaw,
)
from fovea.events import (
    CalibrationDone,
    Diagnostics,
    FoveaEvent,
    GazePoint,
    GazeTestDone,
    GazeTestPoint,
    TrackingState,
    TrackingStatus,
)
from fovea.util import percentile as util_percentile


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


def test_percentile_is_the_shared_util_helper() -> None:
    assert percentile is util_percentile
    assert util_percentile([0.0, 1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.8)
    assert util_percentile([], 0.5) is None


def test_benchmark_config_rejects_non_positive_fps_cap() -> None:
    with pytest.raises(ValueError, match="max_fps"):
        BenchmarkConfig(30.0, 20.0, 640, 480, "camera", "office", "none", max_fps=0.0)


class _ScriptedSource:
    def __init__(self, events: list[FoveaEvent]) -> None:
        self._events = events
        self.calibration_starts = 0
        self.test_starts = 0

    def events(self) -> Iterator[FoveaEvent]:
        yield from self._events

    def start_calibration(self) -> None:
        self.calibration_starts += 1

    def start_gaze_test(self) -> None:
        self.test_starts += 1


def _gaze_test(timestamp_ns: int) -> GazeTestDone:
    point = GazeTestPoint(0.5, 0.5, 0.53, 0.52, 0.04)
    return GazeTestDone(1, 0.04, 0.04, 0.04, (point,), timestamp_ns)


def test_live_benchmark_reports_end_to_end_latency_and_fps_cap() -> None:
    def diagnostics(timestamp_ns: int) -> Diagnostics:
        return Diagnostics(30.0, 5.0, 0.2, -18.0, 0.0, timestamp_ns)

    script: list[FoveaEvent] = [
        GazePoint(0.5, 0.5, 0.9, 1, latency_ms=10.0),
        diagnostics(2),
        CalibrationDone(5, 0.76, 0.08, 3),
        GazePoint(0.5, 0.5, 0.9, 4, latency_ms=20.0),
        _gaze_test(5),
        GazePoint(0.5, 0.5, 0.9, 6, latency_ms=None),
        _gaze_test(7),
        diagnostics(8),
        _gaze_test(9),
        GazePoint(0.5, 0.5, 0.9, 10, latency_ms=30.0),  # jitter phase
        diagnostics(11),  # yaw left
        TrackingState(TrackingStatus.ACTIVE, 0.9, 12),  # yaw right
        GazePoint(0.5, 0.5, 0.9, 13),  # drift collection
        _gaze_test(14),  # drift re-test
    ]
    source = _ScriptedSource(script)
    config = BenchmarkConfig(
        30.0,
        20.0,
        640,
        480,
        "camera",
        "office",
        "none",
        fixation_seconds=2.0,
        yaw_seconds=2.0,
        drift_seconds=2.0,
        max_fps=30.0,
    )
    ticks = itertools.count()

    report = run_live_benchmark(
        source,
        config,
        prompt=lambda _message: None,
        clock=lambda: float(next(ticks)),
    )

    environment = report["environment"]
    assert isinstance(environment, dict)
    assert environment["processing_fps_cap"] == config.max_fps
    assert report["end_to_end_latency_ms"] == distribution([10.0, 20.0, 30.0])
    assert report["latency_ms"] == distribution([5.0, 5.0, 5.0])
    assert source.calibration_starts == 1
    assert source.test_starts == 4
