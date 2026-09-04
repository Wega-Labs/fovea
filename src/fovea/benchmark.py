"""Guided live benchmark orchestration and camera-independent report math."""

from __future__ import annotations

import math
import platform
import statistics
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from fovea.events import (
    CalibrationDone,
    Diagnostics,
    FoveaEvent,
    GazePoint,
    GazeTestDone,
    TrackingState,
    TrackingStatus,
)
from fovea.interfaces import EventSource
from fovea.util import percentile

MEDIAN_ERROR_TARGET = 0.06
MEDIAN_DEGREES_TARGET = 3.0
DISTANCES_CM = (50.0, 60.0, 75.0)

type Prompt = Callable[[str], None]
type Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    screen_width_cm: float
    screen_height_cm: float
    capture_width: int
    capture_height: int
    camera_name: str
    lighting: str
    glasses: str
    camera_index: int = 0
    fovea_version: str = "unknown"
    machine: str = platform.platform()
    fixation_seconds: float = 2.0
    yaw_seconds: float = 2.0
    drift_seconds: float = 600.0
    backend: str = "mediapipe"
    max_fps: float | None = None

    def __post_init__(self) -> None:
        if self.capture_width < 1 or self.capture_height < 1:
            raise ValueError("capture dimensions must be positive")
        if self.max_fps is not None and not (math.isfinite(self.max_fps) and self.max_fps > 0.0):
            raise ValueError("max_fps must be a finite number greater than zero")
        if self.camera_index < 0:
            raise ValueError("camera index must be non-negative")
        positive = (self.screen_width_cm, self.screen_height_cm, self.fixation_seconds)
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("screen dimensions and fixation duration must be positive")
        nonnegative = (self.yaw_seconds, self.drift_seconds)
        if not all(math.isfinite(value) and value >= 0.0 for value in nonnegative):
            raise ValueError("yaw and drift durations must be non-negative")
        metadata = (self.camera_name, self.lighting, self.glasses, self.backend)
        if not all(value.strip() for value in metadata):
            raise ValueError("camera, lighting, glasses, and backend metadata must be non-empty")


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def summarize_accuracy(
    report: GazeTestDone,
    *,
    distance_cm: float,
    screen_width_cm: float,
    screen_height_cm: float,
) -> dict[str, object]:
    geometry = (distance_cm, screen_width_cm, screen_height_cm)
    if not all(math.isfinite(value) and value > 0.0 for value in geometry):
        raise ValueError("distance and screen dimensions must be positive")
    normalized = [point.error for point in report.points]
    angular: list[float] = []
    for point in report.points:
        physical_error_cm = math.hypot(
            (point.predicted_x - point.expected_x) * screen_width_cm,
            (point.predicted_y - point.expected_y) * screen_height_cm,
        )
        angular.append(math.degrees(math.atan2(physical_error_cm, distance_cm)))
    return {
        "distance_cm": distance_cm,
        "normalized_error": distribution(normalized),
        "angular_error_degrees": distribution(angular),
        "passes_public_target": (
            report.median_error <= MEDIAN_ERROR_TARGET
            and (statistics.median(angular) if angular else math.inf) <= MEDIAN_DEGREES_TARGET
        ),
        "points": [asdict(point) for point in report.points],
    }


def summarize_jitter(points: list[GazePoint]) -> dict[str, object]:
    if not points:
        return {
            "center": None,
            "radial_error": distribution([]),
            "span_x": None,
            "span_y": None,
        }
    center_x = statistics.median(point.x for point in points)
    center_y = statistics.median(point.y for point in points)
    radial = [math.hypot(point.x - center_x, point.y - center_y) for point in points]
    return {
        "center": {"x": center_x, "y": center_y},
        "radial_error": distribution(radial),
        "span_x": max(point.x for point in points) - min(point.x for point in points),
        "span_y": max(point.y for point in points) - min(point.y for point in points),
    }


def summarize_yaw(
    diagnostics: list[Diagnostics],
    tracking: list[TrackingState],
    target_degrees: float,
) -> dict[str, object]:
    yaw_values = [event.yaw_deg for event in diagnostics]
    active = sum(event.status is TrackingStatus.ACTIVE for event in tracking)
    return {
        "target_degrees": target_degrees,
        "yaw_degrees": distribution(yaw_values),
        "active_tracking_rate": (active / len(tracking) if tracking else None),
        "reached_target": (
            any(value <= target_degrees for value in yaw_values)
            if target_degrees < 0.0
            else any(value >= target_degrees for value in yaw_values)
        ),
    }


def run_live_benchmark(
    source: EventSource,
    config: BenchmarkConfig,
    *,
    prompt: Prompt,
    clock: Clock = time.monotonic,
) -> dict[str, object]:
    """Run the guided protocol over a live event source and return a JSON-safe report."""
    events = iter(source.events())
    latency_samples: list[float] = []
    end_to_end_samples: list[float] = []

    def observe(event: FoveaEvent) -> None:
        if isinstance(event, Diagnostics):
            latency_samples.append(event.latency_ms)
        elif isinstance(event, GazePoint) and event.latency_ms is not None:
            end_to_end_samples.append(event.latency_ms)

    prompt("Sit about 60 cm from the display, then begin the guided calibration.")
    _invoke_source(source, "start_calibration")
    calibration = _until_type(events, CalibrationDone, observe)

    distance_reports: dict[str, object] = {}
    raw_distance_events: dict[float, GazeTestDone] = {}
    for distance_cm in DISTANCES_CM:
        prompt(
            f"Move to about {distance_cm:.0f} cm, keep your head neutral, and begin the gaze test."
        )
        _invoke_source(source, "start_gaze_test")
        gaze_test = _until_type(events, GazeTestDone, observe)
        raw_distance_events[distance_cm] = gaze_test
        distance_reports[f"{distance_cm:.0f}cm"] = summarize_accuracy(
            gaze_test,
            distance_cm=distance_cm,
            screen_width_cm=config.screen_width_cm,
            screen_height_cm=config.screen_height_cm,
        )

    prompt("At 60 cm, look steadily at the display center for the jitter phase.")
    jitter_events = _collect_for(events, config.fixation_seconds, observe, clock)
    jitter_points = [event for event in jitter_events if isinstance(event, GazePoint)]

    yaw_reports: dict[str, object] = {}
    for label, target_yaw in (("left", -20.0), ("right", 20.0)):
        angle = abs(target_yaw)
        prompt(f"Keep looking at the center and turn your head {label} to about {angle:.0f}°.")
        phase = _collect_for(events, config.yaw_seconds, observe, clock)
        phase_diagnostics = [event for event in phase if isinstance(event, Diagnostics)]
        phase_tracking = [event for event in phase if isinstance(event, TrackingState)]
        yaw_reports[label] = summarize_yaw(phase_diagnostics, phase_tracking, target_yaw)

    prompt(
        f"Return to a neutral 60 cm position. Drift collection runs for "
        f"{config.drift_seconds / 60.0:.1f} minutes."
    )
    _collect_for(events, config.drift_seconds, observe, clock)
    prompt("Begin the 60 cm drift re-test.")
    _invoke_source(source, "start_gaze_test")
    drift_test = _until_type(events, GazeTestDone, observe)
    drift_accuracy = summarize_accuracy(
        drift_test,
        distance_cm=60.0,
        screen_width_cm=config.screen_width_cm,
        screen_height_cm=config.screen_height_cm,
    )
    baseline = raw_distance_events[60.0]

    return {
        "schema": "fovea-benchmark-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "software": {
            "fovea": config.fovea_version,
            "python": platform.python_version(),
            "landmark_backend": config.backend,
        },
        "environment": {
            "machine": config.machine,
            "camera": config.camera_name,
            "camera_index": config.camera_index,
            "resolution": {
                "width": config.capture_width,
                "height": config.capture_height,
            },
            "lighting": config.lighting,
            "glasses": config.glasses,
            "screen_width_cm": config.screen_width_cm,
            "screen_height_cm": config.screen_height_cm,
            "processing_fps_cap": config.max_fps,
        },
        "public_targets": {
            "median_normalized_error_max": MEDIAN_ERROR_TARGET,
            "median_angular_error_degrees_max": MEDIAN_DEGREES_TARGET,
        },
        "calibration": asdict(calibration),
        "accuracy": distance_reports,
        "fixation_jitter": {
            "duration_seconds": config.fixation_seconds,
            **summarize_jitter(jitter_points),
        },
        "head_movement": yaw_reports,
        "drift": {
            "duration_seconds": config.drift_seconds,
            "retest": drift_accuracy,
            "median_normalized_error_delta": drift_test.median_error - baseline.median_error,
        },
        "latency_ms": distribution(latency_samples),
        "end_to_end_latency_ms": distribution(end_to_end_samples),
    }


def _until_type[EventT: FoveaEvent](
    events: Iterator[FoveaEvent],
    event_type: type[EventT],
    observe: Callable[[FoveaEvent], None],
) -> EventT:
    for event in events:
        observe(event)
        if isinstance(event, event_type):
            return event
    raise RuntimeError(f"event stream ended before {event_type.__name__}")


def _collect_for(
    events: Iterator[FoveaEvent],
    seconds: float,
    observe: Callable[[FoveaEvent], None],
    clock: Clock,
) -> list[FoveaEvent]:
    collected: list[FoveaEvent] = []
    deadline = clock() + seconds
    while clock() < deadline:
        try:
            event = next(events)
        except StopIteration as exc:
            raise RuntimeError("event stream ended during benchmark phase") from exc
        observe(event)
        collected.append(event)
    return collected


def _invoke_source(source: EventSource, method_name: str) -> None:
    method = getattr(source, method_name, None)
    if not callable(method):
        raise RuntimeError(f"benchmark source does not support {method_name}")
    method()
