"""Webcam-backed EventSource that emits Fovea gaze events."""

from __future__ import annotations

import time
import warnings
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from fovea.events import (
    CalibrationCue,
    CalibrationDone,
    CalibrationWarning,
    Diagnostics,
    FoveaEvent,
    GazePoint,
    GazeTestDone,
    GazeTestPoint,
    TrackingState,
    TrackingStatus,
)
from fovea.webcam.backend import create_landmark_backend
from fovea.webcam.calibration import (
    CalibrationIdentity,
    CalibrationTarget,
    validate_calibration_targets,
)
from fovea.webcam.calibration_view import CalibrationDisplay
from fovea.webcam.camera import Webcam
from fovea.webcam.engine import GazeEngine, GazeOutput, GazeSettings
from fovea.webcam.landmarks import FaceLandmarkEstimator, resolve_model_path
from fovea.webcam.targeting import TargetMatch, TargetRect, TargetTracker, validate_targets
from fovea.webcam.temporal import BlinkDetector, FixationDetector


def _tracking_status(label: str) -> TrackingStatus:
    if label == "GOOD":
        return TrackingStatus.ACTIVE
    if label in {"FAIR", "POOR"}:
        return TrackingStatus.UNCERTAIN
    return TrackingStatus.LOST


_DIAGNOSTICS_INTERVAL_SECONDS = 0.5


def _diagnostics_due(last_emitted: float | None, now: float) -> bool:
    return last_emitted is None or now - last_emitted >= _DIAGNOSTICS_INTERVAL_SECONDS


def _diagnostics_event(output: GazeOutput, timestamp_ns: int) -> Diagnostics:
    features = output.features
    return Diagnostics(
        fps=output.fps,
        latency_ms=output.latency_ms,
        face_width=0.0 if features is None else features.face_width,
        yaw_deg=0.0 if features is None else features.yaw_deg,
        pitch_deg=0.0 if features is None else features.pitch_deg,
        timestamp_ns=timestamp_ns,
    )


def _gaze_test_event(report: dict[str, object], timestamp_ns: int) -> GazeTestDone | None:
    points_raw = report.get("points")
    n_points = report.get("n")
    mean_error = report.get("mean_error")
    median_error = report.get("median_error")
    max_error = report.get("max_error")
    aggregates = (n_points, mean_error, median_error, max_error)
    if any(not isinstance(value, int | float) or isinstance(value, bool) for value in aggregates):
        return None
    if not isinstance(points_raw, list):
        return None
    points: list[GazeTestPoint] = []
    for point_raw in points_raw:
        if not isinstance(point_raw, dict):
            return None
        expected = point_raw.get("expected")
        predicted = point_raw.get("predicted")
        error = point_raw.get("error")
        if not isinstance(expected, list) or len(expected) != 2:
            return None
        if not isinstance(predicted, list) or len(predicted) != 2:
            return None
        coordinates: list[float] = []
        for value in (*expected, *predicted, error):
            if not isinstance(value, int | float) or isinstance(value, bool):
                return None
            coordinates.append(float(value))
        expected_x, expected_y, predicted_x, predicted_y, point_error = coordinates
        points.append(
            GazeTestPoint(
                expected_x=expected_x,
                expected_y=expected_y,
                predicted_x=predicted_x,
                predicted_y=predicted_y,
                error=point_error,
            )
        )
    assert isinstance(n_points, int | float)
    assert isinstance(mean_error, int | float)
    assert isinstance(median_error, int | float)
    assert isinstance(max_error, int | float)
    return GazeTestDone(
        n_points=int(n_points),
        mean_error=float(mean_error),
        median_error=float(median_error),
        max_error=float(max_error),
        points=tuple(points),
        timestamp_ns=timestamp_ns,
    )


@dataclass
class WebcamEventSource:
    """Read webcam frames and yield typed Fovea events."""

    settings: GazeSettings
    project_root: Path | None = None
    device_index: int = 0
    width: int = 640
    height: int = 480
    mirror: bool = True
    model_path: str | Path | None = None
    max_frames: int | None = None
    force_calibrate: bool = False
    force_test: bool = False
    show_calibration: bool = False
    diagnostics: bool = False
    display_id: str | None = None
    display_width: int = 1280
    display_height: int = 720
    backend: str = "mediapipe"
    _camera: Webcam | None = field(default=None, init=False, repr=False)
    _estimator: FaceLandmarkEstimator | None = field(default=None, init=False, repr=False)
    _display: CalibrationDisplay | None = field(default=None, init=False, repr=False)
    _engine: GazeEngine | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=True, init=False, repr=False)
    _pending_events: deque[FoveaEvent] = field(default_factory=deque, init=False, repr=False)
    _calibration_targets: tuple[CalibrationTarget, ...] | None = field(
        default=None, init=False, repr=False
    )
    _test_targets: tuple[CalibrationTarget, ...] | None = field(
        default=None, init=False, repr=False
    )
    _targets: tuple[TargetRect, ...] = field(default=(), init=False, repr=False)
    _target_tracker: TargetTracker | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.project_root is not None:
            warnings.warn(
                "WebcamEventSource.project_root is deprecated and no longer used; "
                "configure an absolute GazeSettings.calibration_path instead",
                DeprecationWarning,
                stacklevel=2,
            )

    def events(self) -> Iterator[FoveaEvent]:
        self.close()
        self._closed = False
        self._pending_events.clear()
        camera = Webcam(self.device_index, self.width, self.height, self.mirror)
        self._camera = camera
        identity = CalibrationIdentity(
            display_id=self.display_id,
            display_width=self.display_width,
            display_height=self.display_height,
            camera_index=self.device_index,
            frame_width=self.width,
            frame_height=self.height,
        )
        engine = GazeEngine(self.settings, identity=identity)
        self._engine = engine
        frames = 0
        t0 = time.perf_counter()
        frame_count = 0
        fps = 0.0
        last_time = time.perf_counter()
        blink_detector = BlinkDetector()
        fixation_detector = FixationDetector(
            stability_ms=self.settings.stability_ms,
            radius=self.settings.hysteresis,
        )
        target_tracker = TargetTracker(
            dwell_ms=self.settings.dwell_ms,
            hysteresis=self.settings.hysteresis,
            expand=self.settings.target_expand,
            snap_radius=self.settings.snap_radius,
            targets=self._targets,
        )
        self._target_tracker = target_tracker
        last_diagnostics_at: float | None = None
        emitted_calibration_report: dict[str, object] | None = None
        emitted_test_report: dict[str, object] | None = None

        try:
            camera.connect()
            estimator = FaceLandmarkEstimator(
                model_path=resolve_model_path(self.model_path),
                backend=create_landmark_backend(self.backend),
            )
            self._estimator = estimator
            if self.force_calibrate or engine.model is None:
                self.start_calibration(self._calibration_targets)
            elif self.force_test:
                self.start_gaze_test(self._test_targets)

            while not self._closed and (self.max_frames is None or frames < self.max_frames):
                while self._pending_events:
                    yield self._pending_events.popleft()
                now = time.perf_counter()
                dt = max(1e-3, now - last_time)
                last_time = now
                frame_count += 1
                elapsed = now - t0
                if elapsed >= 0.5:
                    fps = frame_count / elapsed
                    frame_count = 0
                    t0 = now

                frame = camera.read()
                timestamp_ns = time.time_ns()
                if frame is None:
                    blink_detector.reset()
                    fixation_detector.reset()
                    target_tracker.freeze(timestamp_ns)
                    yield TrackingState(
                        status=TrackingStatus.LOST,
                        confidence=0.0,
                        timestamp_ns=timestamp_ns,
                    )
                    if self.diagnostics and _diagnostics_due(last_diagnostics_at, now):
                        yield Diagnostics(
                            fps=fps,
                            latency_ms=0.0,
                            face_width=0.0,
                            yaw_deg=0.0,
                            pitch_deg=0.0,
                            timestamp_ns=timestamp_ns,
                        )
                        last_diagnostics_at = now
                    frames += 1
                    continue

                observation = estimator.process(frame)
                h, w = frame.shape[:2]
                landmarks = None if observation is None else observation.landmarks
                blendshapes = None if observation is None else observation.blendshapes
                output = engine.process(
                    landmarks, float(w), float(h), dt, fps, blendshapes=blendshapes
                )

                wizard = engine.wizard
                if wizard is not None and not wizard.done:
                    yield CalibrationCue(
                        label=wizard.label,
                        x=wizard.sx,
                        y=wizard.sy,
                        index=wizard.index,
                        total=len(engine.targets),
                        samples=wizard.samples,
                        needed=wizard.needed,
                        instruction=wizard.instruction,
                        timestamp_ns=timestamp_ns,
                    )
                    if self._display is not None:
                        self._display.show(wizard, engine.targets)
                elif self._display is not None:
                    display = self._display
                    self._display = None
                    display.close()

                status = _tracking_status(output.tracking)
                yield TrackingState(
                    status=status,
                    confidence=output.confidence,
                    timestamp_ns=timestamp_ns,
                    detail="" if output.features is None else output.features.message,
                )

                if self.diagnostics and _diagnostics_due(last_diagnostics_at, now):
                    yield _diagnostics_event(output, timestamp_ns)
                    last_diagnostics_at = now

                calibration_report = engine.last_calibration_report
                if calibration_report and calibration_report is not emitted_calibration_report:
                    yield CalibrationDone(
                        n_points=cast(int, calibration_report["n_points"]),
                        coverage=cast(float, calibration_report["coverage"]),
                        loo_error=cast(float, calibration_report["loo_error"]),
                        timestamp_ns=timestamp_ns,
                    )
                    emitted_calibration_report = calibration_report

                test_report = engine.last_test_report
                if test_report and test_report is not emitted_test_report:
                    test_event = _gaze_test_event(test_report, timestamp_ns)
                    if test_event is not None:
                        engine.resume_after_gaze_test()
                        yield test_event
                        emitted_test_report = test_report

                if output.features is None or output.tracking == "LOST":
                    blink_detector.reset()
                    fixation_detector.reset()
                else:
                    blink_event = blink_detector.update(
                        output.features.blink,
                        output.confidence,
                        timestamp_ns,
                    )
                    if blink_event is not None:
                        yield blink_event

                target_match: TargetMatch | None = None
                target_events: tuple[FoveaEvent, ...] = ()
                if status is TrackingStatus.ACTIVE and output.valid and output.screen is not None:
                    target_update = target_tracker.update(
                        output.screen.x,
                        output.screen.y,
                        output.confidence,
                        timestamp_ns,
                    )
                    target_match = target_update.match
                    target_events = target_update.events
                else:
                    target_match = target_tracker.freeze(timestamp_ns)

                if output.valid and output.screen is not None:
                    yield GazePoint(
                        x=output.screen.x,
                        y=output.screen.y,
                        confidence=output.confidence,
                        timestamp_ns=timestamp_ns,
                        target_id=(None if target_match is None else target_match.target_id),
                        snapped_x=(None if target_match is None else target_match.snapped_x),
                        snapped_y=(None if target_match is None else target_match.snapped_y),
                    )
                    yield from target_events
                    fixation = fixation_detector.update(
                        output.screen.x,
                        output.screen.y,
                        output.confidence,
                        timestamp_ns,
                    )
                    if fixation is not None:
                        yield fixation
                else:
                    fixation_detector.reset()

                frames += 1
        finally:
            self.close()

    def start_calibration(
        self,
        targets: Sequence[CalibrationTarget] | None = None,
    ) -> None:
        """Start or restart calibration between frames."""
        self.force_calibrate = True
        self.force_test = False
        self._calibration_targets = (
            None if targets is None else validate_calibration_targets(targets)
        )
        if self._engine is None:
            return
        self._engine.start_calibration(self._calibration_targets)
        self._queue_calibration_warning()
        self._show_wizard()

    def start_gaze_test(
        self,
        targets: Sequence[CalibrationTarget] | None = None,
    ) -> None:
        """Start or restart the calibrated gaze test between frames."""
        self.force_test = True
        self.force_calibrate = False
        self._test_targets = None if targets is None else validate_calibration_targets(targets)
        if self._engine is None:
            return
        self._engine.start_gaze_test(self._test_targets)
        self._queue_calibration_warning()
        self._show_wizard()

    def _queue_calibration_warning(self) -> None:
        if self._engine is None or not self._engine.calibration_warning:
            return
        self._pending_events.append(
            CalibrationWarning(
                message=self._engine.calibration_warning,
                coverage=min(self._engine.coverage_x, self._engine.coverage_y),
                timestamp_ns=time.time_ns(),
            )
        )

    def set_targets(self, targets: Sequence[TargetRect]) -> None:
        """Replace host-registered targets between frames."""
        self._targets = validate_targets(tuple(targets))
        if self._target_tracker is None:
            return
        self._pending_events.extend(
            self._target_tracker.replace_targets(self._targets, time.time_ns())
        )

    def _show_wizard(self) -> None:
        if not self.show_calibration or self._engine is None or self._engine.wizard is None:
            return
        if self._display is None:
            self._display = CalibrationDisplay(self.display_width, self.display_height)
        self._display.show(self._engine.wizard, self._engine.targets)

    def close(self) -> None:
        """Stop landmark inference and release the webcam capture.

        Safe to call multiple times. This is the OpenCV analogue of stopping
        MediaStream tracks: ``VideoCapture.release()`` ends the camera session.
        """
        self._closed = True
        self._engine = None
        self._target_tracker = None
        display = self._display
        self._display = None
        if display is not None:
            display.close()
        estimator = self._estimator
        self._estimator = None
        if estimator is not None:
            estimator.close()
        camera = self._camera
        self._camera = None
        if camera is not None:
            camera.disconnect()
