"""Webcam-backed EventSource that emits Fovea gaze events."""

from __future__ import annotations

import threading
import time
import warnings
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from fovea.events import CalibrationWarning, FoveaEvent, GazePoint
from fovea.webcam.backend import create_landmark_backend
from fovea.webcam.calibration import (
    CalibrationIdentity,
    CalibrationTarget,
    validate_calibration_targets,
)
from fovea.webcam.calibration_view import CalibrationDisplay
from fovea.webcam.camera import Webcam
from fovea.webcam.capture import CaptureClosed, CaptureSession
from fovea.webcam.engine import GazeEngine, GazeSettings
from fovea.webcam.frame_processor import GazeFrameProcessor
from fovea.webcam.landmarks import FaceLandmarkEstimator, resolve_model_path
from fovea.webcam.targeting import TargetRect, validate_targets

_DIAGNOSTICS_INTERVAL_SECONDS = 0.5
# Number of most recent emitted gaze points summarized by the Diagnostics percentiles.
_LATENCY_WINDOW = 90


def _diagnostics_due(last_emitted: float | None, now: float) -> bool:
    return last_emitted is None or now - last_emitted >= _DIAGNOSTICS_INTERVAL_SECONDS


def _attach_latency(
    events: Sequence[FoveaEvent],
    latency_ms: float,
) -> tuple[FoveaEvent, ...]:
    """Attach one capture-to-ready measurement to this frame's gaze point."""
    return tuple(
        replace(event, latency_ms=latency_ms) if isinstance(event, GazePoint) else event
        for event in events
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
    max_fps: float | None = None
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
    _session: CaptureSession | None = field(default=None, init=False, repr=False)
    _close_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _pending_events: deque[FoveaEvent] = field(default_factory=deque, init=False, repr=False)
    _calibration_targets: tuple[CalibrationTarget, ...] | None = field(
        default=None, init=False, repr=False
    )
    _test_targets: tuple[CalibrationTarget, ...] | None = field(
        default=None, init=False, repr=False
    )
    _targets: tuple[TargetRect, ...] = field(default=(), init=False, repr=False)
    _processor: GazeFrameProcessor | None = field(default=None, init=False, repr=False)

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
        # One processor per session: its detectors are the session boundary.
        processor = GazeFrameProcessor(engine, self.settings, targets=self._targets)
        self._processor = processor
        frames = 0
        t0 = time.perf_counter()
        frame_count = 0
        fps = 0.0
        last_time = time.perf_counter()
        last_diagnostics_at: float | None = None
        latency_window: deque[float] = deque(maxlen=_LATENCY_WINDOW)

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
            # Capture starts last so neither model load nor display setup counts as dropped frames.
            session = CaptureSession(camera, max_fps=self.max_fps)
            self._session = session
            session.start()

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

                try:
                    captured = session.next_frame()
                except CaptureClosed:
                    break
                timestamp_ns = time.time_ns() if captured is None else captured.timestamp_ns
                diagnostics_due = self.diagnostics and _diagnostics_due(last_diagnostics_at, now)
                if captured is None:
                    events = processor.events_for_lost_frame(
                        fps,
                        timestamp_ns,
                        diagnostics_due=diagnostics_due,
                        latency_window=latency_window,
                        dropped_frames=session.dropped_frames,
                    )
                else:
                    frame = captured.pixels
                    observation = estimator.process(frame)
                    h, w = frame.shape[:2]
                    landmarks = None if observation is None else observation.landmarks
                    blendshapes = None if observation is None else observation.blendshapes
                    events = processor.events_for_frame(
                        landmarks,
                        float(w),
                        float(h),
                        dt,
                        fps,
                        timestamp_ns,
                        blendshapes,
                        diagnostics_due=diagnostics_due,
                        latency_window=latency_window,
                        dropped_frames=session.dropped_frames,
                    )
                    self._sync_display(engine)
                    if any(isinstance(event, GazePoint) for event in events):
                        latency_ms = (time.monotonic_ns() - captured.captured_ns) / 1e6
                        events = _attach_latency(events, latency_ms)
                        latency_window.append(latency_ms)
                if diagnostics_due:
                    last_diagnostics_at = now
                yield from events
                frames += 1
        finally:
            self.close()

    def _sync_display(self, engine: GazeEngine) -> None:
        wizard = engine.wizard
        if wizard is not None and not wizard.done:
            if self._display is not None:
                self._display.show(wizard, engine.targets)
        elif self._display is not None:
            display = self._display
            self._display = None
            display.close()

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
        processor = self._processor
        if processor is None:
            return
        self._pending_events.extend(processor.replace_targets(self._targets, time.time_ns()))

    def _show_wizard(self) -> None:
        if not self.show_calibration or self._engine is None or self._engine.wizard is None:
            return
        if self._display is None:
            self._display = CalibrationDisplay(self.display_width, self.display_height)
        self._display.show(self._engine.wizard, self._engine.targets)

    def close(self) -> None:
        """Stop capture and landmark inference and release the webcam.

        Safe to call multiple times and from any thread. The capture thread is
        joined before the camera is released, so ``release()`` never runs while
        a ``read()`` is in flight. This is the OpenCV analogue of stopping
        MediaStream tracks: ``VideoCapture.release()`` ends the camera session.
        """
        with self._close_lock:
            self._closed = True
            session = self._session
            if session is not None:
                session.stop()
            self._session = None
            self._engine = None
            self._processor = None
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
