"""Webcam-backed EventSource that emits Fovea gaze events."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from fovea.events import Blink, Eye, FoveaEvent, GazePoint, TrackingState, TrackingStatus
from fovea.webcam.camera import Webcam
from fovea.webcam.engine import GazeEngine, GazeSettings
from fovea.webcam.landmarks import FaceLandmarkEstimator, resolve_model_path


def _tracking_status(label: str) -> TrackingStatus:
    if label == "GOOD":
        return TrackingStatus.ACTIVE
    if label in {"FAIR", "POOR"}:
        return TrackingStatus.UNCERTAIN
    return TrackingStatus.LOST


@dataclass
class WebcamEventSource:
    """Read webcam frames and yield typed Fovea events."""

    settings: GazeSettings
    project_root: Path
    device_index: int = 0
    width: int = 640
    height: int = 480
    mirror: bool = True
    model_path: str | Path | None = None
    max_frames: int | None = None
    force_calibrate: bool = False

    def events(self) -> Iterator[FoveaEvent]:
        camera = Webcam(self.device_index, self.width, self.height, self.mirror)
        engine = GazeEngine(self.settings, self.project_root)
        estimator: FaceLandmarkEstimator | None = None
        frames = 0
        t0 = time.perf_counter()
        frame_count = 0
        fps = 0.0
        last_time = time.perf_counter()
        last_blink = False

        try:
            camera.connect()
            estimator = FaceLandmarkEstimator(model_path=resolve_model_path(self.model_path))
            if self.force_calibrate or engine.model is None:
                engine.start_calibration()

            while self.max_frames is None or frames < self.max_frames:
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
                    yield TrackingState(
                        status=TrackingStatus.LOST,
                        confidence=0.0,
                        timestamp_ns=timestamp_ns,
                    )
                    frames += 1
                    continue

                observation = estimator.process(frame)
                h, w = frame.shape[:2]
                landmarks = None if observation is None else observation.landmarks
                blendshapes = None if observation is None else observation.blendshapes
                output = engine.process(
                    landmarks, float(w), float(h), dt, fps, blendshapes=blendshapes
                )

                status = _tracking_status(output.tracking)
                yield TrackingState(
                    status=status,
                    confidence=output.confidence,
                    timestamp_ns=timestamp_ns,
                )

                if output.features is not None and output.features.blink and not last_blink:
                    yield Blink(
                        eye=Eye.BOTH,
                        duration_ms=0.0,
                        confidence=output.confidence,
                        timestamp_ns=timestamp_ns,
                    )
                last_blink = bool(output.features and output.features.blink)

                if output.valid and output.screen is not None:
                    yield GazePoint(
                        x=output.screen.x,
                        y=output.screen.y,
                        confidence=output.confidence,
                        timestamp_ns=timestamp_ns,
                    )

                frames += 1
        finally:
            if estimator is not None:
                estimator.close()
            camera.disconnect()

    def close(self) -> None:
        """Compatibility hook for callers that manage lifecycle explicitly."""
