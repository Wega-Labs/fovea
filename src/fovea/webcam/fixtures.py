"""Privacy-safe landmark recording and deterministic replay."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fovea.events import FoveaEvent
from fovea.webcam.camera import Webcam
from fovea.webcam.engine import GazeEngine, GazeSettings
from fovea.webcam.frame_processor import GazeFrameProcessor
from fovea.webcam.landmarks import FaceLandmarkEstimator, resolve_model_path


@dataclass(frozen=True, slots=True)
class RecordedLandmark:
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class LandmarkFrame:
    ts_ns: int
    w: int
    h: int
    landmarks: tuple[RecordedLandmark, ...]
    blendshapes: dict[str, float]
    transform: object | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "ts_ns": self.ts_ns,
            "w": self.w,
            "h": self.h,
            "landmarks": [[point.x, point.y, point.z] for point in self.landmarks],
            "blendshapes": self.blendshapes,
            "transform": self.transform,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LandmarkFrame:
        rows = data.get("landmarks")
        if not isinstance(rows, list):
            raise ValueError("fixture landmarks must be a list")
        landmarks: list[RecordedLandmark] = []
        for row in rows:
            if not isinstance(row, list | tuple) or len(row) != 3:
                raise ValueError("each fixture landmark must contain x, y, and z")
            landmarks.append(RecordedLandmark(float(row[0]), float(row[1]), float(row[2])))
        blendshapes_raw = data.get("blendshapes", {})
        if not isinstance(blendshapes_raw, dict):
            raise ValueError("fixture blendshapes must be an object")
        width = int(data["w"])
        height = int(data["h"])
        if width < 1 or height < 1:
            raise ValueError("fixture width and height must be positive")
        return cls(
            ts_ns=int(data["ts_ns"]),
            w=width,
            h=height,
            landmarks=tuple(landmarks),
            blendshapes={str(key): float(value) for key, value in blendshapes_raw.items()},
            transform=data.get("transform"),
        )


def frame_to_json(frame: LandmarkFrame) -> str:
    return json.dumps(frame.to_dict(), ensure_ascii=False, separators=(",", ":"))


def read_landmark_frames(path: Path) -> Iterator[LandmarkFrame]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError("fixture row must be an object")
                yield LandmarkFrame.from_dict(data)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid landmark fixture {path}:{line_number}: {exc}") from exc


def record_landmarks(
    path: Path,
    *,
    seconds: float,
    device_index: int = 0,
    width: int = 640,
    height: int = 480,
    mirror: bool = True,
    model_path: str | Path | None = None,
    max_frames: int | None = None,
) -> int:
    """Record landmarks and blendshapes only; camera pixels are never written."""
    camera = Webcam(device_index, width, height, mirror)
    estimator: FaceLandmarkEstimator | None = None
    written = 0
    deadline = time.monotonic() + seconds
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        camera.connect()
        estimator = FaceLandmarkEstimator(model_path=resolve_model_path(model_path))
        with path.open("w", encoding="utf-8") as handle:
            while time.monotonic() < deadline and (max_frames is None or written < max_frames):
                pixels = camera.read()
                if pixels is None:
                    continue
                frame_height, frame_width = pixels.shape[:2]
                observation = estimator.process(pixels)
                points: tuple[RecordedLandmark, ...] = ()
                blendshapes: dict[str, float] = {}
                if observation is not None:
                    points = tuple(
                        RecordedLandmark(
                            float(point.x),
                            float(point.y),
                            float(getattr(point, "z", 0.0)),
                        )
                        for point in observation.landmarks
                    )
                    blendshapes = observation.blendshapes
                frame = LandmarkFrame(
                    ts_ns=time.time_ns(),
                    w=int(frame_width),
                    h=int(frame_height),
                    landmarks=points,
                    blendshapes=blendshapes,
                )
                handle.write(frame_to_json(frame) + "\n")
                handle.flush()
                written += 1
    finally:
        if estimator is not None:
            estimator.close()
        camera.disconnect()
    return written


@dataclass
class ReplayEventSource:
    """Replay landmark JSONL through the same frame processor as the webcam."""

    path: Path
    settings: GazeSettings
    project_root: Path
    max_frames: int | None = None
    force_calibrate: bool = False
    force_test: bool = False
    _closed: bool = field(default=True, init=False, repr=False)
    _engine: GazeEngine | None = field(default=None, init=False, repr=False)

    def events(self) -> Iterator[FoveaEvent]:
        self.close()
        self._closed = False
        engine = GazeEngine(self.settings, self.project_root)
        self._engine = engine
        processor = GazeFrameProcessor(engine)
        if self.force_calibrate:
            engine.start_calibration()
        elif self.force_test:
            engine.start_gaze_test()
        previous_ts: int | None = None
        frames = 0
        try:
            for frame in read_landmark_frames(self.path):
                if self._closed or (self.max_frames is not None and frames >= self.max_frames):
                    break
                if previous_ts is None:
                    dt = 1.0 / 30.0
                else:
                    dt = max(0.001, (frame.ts_ns - previous_ts) / 1_000_000_000.0)
                previous_ts = frame.ts_ns
                fps = 1.0 / dt
                landmarks = frame.landmarks or None
                yield from processor.events_for_frame(
                    landmarks,
                    float(frame.w),
                    float(frame.h),
                    dt,
                    fps,
                    frame.ts_ns,
                    frame.blendshapes,
                )
                frames += 1
        finally:
            self.close()

    def start_calibration(self) -> None:
        if self._engine is not None:
            self._engine.start_calibration()

    def start_gaze_test(self) -> None:
        if self._engine is not None:
            self._engine.start_gaze_test()

    def observe(
        self,
        x: float,
        y: float,
        weight: float = 1.0,
        timestamp_ns: int | None = None,
    ) -> None:
        if self._engine is not None:
            self._engine.observe(x, y, weight, timestamp_ns)

    def close(self) -> None:
        self._closed = True
        self._engine = None
