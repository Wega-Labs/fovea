from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fovea.webcam.backend import LandmarkBackend, LandmarkObservation
from fovea.webcam.landmarks import FaceLandmarkEstimator


class FakeBackend:
    name = "fixture"

    def __init__(self, observation: LandmarkObservation) -> None:
        self.observation = observation
        self.model: Path | None = None
        self.frames: list[NDArray[np.uint8]] = []
        self.timestamps: list[int] = []
        self.closed = False

    def open(self, model: Path) -> None:
        self.model = model

    def process(
        self,
        frame_rgb: NDArray[np.uint8],
        timestamp_ms: int,
    ) -> LandmarkObservation | None:
        self.frames.append(frame_rgb.copy())
        self.timestamps.append(timestamp_ms)
        return self.observation

    def close(self) -> None:
        self.closed = True


def test_estimator_adapts_camera_frames_to_landmark_backend(tmp_path: Path) -> None:
    observation = LandmarkObservation([object()], {"eyeblinkleft": 0.4})
    backend = FakeBackend(observation)
    model = tmp_path / "fixture.task"

    assert isinstance(backend, LandmarkBackend)
    estimator = FaceLandmarkEstimator(model_path=model, backend=backend)
    frame_bgr = np.array([[[1, 2, 3]]], dtype=np.uint8)

    assert estimator.backend_name == "fixture"
    assert estimator.process(frame_bgr) is observation
    assert estimator.process(frame_bgr) is observation
    assert backend.model == model
    assert backend.frames[0].tolist() == [[[3, 2, 1]]]
    assert backend.timestamps[1] > backend.timestamps[0]

    estimator.close()
    estimator.close()
    assert backend.closed
