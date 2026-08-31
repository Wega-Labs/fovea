"""Compatibility session around the pluggable landmark backend boundary."""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from fovea.webcam.backend import (
    LandmarkBackend,
    LandmarkObservation,
    create_landmark_backend,
)
from fovea.webcam.backend import (
    MediaPipeUnavailableError as MediaPipeUnavailableError,
)
from fovea.webcam.backend import (
    mediapipe_available as mediapipe_available,
)
from fovea.webcam.model import DEFAULT_MODEL_PATH, FACE_LANDMARKER_URL

MODEL_URL = FACE_LANDMARKER_URL


FaceObservation = LandmarkObservation


def resolve_model_path(configured: str | Path | None) -> Path:
    if not configured:
        return DEFAULT_MODEL_PATH
    path = Path(configured)
    return path


class FaceLandmarkEstimator:
    """BGR camera-frame session retained for source compatibility."""

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        backend: LandmarkBackend | None = None,
    ) -> None:
        path = model_path or DEFAULT_MODEL_PATH
        self._backend: LandmarkBackend = backend or create_landmark_backend("mediapipe")
        self._backend.open(path)
        self._timestamp_ms = -1

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def close(self) -> None:
        self._backend.close()

    def process(self, frame_bgr: NDArray[np.uint8]) -> FaceObservation | None:
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        now_ms = time.monotonic_ns() // 1_000_000
        self._timestamp_ms = max(self._timestamp_ms + 1, now_ms)
        return self._backend.process(cast(NDArray[np.uint8], rgb), self._timestamp_ms)
