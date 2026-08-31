"""Landmark inference backend contract and the MediaPipe implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class LandmarkObservation:
    """Backend-neutral landmarks, blendshapes, and optional transform."""

    landmarks: Sequence[Any]
    blendshapes: dict[str, float]
    transform: object | None = None


@runtime_checkable
class LandmarkBackend(Protocol):
    """Inference boundary shared by live and future landmark implementations."""

    @property
    def name(self) -> str:
        """Stable backend name used by the CLI and handshake."""
        ...

    def open(self, model: Path) -> None:
        """Load a verified model asset and allocate inference state."""
        ...

    def process(
        self,
        frame_rgb: NDArray[np.uint8],
        timestamp_ms: int,
    ) -> LandmarkObservation | None:
        """Infer one RGB frame at a strictly increasing millisecond timestamp."""
        ...

    def close(self) -> None:
        """Release all inference resources; repeated calls are safe."""
        ...


class MediaPipeUnavailableError(RuntimeError):
    """Raised when the selected MediaPipe backend cannot be initialized."""


def mediapipe_available() -> bool:
    try:
        import mediapipe  # noqa: F401
    except Exception:
        return False
    return True


class MediaPipeBackend:
    """MediaPipe Tasks implementation of :class:`LandmarkBackend`."""

    name = "mediapipe"

    def __init__(self) -> None:
        self._landmarker: Any | None = None

    def open(self, model: Path) -> None:
        if self._landmarker is not None:
            raise RuntimeError("MediaPipe backend is already open")
        if not mediapipe_available():
            raise MediaPipeUnavailableError(
                "MediaPipe is not installed or cannot be imported. "
                "Install fovea-input with OpenCV and MediaPipe dependencies."
            )
        if not model.is_file():
            raise MediaPipeUnavailableError(
                f"Face landmarker model not found: {model}. "
                "Run: python scripts/download_mediapipe_model.py"
            )

        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode,
        )
        from mediapipe.tasks.python.vision.face_landmarker import (
            FaceLandmarker,
            FaceLandmarkerOptions,
        )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def process(
        self,
        frame_rgb: NDArray[np.uint8],
        timestamp_ms: int,
    ) -> LandmarkObservation | None:
        landmarker = self._landmarker
        if landmarker is None:
            raise RuntimeError("MediaPipe backend is not open")
        import mediapipe as mp

        rgb = np.ascontiguousarray(frame_rgb)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return None
        blendshapes: dict[str, float] = {}
        if result.face_blendshapes:
            for category in result.face_blendshapes[0]:
                name = category.category_name or category.display_name
                if name and category.score is not None:
                    blendshapes[name.lower()] = float(category.score)
        return LandmarkObservation(
            landmarks=result.face_landmarks[0],
            blendshapes=blendshapes,
            transform=None,
        )

    def close(self) -> None:
        landmarker = self._landmarker
        self._landmarker = None
        if landmarker is None:
            return
        closer = getattr(landmarker, "close", None)
        if callable(closer):
            closer()


BACKEND_NAMES = ("mediapipe",)


def create_landmark_backend(name: str) -> LandmarkBackend:
    if name == "mediapipe":
        return MediaPipeBackend()
    raise ValueError(f"unknown landmark backend: {name}")


def backend_available(name: str) -> bool:
    if name == "mediapipe":
        return mediapipe_available()
    return False
