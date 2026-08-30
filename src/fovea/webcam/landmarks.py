"""MediaPipe face landmarks for gaze estimation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)


@dataclass(frozen=True)
class FaceObservation:
    landmarks: Sequence[Any]
    blendshapes: dict[str, float]


class MediaPipeUnavailableError(RuntimeError):
    pass


def mediapipe_available() -> bool:
    try:
        import mediapipe  # noqa: F401
    except Exception:
        return False
    return True


def resolve_model_path(configured: str | Path | None) -> Path:
    if not configured:
        return DEFAULT_MODEL_PATH
    path = Path(configured)
    return path


class FaceLandmarkEstimator:
    """Adapter around MediaPipe Tasks FaceLandmarker."""

    def __init__(self, model_path: Path | None = None) -> None:
        if not mediapipe_available():
            msg = (
                "MediaPipe is not installed or cannot be imported. "
                "Install fovea-input with OpenCV and MediaPipe dependencies."
            )
            raise MediaPipeUnavailableError(msg)
        path = model_path or DEFAULT_MODEL_PATH
        if not path.is_file():
            msg = (
                f"Face landmarker model not found: {path}. "
                "Run: python scripts/download_mediapipe_model.py"
            )
            raise MediaPipeUnavailableError(msg)

        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode,
        )
        from mediapipe.tasks.python.vision.face_landmarker import (
            FaceLandmarker,
            FaceLandmarkerOptions,
        )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def close(self) -> None:
        closer = getattr(self._landmarker, "close", None)
        if closer is not None:
            closer()

    def process(self, frame_bgr: np.ndarray) -> FaceObservation | None:
        import cv2
        import mediapipe as mp

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33
        result = self._landmarker.detect_for_video(image, self._timestamp_ms)
        if not result.face_landmarks:
            return None
        blendshapes: dict[str, float] = {}
        if result.face_blendshapes:
            for category in result.face_blendshapes[0]:
                name = category.category_name or category.display_name
                if name and category.score is not None:
                    blendshapes[name.lower()] = float(category.score)
        return FaceObservation(
            landmarks=result.face_landmarks[0],
            blendshapes=blendshapes,
        )
