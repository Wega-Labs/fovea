from pathlib import Path

import pytest

from fovea.webcam.landmarks import (
    FaceLandmarkEstimator,
    MediaPipeUnavailableError,
    mediapipe_available,
)

pytestmark = pytest.mark.skipif(
    not mediapipe_available(),
    reason="MediaPipe not installed",
)


def test_missing_model_raises_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "nope.task"
    with pytest.raises(MediaPipeUnavailableError, match="model not found"):
        FaceLandmarkEstimator(model_path=missing)
