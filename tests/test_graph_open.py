"""Open the real MediaPipe graph on a synthetic frame.

This regression test catches the macOS graph-open crash tracked in issue #2.
"""

import numpy as np
import pytest

from fovea.webcam.model import DEFAULT_MODEL_PATH

pytestmark = pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.is_file(),
    reason="run scripts/download_mediapipe_model.py",
)


def test_real_graph_opens_on_synthetic_frame() -> None:
    from fovea.webcam.landmarks import FaceLandmarkEstimator

    estimator = FaceLandmarkEstimator()
    try:
        frame = np.random.default_rng(0).integers(
            0,
            255,
            (480, 640, 3),
            dtype=np.uint8,
        )
        for _ in range(3):
            assert estimator.process(frame) is None
    finally:
        estimator.close()
