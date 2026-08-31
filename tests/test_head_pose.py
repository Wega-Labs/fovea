"""Projection-based tests for the public head-pose sign convention."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from fovea.webcam.calibration import uncalibrated_map
from fovea.webcam.features import (
    _FACE_3D,
    CHIN,
    LEFT_EYE_OUTER_FOR_POSE,
    MOUTH_LEFT,
    MOUTH_RIGHT,
    NOSE,
    RIGHT_EYE_OUTER_FOR_POSE,
    EyeBox,
    GazeFeatures,
    estimate_head_pose,
)

_POSE_IDS = (
    NOSE,
    CHIN,
    LEFT_EYE_OUTER_FOR_POSE,
    RIGHT_EYE_OUTER_FOR_POSE,
    MOUTH_LEFT,
    MOUTH_RIGHT,
)


def _rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Build a rotation using Fovea's visual, y-down sign convention."""
    pitch_r, yaw_r, roll_r = np.radians((-pitch, -yaw, roll))
    cx, sx = np.cos(pitch_r), np.sin(pitch_r)
    cy, sy = np.cos(yaw_r), np.sin(yaw_r)
    cz, sz = np.cos(roll_r), np.sin(roll_r)
    rotate_x = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=np.float64)
    rotate_y = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=np.float64)
    rotate_z = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=np.float64)
    return rotate_z @ rotate_y @ rotate_x


def _project_face(
    pitch: float,
    yaw: float,
    roll: float,
    image_w: int = 640,
    image_h: int = 480,
) -> list[SimpleNamespace]:
    camera = np.array(
        (
            (image_w, 0.0, image_w / 2),
            (0.0, image_w, image_h / 2),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    rvec, _ = cv2.Rodrigues(_rotation_matrix(pitch, yaw, roll))
    image_points, _ = cv2.projectPoints(
        _FACE_3D,
        rvec,
        np.array(((0.0,), (0.0,), (1000.0,)), dtype=np.float64),
        camera,
        np.zeros((4, 1), dtype=np.float64),
    )
    landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(292)]
    for index, (x, y) in zip(_POSE_IDS, image_points.reshape(-1, 2), strict=True):
        landmarks[index] = SimpleNamespace(x=float(x / image_w), y=float(y / image_h), z=0.0)
    return landmarks


@pytest.mark.parametrize(
    ("pitch", "yaw", "roll"),
    [
        (0.0, 0.0, 0.0),
        (15.0, 0.0, 0.0),
        (-15.0, 0.0, 0.0),
        (0.0, 20.0, 0.0),
        (0.0, -20.0, 0.0),
        (0.0, 0.0, 10.0),
        (0.0, 0.0, -10.0),
    ],
)
def test_projected_pose_recovers_documented_signs(
    pitch: float,
    yaw: float,
    roll: float,
) -> None:
    recovered_yaw, recovered_pitch, recovered_roll = estimate_head_pose(
        _project_face(pitch, yaw, roll), 640, 480
    )
    assert recovered_pitch == pytest.approx(pitch, abs=2.0)
    assert recovered_yaw == pytest.approx(yaw, abs=2.0)
    assert recovered_roll == pytest.approx(roll, abs=2.0)


def test_uncalibrated_frontal_face_maps_to_center() -> None:
    yaw, pitch, roll = estimate_head_pose(_project_face(0.0, 0.0, 0.0), 640, 480)
    eye = EyeBox(0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.35, True)
    features = GazeFeatures(
        left=eye,
        right=eye,
        iris_nx=0.5,
        iris_ny=0.5,
        blend_x=0.0,
        blend_y=0.0,
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        face_width=0.3,
        blink=False,
        both_eyes=True,
        tracking="GOOD",
        message="",
    )
    assert uncalibrated_map(features) == pytest.approx((0.5, 0.5), abs=0.05)
