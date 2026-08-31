"""Projection-based tests for the public head-pose sign convention."""

import pytest

from fovea.webcam.calibration import uncalibrated_map
from fovea.webcam.features import (
    EyeBox,
    GazeFeatures,
    estimate_head_pose,
)
from tests.synth import synthetic_landmarks


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
        synthetic_landmarks(pitch=pitch, yaw=yaw, roll=roll), 640, 480
    )
    assert recovered_pitch == pytest.approx(pitch, abs=2.0)
    assert recovered_yaw == pytest.approx(yaw, abs=2.0)
    assert recovered_roll == pytest.approx(roll, abs=2.0)


def test_uncalibrated_frontal_face_maps_to_center() -> None:
    yaw, pitch, roll = estimate_head_pose(synthetic_landmarks(), 640, 480)
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
