"""Synthetic 478-point faces with exact pose, gaze, and blink controls."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from fovea.util import clamp01
from fovea.webcam.features import (
    _FACE_3D,
    CHIN,
    LEFT_EYE_OUTER_FOR_POSE,
    LEFT_INNER,
    LEFT_IRIS_RING,
    LEFT_LOWER,
    LEFT_UPPER,
    MOUTH_LEFT,
    MOUTH_RIGHT,
    NOSE,
    RIGHT_EYE_OUTER_FOR_POSE,
    RIGHT_INNER,
    RIGHT_IRIS_RING,
    RIGHT_LOWER,
    RIGHT_UPPER,
)

_POSE_IDS = (
    NOSE,
    CHIN,
    LEFT_EYE_OUTER_FOR_POSE,
    RIGHT_EYE_OUTER_FOR_POSE,
    MOUTH_LEFT,
    MOUTH_RIGHT,
)


@dataclass(frozen=True, slots=True)
class SyntheticLandmark:
    x: float
    y: float
    z: float = 0.0


def _rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    pitch_r, yaw_r, roll_r = np.radians((-pitch, -yaw, roll))
    cx, sx = np.cos(pitch_r), np.sin(pitch_r)
    cy, sy = np.cos(yaw_r), np.sin(yaw_r)
    cz, sz = np.cos(roll_r), np.sin(roll_r)
    rotate_x = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=np.float64)
    rotate_y = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=np.float64)
    rotate_z = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=np.float64)
    return rotate_z @ rotate_y @ rotate_x


def synthetic_landmarks(
    *,
    pitch: float = 0.0,
    yaw: float = 0.0,
    roll: float = 0.0,
    gaze_dx: float = 0.0,
    gaze_dy: float = 0.0,
    blink: bool = False,
    face_width: float = 0.44,
    image_w: int = 640,
    image_h: int = 480,
) -> list[SyntheticLandmark]:
    """Project a face and add parametric irises/eyelids without image pixels."""
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
    points = [SyntheticLandmark(0.0, 0.0) for _ in range(478)]
    for index, (x, y) in zip(_POSE_IDS, image_points.reshape(-1, 2), strict=True):
        points[index] = SyntheticLandmark(float(x / image_w), float(y / image_h))

    right_outer = points[RIGHT_EYE_OUTER_FOR_POSE]
    left_outer = points[LEFT_EYE_OUTER_FOR_POSE]
    eye_width = 0.10
    eye_height = 0.01 if blink else 0.08
    right_inner_x = right_outer.x + eye_width
    left_inner_x = left_outer.x - eye_width
    points[RIGHT_INNER] = SyntheticLandmark(right_inner_x, right_outer.y)
    points[LEFT_INNER] = SyntheticLandmark(left_inner_x, left_outer.y)

    for outer, inner_x, upper, lower, iris in (
        (right_outer, right_inner_x, RIGHT_UPPER, RIGHT_LOWER, RIGHT_IRIS_RING),
        (left_outer, left_inner_x, LEFT_UPPER, LEFT_LOWER, LEFT_IRIS_RING),
    ):
        top = outer.y - eye_height / 2
        bottom = outer.y + eye_height / 2
        center_x = min(outer.x, inner_x) + clamp01(0.5 + gaze_dx) * eye_width
        center_y = top + clamp01(0.5 + gaze_dy) * eye_height
        for index in upper:
            points[index] = SyntheticLandmark((outer.x + inner_x) / 2, top)
        for index in lower:
            points[index] = SyntheticLandmark((outer.x + inner_x) / 2, bottom)
        for index in iris:
            points[index] = SyntheticLandmark(center_x, center_y)

    center_x = (right_outer.x + left_outer.x) / 2
    points[234] = SyntheticLandmark(center_x - face_width / 2, 0.52)
    points[454] = SyntheticLandmark(center_x + face_width / 2, 0.52)
    return points


def fixture_coordinates(points: list[SyntheticLandmark]) -> list[list[float | int]]:
    """Round synthetic coordinates and compact exact zeros for checked fixtures."""

    def compact(value: float) -> float | int:
        rounded = round(value, 7)
        return 0 if rounded == 0 else rounded

    return [[compact(point.x), compact(point.y), compact(point.z)] for point in points]
