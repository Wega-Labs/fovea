"""Eye-relative gaze features from MediaPipe Face Mesh landmarks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fovea.util import clamp01

RIGHT_OUTER, RIGHT_INNER = 33, 133
RIGHT_UPPER = (159, 158, 157, 173, 160)
RIGHT_LOWER = (145, 144, 163, 153)
RIGHT_IRIS_RING = (468, 469, 470, 471, 472)

LEFT_OUTER, LEFT_INNER = 263, 362
LEFT_UPPER = (386, 385, 384, 398, 387)
LEFT_LOWER = (374, 373, 380, 382)
LEFT_IRIS_RING = (473, 474, 475, 476, 477)

NOSE, CHIN = 1, 152
LEFT_EYE_OUTER_FOR_POSE = 263
RIGHT_EYE_OUTER_FOR_POSE = 33
MOUTH_LEFT, MOUTH_RIGHT = 291, 61

_FACE_3D = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, -330.0, -65.0],
        [225.0, 170.0, -135.0],
        [-225.0, 170.0, -135.0],
        [150.0, -150.0, -125.0],
        [-150.0, -150.0, -125.0],
    ],
    dtype=np.float64,
)

BLEND_LOOK_IN_LEFT = "eyelookinleft"
BLEND_LOOK_OUT_LEFT = "eyelookoutleft"
BLEND_LOOK_IN_RIGHT = "eyelookinright"
BLEND_LOOK_OUT_RIGHT = "eyelookoutright"
BLEND_LOOK_UP_LEFT = "eyelookupleft"
BLEND_LOOK_UP_RIGHT = "eyelookupright"
BLEND_LOOK_DOWN_LEFT = "eyelookdownleft"
BLEND_LOOK_DOWN_RIGHT = "eyelookdownright"


@dataclass(frozen=True)
class EyeBox:
    left: float
    top: float
    right: float
    bottom: float
    iris_x: float
    iris_y: float
    nx: float
    ny: float
    ear: float
    valid: bool


@dataclass(frozen=True)
class GazeFeatures:
    left: EyeBox
    right: EyeBox
    iris_nx: float
    iris_ny: float
    blend_x: float
    blend_y: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    face_width: float
    blink: bool
    both_eyes: bool
    tracking: str
    message: str

    def vector(self) -> NDArray[np.float64]:
        return np.array(
            [
                1.0,
                self.iris_nx - 0.5,
                self.iris_ny - 0.5,
                self.left.nx - self.right.nx,
                self.left.ny - self.right.ny,
                self.yaw_deg / 30.0,
                self.pitch_deg / 30.0,
                self.roll_deg / 30.0,
                self.blend_x,
                self.blend_y,
            ],
            dtype=np.float64,
        )


FEATURE_NAMES = (
    "bias",
    "iris_nx_c",
    "iris_ny_c",
    "delta_nx",
    "delta_ny",
    "yaw_n",
    "pitch_n",
    "roll_n",
    "blend_x",
    "blend_y",
)


def _xy(landmarks: Sequence[Any], index: int) -> tuple[float, float]:
    item = landmarks[index]
    return float(item.x), float(item.y)


def _mean_xy(landmarks: Sequence[Any], indices: tuple[int, ...]) -> tuple[float, float]:
    xs = [float(landmarks[i].x) for i in indices if i < len(landmarks)]
    ys = [float(landmarks[i].y) for i in indices if i < len(landmarks)]
    if not xs:
        return 0.5, 0.5
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _minmax(landmarks: Sequence[Any], indices: tuple[int, ...], axis: str) -> tuple[float, float]:
    vals = [float(getattr(landmarks[i], axis)) for i in indices if i < len(landmarks)]
    if not vals:
        return 0.0, 1.0
    return min(vals), max(vals)


def extract_eye(
    landmarks: Sequence[Any],
    outer: int,
    inner: int,
    upper: tuple[int, ...],
    lower: tuple[int, ...],
    iris_ring: tuple[int, ...],
) -> EyeBox:
    needed = (outer, inner, iris_ring[0])
    if any(i >= len(landmarks) for i in needed):
        return EyeBox(0, 0, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.0, False)

    ox, _oy = _xy(landmarks, outer)
    ix, _iy = _xy(landmarks, inner)
    left = min(ox, ix)
    right = max(ox, ix)
    top, _ = _minmax(landmarks, upper, "y")
    _, bottom = _minmax(landmarks, lower, "y")
    iris_x, iris_y = _mean_xy(landmarks, iris_ring)

    width = right - left
    height = bottom - top
    if width < 0.008 or height < 0.003:
        return EyeBox(left, top, right, bottom, iris_x, iris_y, 0.5, 0.5, 0.0, False)

    nx = clamp01((iris_x - left) / width)
    ny = clamp01((iris_y - top) / height)
    ear = height / width
    return EyeBox(left, top, right, bottom, iris_x, iris_y, nx, ny, ear, True)


def estimate_head_pose(
    landmarks: Sequence[Any],
    image_w: float,
    image_h: float,
) -> tuple[float, float, float]:
    ids = (NOSE, CHIN, LEFT_EYE_OUTER_FOR_POSE, RIGHT_EYE_OUTER_FOR_POSE, MOUTH_LEFT, MOUTH_RIGHT)
    if any(i >= len(landmarks) for i in ids) or image_w < 2 or image_h < 2:
        return 0.0, 0.0, 0.0
    try:
        import cv2
    except Exception:
        return 0.0, 0.0, 0.0

    image_points = np.array(
        [[_xy(landmarks, i)[0] * image_w, _xy(landmarks, i)[1] * image_h] for i in ids],
        dtype=np.float64,
    )
    camera = np.array(
        [[image_w, 0.0, image_w / 2.0], [0.0, image_w, image_h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, _tvec = cv2.solvePnP(
        _FACE_3D, image_points, camera, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0
    rotation, _ = cv2.Rodrigues(rvec)
    sy = float(np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2))
    if sy < 1e-6:
        pitch = float(np.degrees(np.arctan2(-rotation[1, 2], rotation[1, 1])))
        yaw = 0.0
        roll = 0.0
    else:
        pitch = float(np.degrees(np.arctan2(-rotation[2, 1], rotation[2, 2])))
        yaw = float(np.degrees(np.arctan2(rotation[2, 0], sy)))
        roll = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    return yaw, pitch, roll


def face_width(landmarks: Sequence[Any]) -> float:
    if len(landmarks) <= 454:
        xs = [float(lm.x) for lm in landmarks]
        return max(xs) - min(xs) if xs else 0.0
    return abs(float(landmarks[454].x) - float(landmarks[234].x))


def _blend_score(blendshapes: Mapping[str, float], *names: str) -> float:
    lowered = {key.lower(): float(value) for key, value in blendshapes.items()}
    values = [lowered[name] for name in names if name in lowered]
    if not values:
        return 0.0
    return sum(values) / len(values)


def blendshape_offset(blendshapes: Mapping[str, float] | None) -> tuple[float, float]:
    if not blendshapes:
        return 0.0, 0.0
    look_left = _blend_score(blendshapes, BLEND_LOOK_OUT_LEFT, BLEND_LOOK_IN_RIGHT)
    look_right = _blend_score(blendshapes, BLEND_LOOK_OUT_RIGHT, BLEND_LOOK_IN_LEFT)
    look_up = _blend_score(blendshapes, BLEND_LOOK_UP_LEFT, BLEND_LOOK_UP_RIGHT)
    look_down = _blend_score(blendshapes, BLEND_LOOK_DOWN_LEFT, BLEND_LOOK_DOWN_RIGHT)
    return look_right - look_left, look_down - look_up


def extract_features(
    landmarks: Sequence[Any],
    image_w: float,
    image_h: float,
    blink_ear: float,
    min_face_width: float,
    max_yaw_deg: float,
    blendshapes: Mapping[str, float] | None = None,
) -> GazeFeatures:
    left = extract_eye(landmarks, LEFT_OUTER, LEFT_INNER, LEFT_UPPER, LEFT_LOWER, LEFT_IRIS_RING)
    right = extract_eye(
        landmarks, RIGHT_OUTER, RIGHT_INNER, RIGHT_UPPER, RIGHT_LOWER, RIGHT_IRIS_RING
    )
    yaw, pitch, roll = estimate_head_pose(landmarks, image_w, image_h)
    width = face_width(landmarks)
    blend_x, blend_y = blendshape_offset(blendshapes)
    both = left.valid and right.valid
    if both:
        iris_nx = 0.5 * (left.nx + right.nx)
        iris_ny = 0.5 * (left.ny + right.ny)
        ear = 0.5 * (left.ear + right.ear)
    elif left.valid:
        iris_nx, iris_ny, ear = left.nx, left.ny, left.ear
    elif right.valid:
        iris_nx, iris_ny, ear = right.nx, right.ny, right.ear
    else:
        return GazeFeatures(
            left,
            right,
            0.5,
            0.5,
            blend_x,
            blend_y,
            yaw,
            pitch,
            roll,
            width,
            True,
            False,
            "LOST",
            "Eyes not detected",
        )

    blink = ear < blink_ear
    if width < min_face_width:
        tracking, message = "POOR", "Face too far from camera"
    elif abs(yaw) > max_yaw_deg:
        tracking, message = "POOR", "Please face the camera"
    elif not both:
        tracking, message = "FAIR", "One eye missing"
    elif blink:
        tracking, message = "FAIR", "Blink / eyes closed"
    else:
        tracking, message = "GOOD", ""

    return GazeFeatures(
        left=left,
        right=right,
        iris_nx=iris_nx,
        iris_ny=iris_ny,
        blend_x=blend_x,
        blend_y=blend_y,
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        face_width=width,
        blink=blink,
        both_eyes=both,
        tracking=tracking,
        message=message,
    )
