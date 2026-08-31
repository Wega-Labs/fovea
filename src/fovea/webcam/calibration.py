"""10-point ridge calibration: eye/face features → normalized screen XY."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fovea.util import clamp01
from fovea.webcam.features import FEATURE_NAMES, GazeFeatures

CALIBRATION_VERSION = 3
MINIMUM_CALIBRATION_VERSION = 2


@dataclass(frozen=True, slots=True)
class CalibrationTarget:
    """One calibration look-target in normalized screen space (origin top-left)."""

    label: str
    x: float
    y: float

    def pixel_xy(self, width: int, height: int) -> tuple[int, int]:
        """Map this target onto a canvas of ``width`` x ``height`` pixels."""
        if width < 1 or height < 1:
            msg = "width and height must be at least 1"
            raise ValueError(msg)
        px = round(self.x * (width - 1))
        py = round(self.y * (height - 1))
        return px, py


CALIBRATION_LAYOUT: tuple[CalibrationTarget, ...] = (
    CalibrationTarget("center", 0.50, 0.50),
    CalibrationTarget("top_left", 0.12, 0.12),
    CalibrationTarget("top_center", 0.50, 0.12),
    CalibrationTarget("top_right", 0.88, 0.12),
    CalibrationTarget("center_left", 0.12, 0.50),
    CalibrationTarget("center", 0.50, 0.50),
    CalibrationTarget("center_right", 0.88, 0.50),
    CalibrationTarget("bottom_left", 0.12, 0.88),
    CalibrationTarget("bottom_center", 0.50, 0.88),
    CalibrationTarget("bottom_right", 0.88, 0.88),
)

DEFAULT_RIDGE = 0.05


@dataclass(frozen=True, slots=True)
class CalibrationIdentity:
    """Display, camera, and capture geometry associated with a calibration."""

    display_id: str | None
    display_width: int
    display_height: int
    camera_index: int
    frame_width: int
    frame_height: int

    def __post_init__(self) -> None:
        if (
            min(
                self.display_width,
                self.display_height,
                self.frame_width,
                self.frame_height,
            )
            < 1
        ):
            raise ValueError("display and frame dimensions must be positive")
        if self.camera_index < 0:
            raise ValueError("camera_index must be zero or greater")

    def to_dict(self) -> dict[str, object]:
        return {
            "display": {
                "id": self.display_id,
                "width": self.display_width,
                "height": self.display_height,
            },
            "camera_index": self.camera_index,
            "frame": {"w": self.frame_width, "h": self.frame_height},
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CalibrationIdentity | None:
        display = data.get("display")
        frame = data.get("frame")
        camera_index = data.get("camera_index")
        if not isinstance(display, dict) or not isinstance(frame, dict):
            return None
        display_id = display.get("id")
        if display_id is not None and not isinstance(display_id, str):
            return None
        display_width = display.get("width")
        display_height = display.get("height")
        frame_width = frame.get("w")
        frame_height = frame.get("h")
        if (
            not isinstance(display_width, int)
            or isinstance(display_width, bool)
            or not isinstance(display_height, int)
            or isinstance(display_height, bool)
            or not isinstance(camera_index, int)
            or isinstance(camera_index, bool)
            or not isinstance(frame_width, int)
            or isinstance(frame_width, bool)
            or not isinstance(frame_height, int)
            or isinstance(frame_height, bool)
        ):
            return None
        try:
            return cls(
                display_id=display_id,
                display_width=display_width,
                display_height=display_height,
                camera_index=camera_index,
                frame_width=frame_width,
                frame_height=frame_height,
            )
        except ValueError:
            return None

    def matches(self, expected: CalibrationIdentity) -> bool:
        id_matches = expected.display_id is None or self.display_id == expected.display_id
        return id_matches and (
            self.display_width,
            self.display_height,
            self.camera_index,
            self.frame_width,
            self.frame_height,
        ) == (
            expected.display_width,
            expected.display_height,
            expected.camera_index,
            expected.frame_width,
            expected.frame_height,
        )


@dataclass(frozen=True)
class CalibrationModel:
    coef_x: tuple[float, ...]
    coef_y: tuple[float, ...]
    feature_names: tuple[str, ...]
    samples: dict[str, int]
    quality: dict[str, str]
    created: str
    version: int = CALIBRATION_VERSION
    ridge: float = DEFAULT_RIDGE
    identity: CalibrationIdentity | None = None

    def predict(self, features: GazeFeatures) -> tuple[float, float]:
        vec = features.vector()
        cx = np.array(self.coef_x, dtype=np.float64)
        cy = np.array(self.coef_y, dtype=np.float64)
        if vec.shape[0] != cx.shape[0]:
            return uncalibrated_map(features)
        return clamp01(float(vec @ cx)), clamp01(float(vec @ cy))

    def to_dict(self) -> dict[str, object]:
        if self.version >= CALIBRATION_VERSION and self.identity is None:
            raise ValueError("calibration version 3 requires display, camera, and frame identity")
        data: dict[str, object] = {
            "version": self.version,
            "created": self.created,
            "ridge": self.ridge,
            "coef_x": list(self.coef_x),
            "coef_y": list(self.coef_y),
            "feature_names": list(self.feature_names),
            "samples": self.samples,
            "quality": self.quality,
        }
        if self.identity is not None:
            data.update(self.identity.to_dict())
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CalibrationModel | None:
        version_raw = data.get("version", 1)
        if not isinstance(version_raw, int):
            return None
        if version_raw < MINIMUM_CALIBRATION_VERSION:
            return None
        names_raw = data.get("feature_names", ())
        if not isinstance(names_raw, tuple | list):
            return None
        names = tuple(str(v) for v in names_raw)
        if names and names != FEATURE_NAMES:
            return None
        coef_x_raw = data.get("coef_x")
        coef_y_raw = data.get("coef_y")
        if not isinstance(coef_x_raw, list) or not isinstance(coef_y_raw, list):
            return None
        samples_raw = data.get("samples", {})
        quality_raw = data.get("quality", {})
        if not isinstance(samples_raw, dict) or not isinstance(quality_raw, dict):
            return None
        ridge_raw = data.get("ridge", DEFAULT_RIDGE)
        if not isinstance(ridge_raw, int | float):
            ridge_raw = DEFAULT_RIDGE
        identity = None
        if version_raw >= CALIBRATION_VERSION:
            identity = CalibrationIdentity.from_dict(data)
            if identity is None:
                return None
        return cls(
            coef_x=tuple(float(v) for v in coef_x_raw),
            coef_y=tuple(float(v) for v in coef_y_raw),
            feature_names=FEATURE_NAMES,
            samples={str(k): int(v) for k, v in samples_raw.items()},
            quality={str(k): str(v) for k, v in quality_raw.items()},
            created=str(data.get("created", "")),
            version=version_raw,
            ridge=float(ridge_raw),
            identity=identity,
        )


def robust_median_rows(rows: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    stacked = np.vstack(rows)
    median = np.median(stacked, axis=0)
    return np.asarray(median, dtype=np.float64)


def quality_label(count: float, min_good: int) -> str:
    if count >= min_good:
        return "GOOD"
    if count >= max(4, min_good // 2):
        return "FAIR"
    return "POOR"


def fit_ridge(
    feature_rows: list[NDArray[np.float64]],
    screen_xy: list[tuple[float, float]],
    sample_counts: dict[str, int],
    qualities: dict[str, str],
    ridge: float = DEFAULT_RIDGE,
    *,
    identity: CalibrationIdentity | None = None,
) -> CalibrationModel:
    if len(feature_rows) < 3:
        msg = "Need at least 3 calibration points to fit a mapping."
        raise ValueError(msg)
    x_mat = np.vstack(feature_rows)
    y_x = np.array([p[0] for p in screen_xy], dtype=np.float64)
    y_y = np.array([p[1] for p in screen_xy], dtype=np.float64)
    n_feat = x_mat.shape[1]
    penalty = ridge * np.eye(n_feat, dtype=np.float64)
    penalty[0, 0] = 0.0
    xtx = x_mat.T @ x_mat + penalty
    coef_x = np.linalg.solve(xtx, x_mat.T @ y_x)
    coef_y = np.linalg.solve(xtx, x_mat.T @ y_y)
    return CalibrationModel(
        coef_x=tuple(float(v) for v in coef_x),
        coef_y=tuple(float(v) for v in coef_y),
        feature_names=FEATURE_NAMES,
        samples=sample_counts,
        quality=qualities,
        created=datetime.now(UTC).isoformat(),
        version=(CALIBRATION_VERSION if identity is not None else MINIMUM_CALIBRATION_VERSION),
        ridge=ridge,
        identity=identity,
    )


def uncalibrated_map(features: GazeFeatures) -> tuple[float, float]:
    dx = (features.iris_nx - 0.5) * 2.2 + 0.35 * features.blend_x
    dy = (features.iris_ny - 0.5) * 1.6 + 0.85 * features.blend_y
    # Compensate head motion using the documented post-mirror convention:
    # rightward turn and downward nod are positive.
    dx -= 0.15 * (features.yaw_deg / 30.0)
    dy -= 0.10 * (features.pitch_deg / 30.0)
    return clamp01(0.5 + dx), clamp01(0.5 + dy)


def save_model(model: CalibrationModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")


def load_model(
    path: Path,
    *,
    expect: CalibrationIdentity | None = None,
) -> CalibrationModel | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "coef_x" not in data:
        return None
    model = CalibrationModel.from_dict(data)
    if model is None:
        return None
    if expect is not None and (model.identity is None or not model.identity.matches(expect)):
        return None
    return model
