"""10-point ridge calibration: eye/face features → normalized screen XY."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from fovea.util import clamp01
from fovea.webcam.features import FEATURE_NAMES, GazeFeatures

CALIBRATION_VERSION = 2

CALIBRATION_LAYOUT: tuple[tuple[str, float, float], ...] = (
    ("center", 0.50, 0.50),
    ("top_left", 0.12, 0.12),
    ("top_center", 0.50, 0.12),
    ("top_right", 0.88, 0.12),
    ("center_left", 0.12, 0.50),
    ("center", 0.50, 0.50),
    ("center_right", 0.88, 0.50),
    ("bottom_left", 0.12, 0.88),
    ("bottom_center", 0.50, 0.88),
    ("bottom_right", 0.88, 0.88),
)

DEFAULT_RIDGE = 0.05


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

    def predict(self, features: GazeFeatures) -> tuple[float, float]:
        vec = features.vector()
        cx = np.array(self.coef_x, dtype=np.float64)
        cy = np.array(self.coef_y, dtype=np.float64)
        if vec.shape[0] != cx.shape[0]:
            return uncalibrated_map(features)
        return clamp01(float(vec @ cx)), clamp01(float(vec @ cy))

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "created": self.created,
            "ridge": self.ridge,
            "coef_x": list(self.coef_x),
            "coef_y": list(self.coef_y),
            "feature_names": list(self.feature_names),
            "samples": self.samples,
            "quality": self.quality,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CalibrationModel | None:
        version_raw = data.get("version", 1)
        if not isinstance(version_raw, int):
            return None
        if version_raw < CALIBRATION_VERSION:
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
        return cls(
            coef_x=tuple(float(v) for v in coef_x_raw),
            coef_y=tuple(float(v) for v in coef_y_raw),
            feature_names=FEATURE_NAMES,
            samples={str(k): int(v) for k, v in samples_raw.items()},
            quality={str(k): str(v) for k, v in quality_raw.items()},
            created=str(data.get("created", "")),
            version=version_raw,
            ridge=float(ridge_raw),
        )


def robust_median_rows(rows: list[np.ndarray]) -> np.ndarray:
    stacked = np.vstack(rows)
    median = np.median(stacked, axis=0)
    return np.asarray(median, dtype=np.float64)


def quality_label(count: int, min_good: int) -> str:
    if count >= min_good:
        return "GOOD"
    if count >= max(4, min_good // 2):
        return "FAIR"
    return "POOR"


def fit_ridge(
    feature_rows: list[np.ndarray],
    screen_xy: list[tuple[float, float]],
    sample_counts: dict[str, int],
    qualities: dict[str, str],
    ridge: float = DEFAULT_RIDGE,
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
        version=CALIBRATION_VERSION,
        ridge=ridge,
    )


def uncalibrated_map(features: GazeFeatures) -> tuple[float, float]:
    dx = (features.iris_nx - 0.5) * 2.2 + 0.35 * features.blend_x
    dy = (features.iris_ny - 0.5) * 1.6 + 0.85 * features.blend_y
    dx -= 0.15 * (features.yaw_deg / 30.0)
    dy -= 0.10 * (features.pitch_deg / 30.0)
    return clamp01(0.5 + dx), clamp01(0.5 + dy)


def save_model(model: CalibrationModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")


def load_model(path: Path) -> CalibrationModel | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "coef_x" not in data:
        return None
    return CalibrationModel.from_dict(data)
