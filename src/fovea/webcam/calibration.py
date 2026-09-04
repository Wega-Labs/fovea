"""10-point ridge calibration: eye/face features → normalized screen XY."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fovea.util import clamp01
from fovea.webcam.features import FEATURE_NAMES, GazeFeatures

CALIBRATION_VERSION = 4
IDENTITY_VERSION = 3
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

    def to_dict(self) -> dict[str, object]:
        return {"label": self.label, "x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CalibrationTarget | None:
        label = data.get("label")
        x = data.get("x")
        y = data.get("y")
        if not isinstance(label, str) or not label.strip():
            return None
        if (
            not isinstance(x, int | float)
            or isinstance(x, bool)
            or not isinstance(y, int | float)
            or isinstance(y, bool)
        ):
            return None
        return cls(label=label, x=float(x), y=float(y))


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


def validate_calibration_targets(
    targets: Sequence[CalibrationTarget],
) -> tuple[CalibrationTarget, ...]:
    result = tuple(targets)
    if len(result) < 5:
        raise ValueError("calibration requires at least 5 targets")
    for target in result:
        if not target.label.strip():
            raise ValueError("calibration target labels must be non-empty")
        if not math.isfinite(target.x) or not math.isfinite(target.y):
            raise ValueError("calibration target coordinates must be finite")
        if not 0.0 <= target.x <= 1.0 or not 0.0 <= target.y <= 1.0:
            raise ValueError("calibration target coordinates must be within [0, 1]")
    return result


def calibration_coverage(targets: Sequence[CalibrationTarget]) -> tuple[float, float]:
    validated = validate_calibration_targets(targets)
    x_values = [target.x for target in validated]
    y_values = [target.y for target in validated]
    return max(x_values) - min(x_values), max(y_values) - min(y_values)


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


@dataclass(frozen=True, slots=True)
class CalibrationAnchor:
    """One pinned wizard anchor used by every online refit."""

    row: tuple[float, ...]
    xy: tuple[float, float]

    def to_dict(self) -> dict[str, object]:
        return {"row": list(self.row), "xy": list(self.xy)}


@dataclass(frozen=True, slots=True)
class OnlineObservation:
    """One trusted host-confirmed observation in commit order."""

    row: tuple[float, ...]
    xy: tuple[float, float]
    host_weight: float
    commit_seq: int

    def to_dict(self) -> dict[str, object]:
        return {
            "row": list(self.row),
            "xy": list(self.xy),
            "host_weight": self.host_weight,
            "commit_seq": self.commit_seq,
        }


def _parse_row(value: object) -> tuple[float, ...] | None:
    if not isinstance(value, list) or len(value) != len(FEATURE_NAMES):
        return None
    row: list[float] = []
    for item in value:
        if not isinstance(item, int | float) or isinstance(item, bool):
            return None
        parsed = float(item)
        if not math.isfinite(parsed):
            return None
        row.append(parsed)
    return tuple(row)


def _parse_xy(value: object) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    parsed: list[float] = []
    for item in value:
        if not isinstance(item, int | float) or isinstance(item, bool):
            return None
        coordinate = float(item)
        if not math.isfinite(coordinate) or not 0.0 <= coordinate <= 1.0:
            return None
        parsed.append(coordinate)
    return parsed[0], parsed[1]


def _parse_anchors(value: object) -> tuple[CalibrationAnchor, ...] | None:
    if not isinstance(value, list):
        return None
    anchors: list[CalibrationAnchor] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"row", "xy"}:
            return None
        row = _parse_row(item["row"])
        xy = _parse_xy(item["xy"])
        if row is None or xy is None:
            return None
        anchors.append(CalibrationAnchor(row, xy))
    return tuple(anchors)


def _parse_observations(value: object) -> tuple[OnlineObservation, ...] | None:
    if not isinstance(value, list):
        return None
    observations: list[OnlineObservation] = []
    previous_seq = -1
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "row",
            "xy",
            "host_weight",
            "commit_seq",
        }:
            return None
        row = _parse_row(item["row"])
        xy = _parse_xy(item["xy"])
        weight_raw = item["host_weight"]
        seq_raw = item["commit_seq"]
        if row is None or xy is None:
            return None
        if not isinstance(weight_raw, int | float) or isinstance(weight_raw, bool):
            return None
        weight = float(weight_raw)
        if not math.isfinite(weight) or not 0.0 < weight <= 1.0:
            return None
        if not isinstance(seq_raw, int) or isinstance(seq_raw, bool) or seq_raw < 0:
            return None
        if seq_raw <= previous_seq:
            return None
        observations.append(OnlineObservation(row, xy, weight, seq_raw))
        previous_seq = seq_raw
    return tuple(observations)


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
    targets: tuple[CalibrationTarget, ...] = ()
    anchors: tuple[CalibrationAnchor, ...] = ()
    observations: tuple[OnlineObservation, ...] = ()
    baseline_anchor_error: float | None = None
    n: int = 0
    commit_seq: int = 0

    def predict(self, features: GazeFeatures) -> tuple[float, float]:
        vec = features.vector()
        cx = np.array(self.coef_x, dtype=np.float64)
        cy = np.array(self.coef_y, dtype=np.float64)
        if vec.shape[0] != cx.shape[0]:
            return uncalibrated_map(features)
        return clamp01(float(vec @ cx)), clamp01(float(vec @ cy))

    def to_dict(self) -> dict[str, object]:
        if self.version >= IDENTITY_VERSION and self.identity is None:
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
        if self.targets:
            data["targets"] = [target.to_dict() for target in self.targets]
        if self.version >= CALIBRATION_VERSION:
            if self.baseline_anchor_error is None:
                raise ValueError("calibration version 4 requires an anchor baseline")
            data.update(
                {
                    "anchors": [anchor.to_dict() for anchor in self.anchors],
                    "observations": [observation.to_dict() for observation in self.observations],
                    "baseline_anchor_error": self.baseline_anchor_error,
                    "n": self.n,
                    "commit_seq": self.commit_seq,
                }
            )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CalibrationModel | None:
        version_raw = data.get("version", 1)
        if not isinstance(version_raw, int) or isinstance(version_raw, bool):
            return None
        if version_raw < MINIMUM_CALIBRATION_VERSION:
            return None
        names_raw = data.get("feature_names", ())
        if not isinstance(names_raw, tuple | list):
            return None
        names = tuple(str(v) for v in names_raw)
        if names and names != FEATURE_NAMES:
            return None
        if version_raw >= CALIBRATION_VERSION and names != FEATURE_NAMES:
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
        if version_raw >= IDENTITY_VERSION:
            identity = CalibrationIdentity.from_dict(data)
            if identity is None:
                return None
        targets_raw = data.get("targets", [])
        if not isinstance(targets_raw, list):
            return None
        targets: list[CalibrationTarget] = []
        for target_raw in targets_raw:
            if not isinstance(target_raw, dict):
                return None
            target = CalibrationTarget.from_dict(target_raw)
            if target is None:
                return None
            targets.append(target)
        if targets:
            try:
                validate_calibration_targets(targets)
            except ValueError:
                return None
        anchors: tuple[CalibrationAnchor, ...] = ()
        observations: tuple[OnlineObservation, ...] = ()
        baseline_anchor_error: float | None = None
        n = 0
        commit_seq = 0
        if version_raw >= CALIBRATION_VERSION:
            parsed_anchors = _parse_anchors(data.get("anchors"))
            parsed_observations = _parse_observations(data.get("observations"))
            baseline_raw = data.get("baseline_anchor_error")
            n_raw = data.get("n")
            commit_seq_raw = data.get("commit_seq")
            if parsed_anchors is None or parsed_observations is None:
                return None
            if len(parsed_anchors) < 3 or len(parsed_observations) > 200:
                return None
            if (
                not isinstance(baseline_raw, int | float)
                or isinstance(baseline_raw, bool)
                or not math.isfinite(float(baseline_raw))
                or float(baseline_raw) < 0.0
            ):
                return None
            if not isinstance(n_raw, int) or isinstance(n_raw, bool) or n_raw < 0:
                return None
            if (
                not isinstance(commit_seq_raw, int)
                or isinstance(commit_seq_raw, bool)
                or commit_seq_raw < 0
            ):
                return None
            if any(observation.commit_seq > commit_seq_raw for observation in parsed_observations):
                return None
            if n_raw < len(parsed_observations) or commit_seq_raw > n_raw:
                return None
            if targets and (
                len(parsed_anchors) != len(targets)
                or any(
                    anchor.xy != (target.x, target.y)
                    for anchor, target in zip(parsed_anchors, targets, strict=True)
                )
            ):
                return None
            anchors = parsed_anchors
            observations = parsed_observations
            baseline_anchor_error = float(baseline_raw)
            n = n_raw
            commit_seq = commit_seq_raw
        try:
            coef_x = tuple(float(v) for v in coef_x_raw)
            coef_y = tuple(float(v) for v in coef_y_raw)
            if not all(math.isfinite(v) for v in (*coef_x, *coef_y)):
                return None
            if version_raw >= CALIBRATION_VERSION and (
                len(coef_x) != len(FEATURE_NAMES) or len(coef_y) != len(FEATURE_NAMES)
            ):
                return None
            ridge = float(ridge_raw)
            if not math.isfinite(ridge) or ridge < 0.0:
                return None
            samples = {str(k): int(v) for k, v in samples_raw.items()}
        except (TypeError, ValueError):
            return None
        return cls(
            coef_x=coef_x,
            coef_y=coef_y,
            feature_names=FEATURE_NAMES,
            samples=samples,
            quality={str(k): str(v) for k, v in quality_raw.items()},
            created=str(data.get("created", "")),
            version=version_raw,
            ridge=ridge,
            identity=identity,
            targets=tuple(targets),
            anchors=anchors,
            observations=observations,
            baseline_anchor_error=baseline_anchor_error,
            n=n,
            commit_seq=commit_seq,
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
    targets: Sequence[CalibrationTarget] = (),
) -> CalibrationModel:
    if len(feature_rows) < 3:
        msg = "Need at least 3 calibration points to fit a mapping."
        raise ValueError(msg)
    if len(feature_rows) != len(screen_xy):
        raise ValueError("calibration features and screen targets must have equal lengths")
    saved_targets = tuple(targets)
    if saved_targets:
        saved_targets = validate_calibration_targets(saved_targets)
        if len(saved_targets) != len(feature_rows):
            raise ValueError("stored calibration targets must match the fitted rows")
    x_mat = np.vstack(feature_rows)
    y_x = np.array([p[0] for p in screen_xy], dtype=np.float64)
    y_y = np.array([p[1] for p in screen_xy], dtype=np.float64)
    coef_x = _ridge_coefficients(x_mat, y_x, ridge)
    coef_y = _ridge_coefficients(x_mat, y_y, ridge)
    version = CALIBRATION_VERSION if identity is not None else MINIMUM_CALIBRATION_VERSION
    anchors: tuple[CalibrationAnchor, ...] = ()
    baseline_anchor_error: float | None = None
    if version >= CALIBRATION_VERSION:
        anchors = tuple(
            CalibrationAnchor(
                tuple(float(value) for value in row),
                (float(point[0]), float(point[1])),
            )
            for row, point in zip(feature_rows, screen_xy, strict=True)
        )
        predictions = np.column_stack((x_mat @ coef_x, x_mat @ coef_y))
        baseline_anchor_error = float(
            np.mean(np.linalg.norm(np.clip(predictions, 0.0, 1.0) - np.asarray(screen_xy), axis=1))
        )
    return CalibrationModel(
        coef_x=tuple(float(v) for v in coef_x),
        coef_y=tuple(float(v) for v in coef_y),
        feature_names=FEATURE_NAMES,
        samples=sample_counts,
        quality=qualities,
        created=datetime.now(UTC).isoformat(),
        version=version,
        ridge=ridge,
        identity=identity,
        targets=saved_targets,
        anchors=anchors,
        baseline_anchor_error=baseline_anchor_error,
    )


def _ridge_coefficients(
    feature_matrix: NDArray[np.float64],
    values: NDArray[np.float64],
    ridge: float,
) -> NDArray[np.float64]:
    n_features = feature_matrix.shape[1]
    penalty = ridge * np.eye(n_features, dtype=np.float64)
    penalty[0, 0] = 0.0
    system = feature_matrix.T @ feature_matrix + penalty
    return np.asarray(
        np.linalg.solve(system, feature_matrix.T @ values),
        dtype=np.float64,
    )


def online_refit(
    feature_rows: Sequence[NDArray[np.float64]],
    screen_xy: Sequence[tuple[float, float]],
    weights: Sequence[float],
    ridge: float = DEFAULT_RIDGE,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit a weighted ridge map after normalizing effective weights to mean one."""
    if len(feature_rows) != len(screen_xy) or len(feature_rows) != len(weights):
        raise ValueError("online refit features, targets, and weights must have equal lengths")
    if len(feature_rows) < 3:
        raise ValueError("online refit requires at least 3 paired rows")
    matrix = np.vstack(feature_rows).astype(np.float64, copy=False)
    expected = np.asarray(screen_xy, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    if expected.shape != (len(feature_rows), 2):
        raise ValueError("online refit targets must contain XY pairs")
    if matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError("online refit rows use an incompatible feature vector")
    if (
        not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(expected))
        or not np.all(np.isfinite(weight_array))
        or np.any(weight_array <= 0.0)
        or not math.isfinite(ridge)
        or ridge < 0.0
    ):
        raise ValueError("online refit inputs must be finite with positive weights")
    normalized = weight_array / float(np.mean(weight_array))
    scale = np.sqrt(normalized)
    weighted_matrix = matrix * scale[:, None]
    return (
        _ridge_coefficients(weighted_matrix, expected[:, 0] * scale, ridge),
        _ridge_coefficients(weighted_matrix, expected[:, 1] * scale, ridge),
    )


def weighted_leave_one_out_error(
    feature_rows: Sequence[NDArray[np.float64]],
    screen_xy: Sequence[tuple[float, float]],
    weights: Sequence[float],
    ridge: float = DEFAULT_RIDGE,
) -> float:
    """Return weighted normalized point error with each online row held out once."""
    if len(feature_rows) != len(screen_xy) or len(feature_rows) != len(weights):
        raise ValueError("weighted leave-one-out inputs must have equal lengths")
    if len(feature_rows) < 4:
        raise ValueError("weighted leave-one-out validation requires at least 4 paired rows")
    matrix = np.vstack(feature_rows).astype(np.float64, copy=False)
    expected = np.asarray(screen_xy, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    if not np.all(np.isfinite(weight_array)) or np.any(weight_array <= 0.0):
        raise ValueError("weighted leave-one-out weights must be finite and positive")
    errors: list[float] = []
    for index in range(len(feature_rows)):
        keep = np.arange(len(feature_rows)) != index
        coef_x, coef_y = online_refit(
            [matrix[row] for row in np.flatnonzero(keep)],
            [tuple(point) for point in expected[keep]],
            [float(weight) for weight in weight_array[keep]],
            ridge,
        )
        predicted = np.clip(
            np.array([matrix[index] @ coef_x, matrix[index] @ coef_y], dtype=np.float64),
            0.0,
            1.0,
        )
        errors.append(float(np.linalg.norm(predicted - expected[index])))
    return float(np.average(np.asarray(errors), weights=weight_array))


def leave_one_out_error(
    feature_rows: Sequence[NDArray[np.float64]],
    screen_xy: Sequence[tuple[float, float]],
    ridge: float = DEFAULT_RIDGE,
) -> float:
    """Return mean normalized point error with each calibration target held out once."""
    if len(feature_rows) != len(screen_xy) or len(feature_rows) < 4:
        raise ValueError("leave-one-out validation requires at least 4 paired rows")
    matrix = np.vstack(feature_rows)
    expected = np.asarray(screen_xy, dtype=np.float64)
    errors: list[float] = []
    for index in range(len(feature_rows)):
        train_x = np.delete(matrix, index, axis=0)
        train_y = np.delete(expected, index, axis=0)
        coef_x = _ridge_coefficients(train_x, train_y[:, 0], ridge)
        coef_y = _ridge_coefficients(train_x, train_y[:, 1], ridge)
        held_out = matrix[index]
        predicted_x = clamp01(float(held_out @ coef_x))
        predicted_y = clamp01(float(held_out @ coef_y))
        errors.append(
            float(
                np.hypot(
                    predicted_x - expected[index, 0],
                    predicted_y - expected[index, 1],
                )
            )
        )
    return float(np.mean(errors))


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
    rendered = json.dumps(model.to_dict(), indent=2)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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
