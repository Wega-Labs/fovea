"""Gaze pipeline: features → calibration map → smooth → Fovea events."""

from __future__ import annotations

import math
import time as _time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fovea.paths import default_calibration_path
from fovea.util import ScreenPoint, clamp01
from fovea.webcam.calibration import (
    CALIBRATION_LAYOUT,
    CalibrationIdentity,
    CalibrationModel,
    CalibrationTarget,
    OnlineObservation,
    calibration_coverage,
    fit_ridge,
    leave_one_out_error,
    load_model,
    online_refit,
    save_model,
    uncalibrated_map,
    validate_calibration_targets,
    weighted_leave_one_out_error,
)
from fovea.webcam.features import GazeFeatures, extract_features
from fovea.webcam.sampler import PointCollector
from fovea.webcam.smoothing import OneEuroPoint, ema


@dataclass
class GazeSettings:
    """Gaze pipeline, target selection, dwell, and event-vocabulary thresholds.

    The saccade, pursuit, wink, and blink-trigger values are heuristics tuned
    on synthetic streams rather than guarantees. Velocities are in
    display-normalized units per second. ``pursuit_velocity`` marks the slowest
    motion that counts as moving; ``saccade_velocity`` is the I-VT threshold
    above it. A pursuit is reported only after ``pursuit_ms`` of continuous
    in-band motion whose net displacement over path length reaches
    ``pursuit_coherence``. A wink must last between ``wink_min_ms`` and
    ``wink_max_ms``. Two natural blinks within ``double_blink_ms`` form a double
    blink. ``long_blink_ms`` is the cold-start long-blink floor, raised to
    ``long_blink_factor`` times the median of the last ``natural_blink_window``
    natural blinks once enough have been observed.
    """

    dwell_ms: float = 500.0
    stability_ms: float = 300.0
    hysteresis: float = 0.04
    target_expand: float = 0.03
    snap_radius: float = 0.06
    smoothing_alpha: float = 0.35
    one_euro_mincutoff: float = 1.2
    one_euro_beta: float = 0.02
    blink_ear: float = 0.16
    min_face_width: float = 0.12
    max_yaw_deg: float = 35.0
    samples_per_point: int = 28
    min_good_samples: int = 12
    settle_frames: int = 12
    calibration_path: str | Path | None = None
    debug: bool = True
    online_calibration: bool = True
    saccade_velocity: float = 1.5
    pursuit_velocity: float = 0.3
    pursuit_ms: float = 100.0
    pursuit_coherence: float = 0.7
    wink_min_ms: float = 120.0
    wink_max_ms: float = 700.0
    double_blink_ms: float = 500.0
    long_blink_ms: float = 600.0
    long_blink_factor: float = 2.0
    natural_blink_window: int = 30


@dataclass
class GazeOutput:
    valid: bool
    tracking: str
    message: str
    features: GazeFeatures | None
    screen: ScreenPoint | None
    confidence: float
    fps: float
    calibrated: bool
    frozen: bool
    latency_ms: float = 0.0


@dataclass
class WizardState:
    kind: str
    index: int
    label: str
    sx: float
    sy: float
    samples: int
    needed: int
    quality: str
    instruction: str
    done: bool = False
    report: dict[str, object] = field(default_factory=dict)
    settle_left: int = 0


@dataclass(frozen=True, slots=True)
class OnlineCalibrationReport:
    n: int
    loo_error: float
    refit_ts: int


@dataclass(frozen=True, slots=True)
class _PendingObservation:
    row: tuple[float, ...]
    xy: tuple[float, float]
    host_weight: float
    residual: tuple[float, float]
    admission_epoch: int


FEATURE_RING_SIZE = 60
ASSOCIATION_WINDOW_NS = 200_000_000
ADMISSION_CAP = 0.6
CLUSTER_ABS_FLOOR = 0.05
CLUSTER_MAD_MULTIPLIER = 3.0
CLUSTER_MIN_SUPPORT = 3
QUARANTINE_EPOCHS = 3
REFIT_CADENCE = 5
ONLINE_BUFFER_SIZE = 200
DECAY_HALF_LIFE = 30.0
HUBER_DELTA = 0.05


class GazeEngine:
    def __init__(
        self,
        settings: GazeSettings,
        project_root: Path | None = None,
        identity: CalibrationIdentity | None = None,
        clock: Callable[[], int] = _time.time_ns,
    ) -> None:
        self.settings = settings
        self.identity = identity
        del project_root
        if settings.calibration_path is None:
            self.path = default_calibration_path()
        else:
            self.path = Path(settings.calibration_path).expanduser()
            if not self.path.is_absolute():
                raise ValueError("calibration_path must be absolute or None")
        self.model = load_model(self.path, expect=identity)
        self._clock = clock
        self._filter = OneEuroPoint(settings.one_euro_mincutoff, settings.one_euro_beta)
        self._last_screen: ScreenPoint | None = None
        self._history: deque[tuple[float, float]] = deque(maxlen=12)
        self._frozen: ScreenPoint | None = None
        self.wizard: WizardState | None = None
        self._collectors: list[PointCollector] = []
        self._test_preds: list[tuple[float, float, float, float]] = []
        self.targets: tuple[CalibrationTarget, ...] = CALIBRATION_LAYOUT
        self.coverage_x, self.coverage_y = calibration_coverage(self.targets)
        self.calibration_warning = ""
        self.last_calibration_report: dict[str, object] = {}
        self.last_test_report: dict[str, object] = {}
        self._feature_ring: deque[tuple[int, tuple[float, ...]]] = deque(maxlen=FEATURE_RING_SIZE)
        self._latest_frame_timestamp_ns: int | None = None
        self._quarantine: list[_PendingObservation] = []
        self._promoted: list[_PendingObservation] = []
        self._trusted: list[OnlineObservation] = []
        self._promoted_count = 0
        self._admission_epoch = 0
        self._online_reports: deque[OnlineCalibrationReport] = deque()
        self._last_refit_ts: int | None = None
        self._online_n = 0
        self._commit_seq = 0
        self._seed_online_state()

    def _seed_online_state(self) -> None:
        self._trusted.clear()
        self._online_n = 0
        self._commit_seq = 0
        if not self.settings.online_calibration or self.model is None:
            return
        if not self.model.anchors or self.model.baseline_anchor_error is None:
            return
        self._trusted.extend(self.model.observations[-ONLINE_BUFFER_SIZE:])
        self._online_n = self.model.n
        self._commit_seq = self.model.commit_seq

    def _clear_online_session(self) -> None:
        self._feature_ring.clear()
        self._latest_frame_timestamp_ns = None
        self._quarantine.clear()
        self._promoted.clear()
        self._trusted.clear()
        self._promoted_count = 0
        self._admission_epoch = 0
        self._online_reports.clear()
        self._last_refit_ts = None
        self._online_n = 0
        self._commit_seq = 0

    def start_calibration(
        self,
        targets: Sequence[CalibrationTarget] | None = None,
    ) -> None:
        self._set_targets(CALIBRATION_LAYOUT if targets is None else targets)
        self.model = None
        self._clear_online_session()
        self._filter.reset()
        self._last_screen = None
        self._collectors = [
            PointCollector(self.settings.samples_per_point, self.settings.min_good_samples)
            for _ in self.targets
        ]
        self.last_calibration_report = {}
        self._set_wizard("calibrate", 0)

    def start_gaze_test(
        self,
        targets: Sequence[CalibrationTarget] | None = None,
    ) -> None:
        self._feature_ring.clear()
        selected = targets
        if selected is None and self.model is not None and self.model.targets:
            selected = self.model.targets
        self._set_targets(CALIBRATION_LAYOUT if selected is None else selected)
        self._test_preds = []
        self.last_test_report = {}
        self._collectors = [
            PointCollector(self.settings.samples_per_point, self.settings.min_good_samples)
            for _ in self.targets
        ]
        self._set_wizard("test", 0)

    def resume_after_gaze_test(self) -> None:
        """Return to live gaze output after a completed guided test."""
        if self.wizard is None or self.wizard.kind != "test" or not self.wizard.done:
            return
        self.wizard = None
        self._filter.reset()
        self._last_screen = None

    def _set_targets(self, targets: Sequence[CalibrationTarget]) -> None:
        self.targets = validate_calibration_targets(targets)
        self.coverage_x, self.coverage_y = calibration_coverage(self.targets)
        if self.coverage_x < 0.4 or self.coverage_y < 0.4:
            self.calibration_warning = (
                "Calibration targets cover less than 40% of the display on at least one axis."
            )
        else:
            self.calibration_warning = ""

    def _set_wizard(self, kind: str, index: int) -> None:
        if index >= len(self.targets):
            self.wizard = None
            return
        target = self.targets[index]
        collector = self._collectors[index]
        self.wizard = WizardState(
            kind=kind,
            index=index,
            label=target.label,
            sx=target.x,
            sy=target.y,
            samples=collector.count,
            needed=self.settings.samples_per_point,
            quality=collector.quality(),
            instruction="Look at the point.",
            settle_left=self.settings.settle_frames,
        )

    def process(
        self,
        landmarks: Sequence[Any] | None,
        image_w: float,
        image_h: float,
        dt: float,
        fps: float,
        blendshapes: Mapping[str, float] | None = None,
        timestamp_ns: int | None = None,
    ) -> GazeOutput:
        t0 = _time.perf_counter()
        if timestamp_ns is not None:
            self._latest_frame_timestamp_ns = timestamp_ns
        if landmarks is None:
            self._feature_ring.clear()
            return GazeOutput(
                False,
                "LOST",
                "Face not detected",
                None,
                self._frozen,
                0.0,
                fps,
                self.model is not None,
                True,
                latency_ms=0.0,
            )
        features = extract_features(
            landmarks,
            image_w,
            image_h,
            self.settings.blink_ear,
            self.settings.min_face_width,
            self.settings.max_yaw_deg,
            blendshapes=blendshapes,
        )
        if features.tracking == "LOST":
            self._feature_ring.clear()
        if self.wizard is not None:
            if not self.wizard.done:
                self._wizard_sample(features)
            latency = (_time.perf_counter() - t0) * 1000.0
            return GazeOutput(
                False,
                features.tracking,
                self.wizard.instruction if self.wizard else "",
                features,
                ScreenPoint(self.wizard.sx, self.wizard.sy) if self.wizard else None,
                0.0,
                fps,
                self.model is not None,
                True,
                latency_ms=latency,
            )

        invalid = features.tracking == "LOST" or features.blink
        if invalid:
            latency = (_time.perf_counter() - t0) * 1000.0
            return GazeOutput(
                False,
                features.tracking if not features.blink else "FAIR",
                features.message or "Tracking lost",
                features,
                self._frozen,
                0.0,
                fps,
                self.model is not None,
                True,
                latency_ms=latency,
            )

        if timestamp_ns is not None and features.tracking in {"GOOD", "FAIR"}:
            self._feature_ring.append(
                (timestamp_ns, tuple(float(value) for value in features.vector()))
            )

        if self.model is not None:
            sx, sy = self.model.predict(features)
        else:
            sx, sy = uncalibrated_map(features)
        sx, sy = self._filter.filter(sx, sy, dt)
        if self.settings.smoothing_alpha < 1.0:
            sx = ema(
                self._last_screen.x if self._last_screen else None,
                sx,
                self.settings.smoothing_alpha,
            )
            sy = ema(
                self._last_screen.y if self._last_screen else None,
                sy,
                self.settings.smoothing_alpha,
            )
        screen = ScreenPoint(clamp01(sx), clamp01(sy))
        self._last_screen = screen
        self._frozen = screen
        self._history.append((screen.x, screen.y))
        conf = _confidence(features, screen, self._history)
        latency = (_time.perf_counter() - t0) * 1000.0
        return GazeOutput(
            True,
            features.tracking,
            features.message,
            features,
            screen,
            conf,
            fps,
            self.model is not None,
            False,
            latency_ms=latency,
        )

    def _wizard_sample(self, features: GazeFeatures) -> None:
        assert self.wizard is not None
        if self.wizard.settle_left > 0:
            self.wizard.settle_left -= 1
            self.wizard.instruction = (
                "Look at the point."
                if features.tracking in {"GOOD", "FAIR"}
                else (features.message or "Move closer / improve lighting / keep face visible.")
            )
            return
        collector = self._collectors[self.wizard.index]
        before = collector.count
        collector.add(features.vector(), features.tracking, features.blink)
        self.wizard.samples = collector.count
        self.wizard.quality = collector.quality()
        if collector.count == before and features.tracking == "POOR":
            self.wizard.instruction = features.message or (
                "Move closer / improve lighting / keep face visible."
            )
        elif self.wizard.kind == "calibrate":
            self.wizard.instruction = (
                f"Calibration point {self.wizard.index + 1}/{len(self.targets)}  "
                f"Samples: {collector.count}  Quality: {collector.quality()}"
            )
        else:
            self.wizard.instruction = (
                f"Test point {self.wizard.index + 1}/{len(self.targets)}  "
                f"Samples: {collector.count}"
            )
        if not collector.done():
            return
        if self.wizard.kind == "test" and self.model is not None:
            pred = self.model.predict(features)
            self._test_preds.append((self.wizard.sx, self.wizard.sy, pred[0], pred[1]))
        nxt = self.wizard.index + 1
        if nxt >= len(self.targets):
            if self.wizard.kind == "calibrate":
                self._finish_calibration()
            else:
                self._finish_test()
            return
        self._set_wizard(self.wizard.kind, nxt)

    def _finish_calibration(self) -> None:
        rows: list[NDArray[np.float64]] = []
        xy: list[tuple[float, float]] = []
        counts: dict[str, int] = {}
        qualities: dict[str, str] = {}
        for target, collector in zip(self.targets, self._collectors, strict=True):
            if collector.count == 0:
                continue
            rows.append(collector.median())
            xy.append((target.x, target.y))
            key = f"{target.label}_{len(counts)}"
            counts[key] = collector.count
            qualities[key] = collector.quality()
        if len(rows) >= 3:
            self.model = fit_ridge(
                rows,
                xy,
                counts,
                qualities,
                identity=self.identity,
                targets=self.targets,
            )
            save_model(self.model, self.path)
            self._seed_online_state()
            self.last_calibration_report = {
                "n_points": len(rows),
                "coverage": min(self.coverage_x, self.coverage_y),
                "loo_error": leave_one_out_error(rows, xy),
            }
        self.wizard = None
        self._filter.reset()
        self._last_screen = None

    def observe(
        self,
        x: float,
        y: float,
        weight: float = 1.0,
        timestamp_ns: int | None = None,
    ) -> None:
        """Associate a host-confirmed point with a recent feature row."""
        if (
            not self.settings.online_calibration
            or self.model is None
            or not self.model.anchors
            or self.model.baseline_anchor_error is None
            or self.wizard is not None
            or not self._feature_ring
        ):
            return
        if (
            not all(math.isfinite(value) for value in (x, y, weight))
            or not 0.0 <= x <= 1.0
            or not 0.0 <= y <= 1.0
            or not 0.0 < weight <= 1.0
        ):
            return
        associated: tuple[int, tuple[float, ...]] | None = None
        if timestamp_ns is None:
            candidate = self._feature_ring[-1]
            latest = self._latest_frame_timestamp_ns
            if latest is not None and 0 <= latest - candidate[0] <= ASSOCIATION_WINDOW_NS:
                associated = candidate
        elif isinstance(timestamp_ns, int) and not isinstance(timestamp_ns, bool):
            candidate = min(self._feature_ring, key=lambda item: abs(item[0] - timestamp_ns))
            if abs(candidate[0] - timestamp_ns) <= ASSOCIATION_WINDOW_NS:
                associated = candidate
        if associated is None:
            return
        row = associated[1]
        predicted = _predict_row(self.model, row)
        residual = (x - predicted[0], y - predicted[1])
        if math.hypot(*residual) > ADMISSION_CAP:
            return
        self._admission_epoch += 1
        self._quarantine.append(
            _PendingObservation(
                row=row,
                xy=(float(x), float(y)),
                host_weight=float(weight),
                residual=residual,
                admission_epoch=self._admission_epoch,
            )
        )
        self._cluster_quarantine()

    def _cluster_quarantine(self) -> None:
        pending = [*self._promoted, *self._quarantine]
        if len(pending) < CLUSTER_MIN_SUPPORT:
            self._expire_quarantine()
            return
        residuals = np.asarray([item.residual for item in pending], dtype=np.float64)
        center = _geometric_median(residuals)
        distances = np.linalg.norm(residuals - center, axis=1)
        median_distance = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median_distance)))
        if float(np.linalg.norm(center)) < 0.03 and mad < 0.02:
            self._expire_quarantine()
            return
        threshold = max(
            CLUSTER_ABS_FLOOR,
            median_distance + CLUSTER_MAD_MULTIPLIER * mad,
        )
        support = int(np.count_nonzero(distances <= threshold))
        if support >= CLUSTER_MIN_SUPPORT:
            promoted_count = len(self._promoted)
            new_promoted = [
                observation
                for index, observation in enumerate(self._quarantine, start=promoted_count)
                if distances[index] <= threshold
            ]
            if new_promoted:
                promoted_ids = {id(observation) for observation in new_promoted}
                self._quarantine = [
                    observation
                    for observation in self._quarantine
                    if id(observation) not in promoted_ids
                ]
                self._promoted.extend(new_promoted)
                self._promoted_count += len(new_promoted)
        self._expire_quarantine()
        if self._promoted_count >= REFIT_CADENCE:
            self._attempt_online_refit()

    def _expire_quarantine(self) -> None:
        self._quarantine = [
            observation
            for observation in self._quarantine
            if self._admission_epoch - observation.admission_epoch < QUARANTINE_EPOCHS
        ]

    def _attempt_online_refit(self) -> None:
        model = self.model
        promoted = tuple(self._promoted)
        if model is None or not promoted or model.baseline_anchor_error is None:
            return
        final_seq = self._commit_seq + len(promoted)
        prospective = [
            OnlineObservation(
                observation.row,
                observation.xy,
                observation.host_weight,
                self._commit_seq + index,
            )
            for index, observation in enumerate(promoted, start=1)
        ]
        observations = [*self._trusted, *prospective]
        observations.sort(key=lambda observation: observation.commit_seq)
        observations = observations[-ONLINE_BUFFER_SIZE:]
        rows = [np.asarray(anchor.row, dtype=np.float64) for anchor in model.anchors]
        xy = [anchor.xy for anchor in model.anchors]
        weights = [1.0] * len(model.anchors)
        for observation in observations:
            rows.append(np.asarray(observation.row, dtype=np.float64))
            xy.append(observation.xy)
            weights.append(
                observation.host_weight
                * 0.5 ** ((final_seq - observation.commit_seq) / DECAY_HALF_LIFE)
            )
        installed = False
        try:
            coef_x, coef_y = online_refit(rows, xy, weights, model.ridge)
            if not np.all(np.isfinite(coef_x)) or not np.all(np.isfinite(coef_y)):
                raise ValueError("online refit produced non-finite coefficients")
            current_error = _weighted_huber_error(model, rows, xy, weights)
            candidate_error = _weighted_huber_coefficients_error(coef_x, coef_y, rows, xy, weights)
            anchor_rows = rows[: len(model.anchors)]
            anchor_xy = xy[: len(model.anchors)]
            candidate_anchor_error = _mean_coefficient_error(coef_x, coef_y, anchor_rows, anchor_xy)
            anchor_ceiling = model.baseline_anchor_error * 1.1 + 0.02
            if candidate_error > current_error + 1e-12 or candidate_anchor_error > anchor_ceiling:
                raise ValueError("online refit failed the transaction guard")
            loo_error = weighted_leave_one_out_error(rows, xy, weights, model.ridge)
            refit_ts = self._clock()
            if self._last_refit_ts is not None:
                refit_ts = max(refit_ts, self._last_refit_ts)
            candidate = replace(
                model,
                coef_x=tuple(float(value) for value in coef_x),
                coef_y=tuple(float(value) for value in coef_y),
                observations=tuple(observations),
                n=self._online_n + len(promoted),
                commit_seq=final_seq,
            )
            save_model(candidate, self.path)
            self.model = candidate
            self._trusted = observations
            self._online_n = candidate.n
            self._commit_seq = candidate.commit_seq
            self._filter.reset()
            self._last_screen = None
            self._last_refit_ts = refit_ts
            self._online_reports.append(OnlineCalibrationReport(candidate.n, loo_error, refit_ts))
            installed = True
        except (OSError, ValueError, np.linalg.LinAlgError):
            pass
        finally:
            self._promoted.clear()
            if installed:
                self._promoted_count -= REFIT_CADENCE
            else:
                self._promoted_count = max(0, self._promoted_count - len(promoted))

    def drain_online_reports(self) -> tuple[OnlineCalibrationReport, ...]:
        """Pop all successful online-refit reports in installation order."""
        reports = tuple(self._online_reports)
        self._online_reports.clear()
        return reports

    def _finish_test(self) -> None:
        errors = [float(np.hypot(px - sx, py - sy)) for sx, sy, px, py in self._test_preds]
        report: dict[str, object] = {}
        if errors:
            report = {
                "n": len(errors),
                "mean_error": float(np.mean(errors)),
                "median_error": float(np.median(errors)),
                "max_error": float(np.max(errors)),
                "points": [
                    {
                        "expected": [sx, sy],
                        "predicted": [px, py],
                        "error": float(np.hypot(px - sx, py - sy)),
                    }
                    for sx, sy, px, py in self._test_preds
                ],
            }
        self.last_test_report = report
        self.wizard = WizardState(
            kind="test",
            index=len(self.targets),
            label="done",
            sx=0.5,
            sy=0.5,
            samples=0,
            needed=0,
            quality="GOOD",
            instruction=(
                f"Gaze test complete. mean={report.get('mean_error', 0):.3f} "
                f"med={report.get('median_error', 0):.3f} max={report.get('max_error', 0):.3f}"
                if report
                else "Gaze test complete (no samples)."
            ),
            done=True,
            report=report,
        )


def _confidence(
    features: GazeFeatures,
    screen: ScreenPoint,
    history: deque[tuple[float, float]],
) -> float:
    del screen
    track = {"GOOD": 1.0, "FAIR": 0.55, "POOR": 0.25, "LOST": 0.0}.get(features.tracking, 0.0)
    both = 1.0 if features.both_eyes else 0.4
    pose = max(0.0, 1.0 - abs(features.yaw_deg) / 45.0)
    if len(history) >= 4:
        arr = np.array(history)
        jitter = float(np.std(arr[:, 0]) + np.std(arr[:, 1]))
        stable = max(0.0, 1.0 - jitter * 8.0)
    else:
        stable = 0.5
    return float(clamp01(0.35 * track + 0.25 * both + 0.15 * pose + 0.25 * stable))


def _geometric_median(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute an order-independent geometric median with Weiszfeld iteration."""
    center = np.mean(points, axis=0)
    for _ in range(100):
        distances = np.linalg.norm(points - center, axis=1)
        inverse = 1.0 / np.maximum(distances, 1e-12)
        updated = np.sum(points * inverse[:, None], axis=0) / float(np.sum(inverse))
        if float(np.linalg.norm(updated - center)) <= 1e-12:
            return np.asarray(updated, dtype=np.float64)
        center = updated
    return np.asarray(center, dtype=np.float64)


def _predict_row(model: CalibrationModel, row: Sequence[float]) -> tuple[float, float]:
    vector = np.asarray(row, dtype=np.float64)
    coef_x = np.asarray(model.coef_x, dtype=np.float64)
    coef_y = np.asarray(model.coef_y, dtype=np.float64)
    if vector.shape != coef_x.shape or vector.shape != coef_y.shape:
        return 0.5, 0.5
    return clamp01(float(vector @ coef_x)), clamp01(float(vector @ coef_y))


def _coefficient_errors(
    coef_x: NDArray[np.float64],
    coef_y: NDArray[np.float64],
    rows: Sequence[NDArray[np.float64]],
    xy: Sequence[tuple[float, float]],
) -> NDArray[np.float64]:
    matrix = np.vstack(rows)
    predicted = np.clip(np.column_stack((matrix @ coef_x, matrix @ coef_y)), 0.0, 1.0)
    return np.asarray(
        np.linalg.norm(predicted - np.asarray(xy, dtype=np.float64), axis=1),
        dtype=np.float64,
    )


def _weighted_huber_coefficients_error(
    coef_x: NDArray[np.float64],
    coef_y: NDArray[np.float64],
    rows: Sequence[NDArray[np.float64]],
    xy: Sequence[tuple[float, float]],
    weights: Sequence[float],
) -> float:
    errors = _coefficient_errors(coef_x, coef_y, rows, xy)
    losses = np.where(
        errors <= HUBER_DELTA,
        0.5 * errors**2,
        HUBER_DELTA * (errors - 0.5 * HUBER_DELTA),
    )
    return float(np.average(losses, weights=np.asarray(weights, dtype=np.float64)))


def _weighted_huber_error(
    model: CalibrationModel,
    rows: Sequence[NDArray[np.float64]],
    xy: Sequence[tuple[float, float]],
    weights: Sequence[float],
) -> float:
    return _weighted_huber_coefficients_error(
        np.asarray(model.coef_x, dtype=np.float64),
        np.asarray(model.coef_y, dtype=np.float64),
        rows,
        xy,
        weights,
    )


def _mean_coefficient_error(
    coef_x: NDArray[np.float64],
    coef_y: NDArray[np.float64],
    rows: Sequence[NDArray[np.float64]],
    xy: Sequence[tuple[float, float]],
) -> float:
    return float(np.mean(_coefficient_errors(coef_x, coef_y, rows, xy)))
