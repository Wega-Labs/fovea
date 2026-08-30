"""Gaze pipeline: features → calibration map → smooth → Fovea events."""

from __future__ import annotations

import time as _time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fovea.util import ScreenPoint, clamp01
from fovea.webcam.calibration import (
    CALIBRATION_LAYOUT,
    fit_ridge,
    load_model,
    save_model,
    uncalibrated_map,
)
from fovea.webcam.features import GazeFeatures, extract_features
from fovea.webcam.sampler import PointCollector
from fovea.webcam.smoothing import OneEuroPoint, ema


@dataclass
class GazeSettings:
    dwell_ms: float = 500.0
    stability_ms: float = 300.0
    hysteresis: float = 0.04
    smoothing_alpha: float = 0.35
    one_euro_mincutoff: float = 1.2
    one_euro_beta: float = 0.02
    blink_ear: float = 0.16
    min_face_width: float = 0.18
    max_yaw_deg: float = 35.0
    samples_per_point: int = 28
    min_good_samples: int = 12
    settle_frames: int = 12
    calibration_path: str = "data/gaze_calibration.json"
    debug: bool = True


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


class GazeEngine:
    def __init__(self, settings: GazeSettings, project_root: Path) -> None:
        self.settings = settings
        self.path = Path(settings.calibration_path)
        if not self.path.is_absolute():
            self.path = project_root / self.path
        self.model = load_model(self.path)
        self._filter = OneEuroPoint(settings.one_euro_mincutoff, settings.one_euro_beta)
        self._last_screen: ScreenPoint | None = None
        self._history: deque[tuple[float, float]] = deque(maxlen=12)
        self._frozen: ScreenPoint | None = None
        self.wizard: WizardState | None = None
        self._collectors: list[PointCollector] = []
        self._test_preds: list[tuple[float, float, float, float]] = []
        self.last_test_report: dict[str, object] = {}

    def start_calibration(self) -> None:
        self.model = None
        self._filter.reset()
        self._last_screen = None
        self._collectors = [
            PointCollector(self.settings.samples_per_point, self.settings.min_good_samples)
            for _ in CALIBRATION_LAYOUT
        ]
        self._set_wizard("calibrate", 0)

    def start_gaze_test(self) -> None:
        self._test_preds = []
        self._collectors = [
            PointCollector(self.settings.samples_per_point, self.settings.min_good_samples)
            for _ in CALIBRATION_LAYOUT
        ]
        self._set_wizard("test", 0)

    def _set_wizard(self, kind: str, index: int) -> None:
        if index >= len(CALIBRATION_LAYOUT):
            self.wizard = None
            return
        target = CALIBRATION_LAYOUT[index]
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
    ) -> GazeOutput:
        t0 = _time.perf_counter()
        if landmarks is None:
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
                f"Calibration point {self.wizard.index + 1}/{len(CALIBRATION_LAYOUT)}  "
                f"Samples: {collector.count}  Quality: {collector.quality()}"
            )
        else:
            self.wizard.instruction = (
                f"Test point {self.wizard.index + 1}/{len(CALIBRATION_LAYOUT)}  "
                f"Samples: {collector.count}"
            )
        if not collector.done():
            return
        if self.wizard.kind == "test" and self.model is not None:
            pred = self.model.predict(features)
            self._test_preds.append((self.wizard.sx, self.wizard.sy, pred[0], pred[1]))
        nxt = self.wizard.index + 1
        if nxt >= len(CALIBRATION_LAYOUT):
            if self.wizard.kind == "calibrate":
                self._finish_calibration()
            else:
                self._finish_test()
            return
        self._set_wizard(self.wizard.kind, nxt)

    def _finish_calibration(self) -> None:
        rows: list[np.ndarray] = []
        xy: list[tuple[float, float]] = []
        counts: dict[str, int] = {}
        qualities: dict[str, str] = {}
        for target, collector in zip(CALIBRATION_LAYOUT, self._collectors, strict=True):
            if collector.count == 0:
                continue
            rows.append(collector.median())
            xy.append((target.x, target.y))
            key = f"{target.label}_{len(counts)}"
            counts[key] = collector.count
            qualities[key] = collector.quality()
        if len(rows) >= 3:
            self.model = fit_ridge(rows, xy, counts, qualities)
            save_model(self.model, self.path)
        self.wizard = None
        self._filter.reset()
        self._last_screen = None

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
            index=len(CALIBRATION_LAYOUT),
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
