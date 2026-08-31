"""Collect frames for one calibration / test point; reject bad samples."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fovea.webcam.calibration import quality_label, robust_median_rows


class PointCollector:
    """Collect calibration rows while reporting poor tracking at half weight.

    Only lost or blinking frames are rejected. ``POOR`` rows still advance the
    wizard so calibration cannot starve, but each contributes 0.5 to the quality
    score while ``FAIR`` and ``GOOD`` rows contribute 1.0.
    """

    def __init__(self, needed: int, min_good: int) -> None:
        self.needed = needed
        self.min_good = min_good
        self.rows: list[NDArray[np.float64]] = []
        self.weights: list[float] = []
        self.rejected = 0

    def add(self, vector: NDArray[np.float64], tracking: str, blink: bool) -> None:
        if blink or tracking == "LOST":
            self.rejected += 1
            return
        if len(self.rows) >= 5:
            med = np.median(np.vstack(self.rows), axis=0)
            dist = float(np.linalg.norm(vector - med))
            spread = float(np.median(np.linalg.norm(np.vstack(self.rows) - med, axis=1)))
            if spread > 1e-6 and dist > 2.8 * spread + 0.35:
                self.rejected += 1
                return
        self.rows.append(vector)
        self.weights.append(0.5 if tracking == "POOR" else 1.0)

    @property
    def count(self) -> int:
        return len(self.rows)

    @property
    def weighted_count(self) -> float:
        return sum(self.weights)

    def done(self) -> bool:
        return self.count >= self.needed

    def quality(self) -> str:
        return quality_label(self.weighted_count, self.min_good)

    def median(self) -> NDArray[np.float64]:
        if not self.rows:
            msg = "No samples"
            raise ValueError(msg)
        return robust_median_rows(self.rows)
