"""Collect frames for one calibration / test point; reject bad samples."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fovea.webcam.calibration import quality_label, robust_median_rows


class PointCollector:
    def __init__(self, needed: int, min_good: int) -> None:
        self.needed = needed
        self.min_good = min_good
        self.rows: list[NDArray[np.float64]] = []
        self.rejected = 0

    def add(self, vector: NDArray[np.float64], tracking: str, blink: bool) -> None:
        if blink or tracking == "LOST":
            self.rejected += 1
            return
        if tracking == "POOR":
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

    @property
    def count(self) -> int:
        return len(self.rows)

    def done(self) -> bool:
        return self.count >= self.needed

    def quality(self) -> str:
        return quality_label(self.count, self.min_good)

    def median(self) -> NDArray[np.float64]:
        if not self.rows:
            msg = "No samples"
            raise ValueError(msg)
        return robust_median_rows(self.rows)
