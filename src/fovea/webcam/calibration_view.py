"""OpenCV display of the same calibration targets the engine samples."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence

import numpy as np

from fovea.webcam.calibration import CALIBRATION_LAYOUT, CalibrationTarget
from fovea.webcam.engine import WizardState

WINDOW_NAME = "Fovea calibration"


def render_calibration_frame(
    width: int,
    height: int,
    wizard: WizardState,
    layout: Sequence[CalibrationTarget] = CALIBRATION_LAYOUT,
) -> np.ndarray:
    """Draw every layout target; highlight the one ``wizard`` is currently sampling."""
    frame = np.full((height, width, 3), 18, dtype=np.uint8)
    active = CalibrationTarget(wizard.label, wizard.sx, wizard.sy)

    for target in layout:
        is_active = target.label == active.label and target.x == active.x and target.y == active.y
        _draw_target(frame, target, active=is_active)

    _draw_target(frame, active, active=True)
    _draw_hud(frame, wizard, len(layout))
    return frame


def _draw_target(frame: np.ndarray, target: CalibrationTarget, *, active: bool) -> None:
    import cv2

    height, width = frame.shape[:2]
    x, y = target.pixel_xy(width, height)
    if active:
        cv2.circle(frame, (x, y), max(18, min(width, height) // 40), (0, 210, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 6, (0, 210, 255), -1, cv2.LINE_AA)
        caption = f"LOOK HERE  {target.label}"
        scale = max(0.45, min(width, height) / 1400)
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        tx = min(max(8, x - tw // 2), width - tw - 8)
        ty = min(max(th + 8, y - 28), height - 8)
        cv2.putText(
            frame, caption, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 210, 255), 2, cv2.LINE_AA
        )
        return
    cv2.circle(frame, (x, y), 7, (70, 70, 70), 2, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 2, (90, 90, 90), -1, cv2.LINE_AA)


def _draw_hud(frame: np.ndarray, wizard: WizardState, total: int) -> None:
    import cv2

    height, width = frame.shape[:2]
    step = min(wizard.index + 1, total)
    lines = [
        f"Calibration {step}/{total}: look at the highlighted point ({wizard.label})",
        wizard.instruction,
        f"Samples {wizard.samples}/{wizard.needed}   Quality {wizard.quality}",
    ]
    scale = max(0.5, min(width, height) / 1600)
    y = int(28 * scale + 16)
    for line in lines:
        if not line:
            continue
        cv2.putText(
            frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (230, 230, 230), 2, cv2.LINE_AA
        )
        y += int(32 * scale)


class CalibrationDisplay:
    """OpenCV window showing engine calibration targets in screen space."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        *,
        fullscreen: bool = True,
    ) -> None:
        self.width = width
        self.height = height
        self.fullscreen = fullscreen
        self._open = False

    def show(self, wizard: WizardState) -> None:
        import cv2

        frame = render_calibration_frame(self.width, self.height, wizard)
        if not self._open:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            if self.fullscreen:
                cv2.setWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN,
                )
            else:
                cv2.resizeWindow(WINDOW_NAME, self.width, self.height)
            self._open = True
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(1)

    def close(self) -> None:
        if not self._open:
            return
        import cv2

        with contextlib.suppress(cv2.error):
            cv2.destroyWindow(WINDOW_NAME)
        self._open = False
