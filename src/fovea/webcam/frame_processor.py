"""Camera-independent translation from landmark frames to Fovea events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from fovea.events import (
    Blink,
    CalibrationCue,
    Eye,
    FoveaEvent,
    GazePoint,
    TrackingState,
    TrackingStatus,
)
from fovea.webcam.calibration import CALIBRATION_LAYOUT
from fovea.webcam.engine import GazeEngine


def _tracking_status(label: str) -> TrackingStatus:
    if label == "GOOD":
        return TrackingStatus.ACTIVE
    if label in {"FAIR", "POOR"}:
        return TrackingStatus.UNCERTAIN
    return TrackingStatus.LOST


class GazeFrameProcessor:
    """Process landmark-only frames using the same path for webcams and replay."""

    def __init__(self, engine: GazeEngine) -> None:
        self.engine = engine
        self._last_blink = False

    def events_for_frame(
        self,
        landmarks: Sequence[Any] | None,
        image_w: float,
        image_h: float,
        dt: float,
        fps: float,
        timestamp_ns: int,
        blendshapes: Mapping[str, float] | None = None,
    ) -> tuple[FoveaEvent, ...]:
        output = self.engine.process(
            landmarks,
            image_w,
            image_h,
            dt,
            fps,
            blendshapes=blendshapes,
        )
        events: list[FoveaEvent] = []

        wizard = self.engine.wizard
        if wizard is not None and not wizard.done:
            events.append(
                CalibrationCue(
                    label=wizard.label,
                    x=wizard.sx,
                    y=wizard.sy,
                    index=wizard.index,
                    total=len(CALIBRATION_LAYOUT),
                    samples=wizard.samples,
                    needed=wizard.needed,
                    instruction=wizard.instruction,
                    timestamp_ns=timestamp_ns,
                )
            )

        detail = output.message
        if output.features is not None:
            detail = output.features.message
        events.append(
            TrackingState(
                status=_tracking_status(output.tracking),
                confidence=output.confidence,
                timestamp_ns=timestamp_ns,
                detail=detail,
            )
        )

        blinking = bool(output.features and output.features.blink)
        if blinking and not self._last_blink:
            events.append(
                Blink(
                    eye=Eye.BOTH,
                    duration_ms=0.0,
                    confidence=output.confidence,
                    timestamp_ns=timestamp_ns,
                )
            )
        self._last_blink = blinking

        if output.valid and output.screen is not None:
            events.append(
                GazePoint(
                    x=output.screen.x,
                    y=output.screen.y,
                    confidence=output.confidence,
                    timestamp_ns=timestamp_ns,
                )
            )
        return tuple(events)
