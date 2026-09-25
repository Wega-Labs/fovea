"""Fovea's public event API."""

import logging

from fovea.events import (
    Blink,
    CalibrationCue,
    CalibrationDone,
    CalibrationUpdated,
    CalibrationWarning,
    Diagnostics,
    DoubleBlink,
    Dwell,
    DwellProgress,
    Eye,
    Fixation,
    FoveaEvent,
    GazePoint,
    GazeTestDone,
    GazeTestPoint,
    Gesture,
    GesturePhase,
    LongBlink,
    Manipulation,
    Saccade,
    TargetEnter,
    TargetLeave,
    TrackingState,
    TrackingStatus,
    Wink,
)
from fovea.interfaces import EventSink, EventSource
from fovea.runtime import AsyncEventSource, ClosableEventSource, EventPump
from fovea.webcam import (
    GazeEngine,
    GazeSettings,
    LandmarkBackend,
    LandmarkObservation,
    TargetRect,
    TargetTracker,
    WebcamEventSource,
)

# A library logs but never configures output; hosts opt in via logging.getLogger("fovea").
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AsyncEventSource",
    "Blink",
    "CalibrationCue",
    "CalibrationDone",
    "CalibrationUpdated",
    "CalibrationWarning",
    "ClosableEventSource",
    "Diagnostics",
    "DoubleBlink",
    "Dwell",
    "DwellProgress",
    "EventPump",
    "EventSink",
    "EventSource",
    "Eye",
    "Fixation",
    "FoveaEvent",
    "GazeEngine",
    "GazePoint",
    "GazeSettings",
    "GazeTestDone",
    "GazeTestPoint",
    "Gesture",
    "GesturePhase",
    "LandmarkBackend",
    "LandmarkObservation",
    "LongBlink",
    "Manipulation",
    "Saccade",
    "TargetEnter",
    "TargetLeave",
    "TargetRect",
    "TargetTracker",
    "TrackingState",
    "TrackingStatus",
    "WebcamEventSource",
    "Wink",
]

__version__ = "0.1.0"
