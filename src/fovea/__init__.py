"""Fovea's public event API."""

from fovea.events import (
    Blink,
    CalibrationCue,
    Eye,
    Fixation,
    FoveaEvent,
    GazePoint,
    Gesture,
    GesturePhase,
    Manipulation,
    TrackingState,
    TrackingStatus,
)
from fovea.interfaces import EventSink, EventSource
from fovea.webcam import GazeEngine, GazeSettings, WebcamEventSource

__all__ = [
    "Blink",
    "CalibrationCue",
    "EventSink",
    "EventSource",
    "Eye",
    "Fixation",
    "FoveaEvent",
    "GazeEngine",
    "GazePoint",
    "GazeSettings",
    "Gesture",
    "GesturePhase",
    "Manipulation",
    "TrackingState",
    "TrackingStatus",
    "WebcamEventSource",
]

__version__ = "0.1.0"
