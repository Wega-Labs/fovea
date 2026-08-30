"""Webcam gaze pipeline: camera, landmarks, calibration, and event source."""

from fovea.webcam.camera import CameraError, Webcam
from fovea.webcam.engine import GazeEngine, GazeOutput, GazeSettings
from fovea.webcam.event_source import WebcamEventSource
from fovea.webcam.landmarks import FaceLandmarkEstimator, MediaPipeUnavailableError

__all__ = [
    "CameraError",
    "FaceLandmarkEstimator",
    "GazeEngine",
    "GazeOutput",
    "GazeSettings",
    "MediaPipeUnavailableError",
    "Webcam",
    "WebcamEventSource",
]
