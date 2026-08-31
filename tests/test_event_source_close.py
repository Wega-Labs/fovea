from __future__ import annotations

from typing import ClassVar

import numpy as np

from fovea.webcam.engine import GazeOutput, GazeSettings, WizardState
from fovea.webcam.event_source import WebcamEventSource


class FakeCamera:
    instances: ClassVar[list[FakeCamera]] = []

    def __init__(self, *_args, **_kwargs) -> None:
        self.connected = False
        self.disconnect_calls = 0
        FakeCamera.instances.append(self)

    def connect(self) -> None:
        self.connected = True

    def read(self):
        return np.zeros((48, 64, 3), dtype=np.uint8)

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False

    @property
    def is_connected(self) -> bool:
        return self.connected


class FakeEstimator:
    close_calls = 0

    def __init__(self, **_kwargs) -> None:
        return None

    def process(self, _frame):
        return None

    def close(self) -> None:
        FakeEstimator.close_calls += 1


class FakeCalibrationDisplay:
    instances: ClassVar[list[FakeCalibrationDisplay]] = []

    def __init__(self) -> None:
        self.show_calls = 0
        self.close_calls = 0
        FakeCalibrationDisplay.instances.append(self)

    def show(self, _wizard: WizardState) -> None:
        self.show_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class FinishingCalibrationEngine:
    def __init__(self, *_args, **_kwargs) -> None:
        self.model = None
        self.wizard: WizardState | None = None

    def start_calibration(self) -> None:
        self.wizard = WizardState(
            kind="calibrate",
            index=0,
            label="center",
            sx=0.5,
            sy=0.5,
            samples=0,
            needed=1,
            quality="POOR",
            instruction="Look at the point.",
        )

    def process(self, *_args, **_kwargs) -> GazeOutput:
        self.wizard = None
        return GazeOutput(
            valid=False,
            tracking="LOST",
            message="Calibration complete",
            features=None,
            screen=None,
            confidence=0.0,
            fps=0.0,
            calibrated=True,
            frozen=False,
        )


def test_close_releases_webcam(monkeypatch, tmp_path) -> None:
    FakeCamera.instances.clear()
    FakeEstimator.close_calls = 0
    monkeypatch.setattr("fovea.webcam.event_source.Webcam", FakeCamera)
    monkeypatch.setattr("fovea.webcam.event_source.FaceLandmarkEstimator", FakeEstimator)

    source = WebcamEventSource(
        GazeSettings(calibration_path=str(tmp_path / "c.json")),
        tmp_path,
        max_frames=2,
        show_calibration=False,
    )
    list(source.events())
    assert FakeCamera.instances
    camera = FakeCamera.instances[-1]
    assert camera.disconnect_calls == 1
    assert camera.connected is False
    assert FakeEstimator.close_calls == 1
    assert source._camera is None
    assert source._estimator is None


def test_close_is_idempotent(monkeypatch, tmp_path) -> None:
    FakeCamera.instances.clear()
    FakeEstimator.close_calls = 0
    monkeypatch.setattr("fovea.webcam.event_source.Webcam", FakeCamera)
    monkeypatch.setattr("fovea.webcam.event_source.FaceLandmarkEstimator", FakeEstimator)

    source = WebcamEventSource(
        GazeSettings(calibration_path=str(tmp_path / "c.json")),
        tmp_path,
        max_frames=1,
        show_calibration=False,
    )
    iterator = source.events()
    next(iterator)
    source.close()
    source.close()
    camera = FakeCamera.instances[-1]
    assert camera.disconnect_calls == 1
    list(iterator)


def test_close_without_starting_is_safe(tmp_path) -> None:
    source = WebcamEventSource(
        GazeSettings(calibration_path=str(tmp_path / "c.json")),
        tmp_path,
        show_calibration=False,
    )
    source.close()
    source.close()


def test_calibration_display_closes_when_calibration_finishes(monkeypatch, tmp_path) -> None:
    FakeCamera.instances.clear()
    FakeEstimator.close_calls = 0
    FakeCalibrationDisplay.instances.clear()
    monkeypatch.setattr("fovea.webcam.event_source.Webcam", FakeCamera)
    monkeypatch.setattr("fovea.webcam.event_source.FaceLandmarkEstimator", FakeEstimator)
    monkeypatch.setattr("fovea.webcam.event_source.GazeEngine", FinishingCalibrationEngine)
    monkeypatch.setattr("fovea.webcam.event_source.CalibrationDisplay", FakeCalibrationDisplay)

    source = WebcamEventSource(
        GazeSettings(calibration_path=str(tmp_path / "missing.json")),
        tmp_path,
        max_frames=1,
        show_calibration=True,
    )
    list(source.events())

    display = FakeCalibrationDisplay.instances[-1]
    assert display.show_calls == 1
    assert display.close_calls == 1
