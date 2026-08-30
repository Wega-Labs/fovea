from __future__ import annotations

from typing import ClassVar

import numpy as np

from fovea.webcam.engine import GazeSettings
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
