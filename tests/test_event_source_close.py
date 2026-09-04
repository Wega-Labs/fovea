from __future__ import annotations

import threading
from collections.abc import Callable
from typing import ClassVar

import numpy as np
import pytest

from fovea.events import CalibrationDone, TrackingState, TrackingStatus
from fovea.webcam.calibration import CALIBRATION_LAYOUT, CalibrationTarget
from fovea.webcam.capture import CaptureClosed
from fovea.webcam.engine import GazeOutput, GazeSettings, WizardState
from fovea.webcam.event_source import WebcamEventSource
from tests.synth import synthetic_landmarks
from tests.test_capture import SAFETY_TIMEOUT_S, GatedCamera


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

    def __init__(self, *_args, **_kwargs) -> None:
        self.show_calls = 0
        self.close_calls = 0
        FakeCalibrationDisplay.instances.append(self)

    def show(self, _wizard: WizardState, *_args) -> None:
        self.show_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class FinishingCalibrationEngine:
    def __init__(self, *_args, **_kwargs) -> None:
        self.model = None
        self.wizard: WizardState | None = None
        self.targets: tuple[CalibrationTarget, ...] = CALIBRATION_LAYOUT
        self.calibration_warning = ""
        self.last_calibration_report: dict[str, object] = {}
        self.last_test_report: dict[str, object] = {}

    def start_calibration(self, targets: tuple[CalibrationTarget, ...] | None = None) -> None:
        if targets is not None:
            self.targets = targets
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
        self.last_calibration_report = {
            "n_points": 5,
            "coverage": 0.8,
            "loo_error": 0.07,
        }
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

    def resume_after_gaze_test(self) -> None:
        return None


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


def test_completed_calibration_emits_report(monkeypatch, tmp_path) -> None:
    FakeCamera.instances.clear()
    FakeEstimator.close_calls = 0
    monkeypatch.setattr("fovea.webcam.event_source.Webcam", FakeCamera)
    monkeypatch.setattr("fovea.webcam.event_source.FaceLandmarkEstimator", FakeEstimator)
    monkeypatch.setattr("fovea.webcam.event_source.GazeEngine", FinishingCalibrationEngine)

    source = WebcamEventSource(
        GazeSettings(calibration_path=str(tmp_path / "missing.json")),
        tmp_path,
        max_frames=1,
        show_calibration=False,
    )
    reports = [event for event in source.events() if isinstance(event, CalibrationDone)]

    assert len(reports) == 1
    assert reports[0].n_points == 5
    assert reports[0].coverage == 0.8
    assert reports[0].loo_error == 0.07


def _thread(target: Callable[[], object]) -> threading.Thread:
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def _capture_thread_alive() -> bool:
    return any(thread.name == "fovea-capture" for thread in threading.enumerate())


class FaceEstimator:
    def __init__(self, **_kwargs) -> None:
        return None

    def process(self, _frame):
        return type("Obs", (), {"landmarks": synthetic_landmarks(), "blendshapes": {}})()

    def close(self) -> None:
        return None


def test_events_leave_no_capture_session_or_thread_behind(monkeypatch, tmp_path) -> None:
    FakeCamera.instances.clear()
    monkeypatch.setattr("fovea.webcam.event_source.Webcam", FakeCamera)
    monkeypatch.setattr("fovea.webcam.event_source.FaceLandmarkEstimator", FakeEstimator)

    source = WebcamEventSource(
        GazeSettings(calibration_path=str(tmp_path / "c.json")),
        max_frames=2,
        show_calibration=False,
    )
    list(source.events())
    assert source._session is None
    assert not _capture_thread_alive()


def test_external_close_synthesizes_no_lost(monkeypatch, tmp_path) -> None:
    camera = GatedCamera([np.zeros((480, 640, 3), dtype=np.uint8), None])
    monkeypatch.setattr("fovea.webcam.event_source.Webcam", lambda *_args, **_kwargs: camera)
    monkeypatch.setattr("fovea.webcam.event_source.FaceLandmarkEstimator", FaceEstimator)

    source = WebcamEventSource(
        GazeSettings(calibration_path=str(tmp_path / "missing.json")),
        show_calibration=False,
    )
    events: list[object] = []
    first_frame_seen = threading.Event()

    def consume() -> None:
        for event in source.events():
            events.append(event)
            if isinstance(event, TrackingState):
                first_frame_seen.set()

    worker = _thread(consume)
    camera.release()
    assert first_frame_seen.wait(SAFETY_TIMEOUT_S)
    camera.wait_until_blocked(2)  # the producer is blocked in the second read
    session = source._session
    assert session is not None

    closer = _thread(source.close)
    with pytest.raises(CaptureClosed):
        session.next_frame()  # wakes once close() has stopped the hand-off
    camera.release()  # the blocked read now returns None, after the close
    closer.join(SAFETY_TIMEOUT_S)
    worker.join(SAFETY_TIMEOUT_S)
    assert not closer.is_alive()
    assert not worker.is_alive()

    tracking = [event for event in events if isinstance(event, TrackingState)]
    assert len(tracking) == 1
    assert tracking[0].status is not TrackingStatus.LOST
    assert camera.reads == 2
    assert camera.disconnect_calls == 1
    assert not camera.disconnected_during_read
    assert not _capture_thread_alive()


def test_concurrent_close_releases_the_camera_once_and_never_during_a_read(
    monkeypatch, tmp_path
) -> None:
    camera = GatedCamera()
    monkeypatch.setattr("fovea.webcam.event_source.Webcam", lambda *_args, **_kwargs: camera)
    monkeypatch.setattr("fovea.webcam.event_source.FaceLandmarkEstimator", FakeEstimator)

    source = WebcamEventSource(
        GazeSettings(calibration_path=str(tmp_path / "missing.json")),
        show_calibration=False,
    )
    worker = _thread(lambda: list(source.events()))
    camera.wait_until_blocked(1)
    session = source._session
    assert session is not None

    closers = [_thread(source.close) for _ in range(2)]
    with pytest.raises(CaptureClosed):
        session.next_frame()
    camera.release()
    for closer in closers:
        closer.join(SAFETY_TIMEOUT_S)
        assert not closer.is_alive()
    worker.join(SAFETY_TIMEOUT_S)
    assert not worker.is_alive()

    assert camera.disconnect_calls == 1
    assert not camera.disconnected_during_read
    assert source._session is None
    assert not _capture_thread_alive()
