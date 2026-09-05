from __future__ import annotations

import threading
from collections import deque
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pytest

from fovea.events import CameraLost, CameraReady, FoveaEvent, TrackingState, TrackingStatus
from fovea.webcam.camera import CameraActuals, CameraError, CameraInfo, ReconnectPolicy, Webcam
from fovea.webcam.capture import CaptureClosed, CapturedFrame
from fovea.webcam.engine import GazeSettings
from fovea.webcam.event_source import WebcamEventSource
from tests.test_camera import FakeCapture
from tests.test_capture import SAFETY_TIMEOUT_S


def _frame(timestamp_ns: int = 1) -> CapturedFrame:
    return CapturedFrame(
        np.zeros((48, 64, 3), dtype=np.uint8),
        captured_ns=timestamp_ns,
        timestamp_ns=timestamp_ns,
        sequence=timestamp_ns,
    )


class ScriptedCamera:
    outcomes: deque[CameraActuals | CameraError]

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.connect_calls = 0
        self.disconnect_calls = 0

    def connect(self) -> CameraActuals:
        self.connect_calls += 1
        outcome = self.outcomes.popleft()
        if isinstance(outcome, CameraError):
            raise outcome
        return outcome

    def disconnect(self) -> None:
        self.disconnect_calls += 1

    def read(self) -> None:
        return None


class ScriptedSession:
    scripts: deque[list[CapturedFrame | Exception | None]]
    instances: ClassVar[list[ScriptedSession]] = []

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.script = deque(self.scripts.popleft())
        self.started = False
        self.stopped = False
        self.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def next_frame(self) -> CapturedFrame | None:
        if not self.script:
            raise CaptureClosed
        result = self.script.popleft()
        if isinstance(result, Exception):
            raise result
        return result

    @property
    def dropped_frames(self) -> int:
        return 0


class FakeEstimator:
    def __init__(self, **_kwargs: object) -> None:
        return None

    def process(self, _frame: object) -> None:
        return None

    def close(self) -> None:
        return None


def _source(tmp_path: Path, **kwargs: object) -> WebcamEventSource:
    return WebcamEventSource(
        GazeSettings(calibration_path=str(tmp_path / "missing.json")),
        show_calibration=False,
        reconnect_policy=ReconnectPolicy(lost_after_s=0.0, initial_delay_s=0.0),
        **kwargs,
    )


def _install(monkeypatch, outcomes, scripts) -> ScriptedCamera:
    ScriptedCamera.outcomes = deque(outcomes)
    ScriptedSession.scripts = deque(scripts)
    ScriptedSession.instances = []
    camera = ScriptedCamera()
    monkeypatch.setattr("fovea.webcam.event_source.Webcam", lambda *_args, **_kwargs: camera)
    monkeypatch.setattr("fovea.webcam.event_source.CaptureSession", ScriptedSession)
    monkeypatch.setattr("fovea.webcam.event_source.FaceLandmarkEstimator", FakeEstimator)
    return camera


def test_failed_read_ends_stream_without_reconnect(monkeypatch, tmp_path) -> None:
    actuals = CameraActuals(0, "Camera", "stable", 640, 480, 30.0)
    _install(monkeypatch, [actuals], [[None]])
    iterator = _source(tmp_path).events()
    assert isinstance(next(iterator), CameraReady)
    lost = next(iterator)
    assert isinstance(lost, CameraLost)
    assert lost.reason == "read_failed"
    assert not lost.reconnecting
    with pytest.raises(CameraError):
        next(iterator)


def test_reconnect_rebinds_session_after_failed_open(monkeypatch, tmp_path) -> None:
    actuals = CameraActuals(0, "Camera", "stable", 640, 480, 30.0)
    camera = _install(
        monkeypatch,
        [actuals, CameraError("not back yet"), actuals],
        [[_frame(), None], [_frame(2)]],
    )
    events = list(_source(tmp_path, reconnect=True, max_frames=3).events())
    lifecycle = [event for event in events if isinstance(event, CameraReady | CameraLost)]
    assert [type(event) for event in lifecycle] == [CameraReady, CameraLost, CameraReady]
    assert lifecycle[1].reconnecting
    assert any(
        isinstance(event, TrackingState) for event in events[events.index(lifecycle[2]) + 1 :]
    )
    assert camera.connect_calls == 3
    assert camera.disconnect_calls == 2
    assert all(session.stopped for session in ScriptedSession.instances)


def test_read_error_emits_lost_tracking_before_camera_lost(monkeypatch, tmp_path) -> None:
    actuals = CameraActuals(0, "Camera", "stable", 640, 480, 30.0)
    _install(
        monkeypatch,
        [actuals, actuals],
        [[_frame(), CameraError("read exploded")], [_frame(2)]],
    )
    events = list(_source(tmp_path, reconnect=True, max_frames=3).events())
    lost_index = next(index for index, event in enumerate(events) if isinstance(event, CameraLost))
    prior = events[lost_index - 1]
    assert isinstance(prior, TrackingState)
    assert prior.status is TrackingStatus.LOST
    assert prior.timestamp_ns == events[lost_index].timestamp_ns
    assert sum(isinstance(event, CameraLost) for event in events) == 1
    assert sum(isinstance(event, CameraReady) for event in events) == 2


def test_reconnect_rejects_a_different_stable_id(monkeypatch, tmp_path) -> None:
    expected = CameraActuals(0, "Camera A", "a", 640, 480, 30.0)
    different = CameraActuals(0, "Camera B", "b", 640, 480, 30.0)
    camera = _install(
        monkeypatch,
        [expected, different, expected],
        [[None], [_frame()]],
    )
    events = list(_source(tmp_path, reconnect=True, max_frames=2).events())
    assert sum(isinstance(event, CameraReady) for event in events) == 2
    assert camera.connect_calls == 3
    assert camera.disconnect_calls == 3


class BlockingCamera(ScriptedCamera):
    """Scripted camera whose ``connect()`` blocks until the test lets it finish."""

    def __init__(self) -> None:
        super().__init__()
        self.connect_entered = threading.Event()
        self.connect_may_finish = threading.Event()

    def connect(self) -> CameraActuals:
        self.connect_entered.set()
        assert self.connect_may_finish.wait(SAFETY_TIMEOUT_S)
        return super().connect()


class BlockingEstimator(FakeEstimator):
    """Landmark estimator whose construction (model load) blocks until released."""

    load_entered = threading.Event()
    load_may_finish = threading.Event()

    def __init__(self, **_kwargs: object) -> None:
        BlockingEstimator.load_entered.set()
        assert BlockingEstimator.load_may_finish.wait(SAFETY_TIMEOUT_S)


def _consume_in_thread(source: WebcamEventSource) -> tuple[threading.Thread, list[FoveaEvent]]:
    events: list[FoveaEvent] = []

    def consume() -> None:
        events.extend(source.events())

    worker = threading.Thread(target=consume, daemon=True)
    worker.start()
    return worker, events


def _capture_thread_alive() -> bool:
    return any(thread.name == "fovea-capture" for thread in threading.enumerate())


def test_close_during_initial_open_releases_camera_and_emits_nothing(monkeypatch, tmp_path) -> None:
    ScriptedCamera.outcomes = deque([CameraActuals(0, "Camera", "stable", 640, 480, 30.0)])
    ScriptedSession.scripts = deque([[_frame()]])
    ScriptedSession.instances = []
    camera = BlockingCamera()
    monkeypatch.setattr("fovea.webcam.event_source.Webcam", lambda *_args, **_kwargs: camera)
    monkeypatch.setattr("fovea.webcam.event_source.CaptureSession", ScriptedSession)
    monkeypatch.setattr("fovea.webcam.event_source.FaceLandmarkEstimator", FakeEstimator)
    source = _source(tmp_path)

    worker, events = _consume_in_thread(source)
    assert camera.connect_entered.wait(SAFETY_TIMEOUT_S)
    source.close()  # shutdown requested while the open is still in flight
    camera.connect_may_finish.set()
    worker.join(SAFETY_TIMEOUT_S)
    assert not worker.is_alive()

    assert events == []
    assert camera.connect_calls == 1
    assert camera.disconnect_calls == 1
    assert ScriptedSession.instances == []
    assert source._session is None
    assert source._camera is None
    assert not source._camera_connected


def test_close_during_model_load_releases_camera_and_starts_no_capture(
    monkeypatch, tmp_path
) -> None:
    BlockingEstimator.load_entered = threading.Event()
    BlockingEstimator.load_may_finish = threading.Event()
    camera = _install(
        monkeypatch,
        [CameraActuals(0, "Camera", "stable", 640, 480, 30.0)],
        [[_frame()]],
    )
    monkeypatch.setattr("fovea.webcam.event_source.FaceLandmarkEstimator", BlockingEstimator)
    source = _source(tmp_path)

    worker, events = _consume_in_thread(source)
    assert BlockingEstimator.load_entered.wait(SAFETY_TIMEOUT_S)
    source.close()  # the camera is owned by now, but capture has not started
    BlockingEstimator.load_may_finish.set()
    worker.join(SAFETY_TIMEOUT_S)
    assert not worker.is_alive()

    assert events == []
    assert camera.disconnect_calls == 1
    assert ScriptedSession.instances == []
    assert source._session is None
    assert not _capture_thread_alive()


def test_reconnect_follows_numeric_selection_to_a_reassigned_index(monkeypatch, tmp_path) -> None:
    """``--camera 1`` must recover the same device after the platform moves it to index 3."""
    cameras: list[CameraInfo] = [CameraInfo(1, "Desk Camera", "desk", False)]
    opened: list[int] = []

    def open_capture(index: int, _api: int) -> FakeCapture:
        opened.append(index)
        return FakeCapture()

    monkeypatch.setattr(cv2, "VideoCapture", open_capture)
    monkeypatch.setattr(
        "fovea.webcam.event_source.Webcam",
        lambda selector, width, height, mirror, **kwargs: Webcam(
            selector, width, height, mirror, enumerator=lambda: tuple(cameras), **kwargs
        ),
    )
    ScriptedSession.scripts = deque([[_frame(), CameraError("read exploded")], [_frame(2)]])
    ScriptedSession.instances = []
    monkeypatch.setattr("fovea.webcam.event_source.CaptureSession", ScriptedSession)
    monkeypatch.setattr("fovea.webcam.event_source.FaceLandmarkEstimator", FakeEstimator)

    iterator = _source(tmp_path, device_index=1, reconnect=True, max_frames=3).events()
    first = next(iterator)
    assert isinstance(first, CameraReady)
    assert (first.index, first.unique_id) == (1, "desk")
    # While unplugged, another device takes index 1 and ours comes back at index 3.
    cameras[:] = [
        CameraInfo(1, "Other Camera", "other", False),
        CameraInfo(3, "Desk Camera", "desk", False),
    ]
    events = list(iterator)

    ready = [event for event in events if isinstance(event, CameraReady)]
    assert [(event.index, event.unique_id) for event in ready] == [(3, "desk")]
    assert opened == [1, 3]
    assert sum(isinstance(event, CameraLost) for event in events) == 1
