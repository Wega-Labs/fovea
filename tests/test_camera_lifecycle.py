from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from fovea.events import CameraLost, CameraReady, TrackingState, TrackingStatus
from fovea.webcam.camera import CameraActuals, CameraError, ReconnectPolicy
from fovea.webcam.capture import CaptureClosed, CapturedFrame
from fovea.webcam.engine import GazeSettings
from fovea.webcam.event_source import WebcamEventSource


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
