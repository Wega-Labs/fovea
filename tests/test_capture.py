"""Camera-free tests for the capture thread, latest-frame hand-off, and fps gate.

Every wait here is event-driven: the ``GatedCamera`` double blocks in ``read()``
until the test grants a permit, and the test observes progress through counters
guarded by a condition. Timeouts are safety nets on waits, never the assertion.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Callable, Iterable

import numpy as np
import pytest

from fovea.events import TrackingState
from fovea.webcam.camera import CameraError
from fovea.webcam.capture import (
    CaptureClosed,
    CapturedFrame,
    CaptureSession,
    FrameRateGate,
    LatestFrameSlot,
)

SAFETY_TIMEOUT_S = 5.0
_MS = 1_000_000


class GatedCamera:
    """Camera double whose ``read()`` blocks until the test releases a permit.

    ``responses`` are consumed one per read: an array is returned, ``None`` is a
    failed read, and an exception instance is raised. Reads beyond the scripted
    responses return a zero frame. ``unblock()`` makes every later read return
    ``None`` immediately, which lets a session stop without a scripted release.
    """

    def __init__(self, responses: Iterable[object] = (), shape=(48, 64, 3)) -> None:
        self._permits = threading.Semaphore(0)
        self._changed = threading.Condition()
        self._responses: deque[object] = deque(responses)
        self._shape = shape
        self._unblocked = False
        self.reads = 0
        self.blocked = 0
        self.in_read = False
        self.connected = False
        self.disconnect_calls = 0
        self.disconnected_during_read = False

    def connect(self) -> None:
        self.connected = True

    def release(self, count: int = 1) -> None:
        for _ in range(count):
            self._permits.release()

    def unblock(self) -> None:
        self._unblocked = True
        self._permits.release()

    def wait_until_blocked(self, count: int) -> None:
        """Wait until ``count`` reads have started blocking (so ``count - 1`` completed)."""
        with self._changed:
            assert self._changed.wait_for(lambda: self.blocked >= count, SAFETY_TIMEOUT_S)

    def read(self):
        self.in_read = True
        try:
            if self._unblocked:
                return None
            with self._changed:
                self.blocked += 1
                self._changed.notify_all()
            self._permits.acquire()
            if self._unblocked:
                return None
            response = (
                self._responses.popleft()
                if self._responses
                else np.zeros(self._shape, dtype=np.uint8)
            )
            if isinstance(response, Exception):
                raise response
            return response
        finally:
            self.in_read = False
            with self._changed:
                self.reads += 1
                self._changed.notify_all()

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        if self.in_read:
            self.disconnected_during_read = True
        self.connected = False

    @property
    def is_connected(self) -> bool:
        return self.connected


class InstantFailureCamera:
    """Camera double whose ``read()`` returns ``None`` immediately, forever."""

    def __init__(self) -> None:
        self.reads = 0
        self.first_read = threading.Event()

    def read(self):
        self.reads += 1
        self.first_read.set()
        return None


def _frame(sequence: int) -> CapturedFrame:
    pixels = np.zeros((1, 1, 3), dtype=np.uint8)
    return CapturedFrame(pixels, sequence * _MS, sequence * _MS, sequence)


def _thread(target: Callable[[], object]) -> threading.Thread:
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def _capture_thread() -> threading.Thread | None:
    return next((t for t in threading.enumerate() if t.name == "fovea-capture"), None)


def _stop_while_blocked(session: CaptureSession, camera: GatedCamera) -> int:
    """Stop a session whose producer is blocked in ``read()``.

    Returns the number of frames the consumer drained before the slot closed.
    ``stop()`` sets its flag and closes the slot before joining, so once the
    consumer sees ``CaptureClosed`` the in-flight read can safely be released.
    """
    stopper = _thread(session.stop)
    drained = 0
    while True:
        try:
            if session.next_frame() is not None:
                drained += 1
        except CaptureClosed:
            break
    camera.release()
    stopper.join(SAFETY_TIMEOUT_S)
    assert not stopper.is_alive()
    assert not session.is_running
    return drained


# ---------------------------------------------------------------- slot rules


def test_slot_keeps_only_the_newest_frame_and_counts_drops() -> None:
    slot = LatestFrameSlot()
    first, second, third = _frame(1), _frame(2), _frame(3)
    slot.put_frame(first)
    slot.put_frame(second)
    slot.put_frame(third)
    assert slot.take() is third
    assert slot.dropped == 2


def test_slot_read_failure_marker_yields_none() -> None:
    slot = LatestFrameSlot()
    slot.put_read_failure()
    assert slot.take() is None


def test_slot_read_failure_never_displaces_a_pending_frame() -> None:
    slot = LatestFrameSlot()
    frame = _frame(1)
    slot.put_frame(frame)
    slot.put_read_failure()
    assert slot.take() is frame
    assert slot.dropped == 0


def test_slot_frame_replaces_read_failure_without_counting_a_drop() -> None:
    slot = LatestFrameSlot()
    frame = _frame(1)
    slot.put_read_failure()
    slot.put_frame(frame)
    assert slot.take() is frame
    assert slot.dropped == 0


def test_slot_close_discards_pending_frame_and_ignores_later_puts() -> None:
    slot = LatestFrameSlot()
    slot.put_frame(_frame(1))
    slot.close()
    with pytest.raises(CaptureClosed):
        slot.take()
    slot.put_frame(_frame(2))
    assert slot.dropped == 0
    with pytest.raises(CaptureClosed):
        slot.take()


def test_slot_delivers_pending_frame_before_raising_producer_error() -> None:
    slot = LatestFrameSlot()
    frame = _frame(1)
    error = CameraError("unplugged")
    slot.put_frame(frame)
    slot.fail(error)
    assert slot.take() is frame
    with pytest.raises(CameraError) as first:
        slot.take()
    assert first.value is error
    with pytest.raises(CameraError):
        slot.take()


def test_slot_error_on_empty_slot_raises_immediately() -> None:
    slot = LatestFrameSlot()
    slot.fail(CameraError("unplugged"))
    with pytest.raises(CameraError):
        slot.take()


# ---------------------------------------------------------------- fps gate


def _series(fps: float, seconds: float, jitter_ms: float = 0.0) -> list[int]:
    interval_ns = 1e9 / fps
    times: list[int] = []
    for index in range(round(fps * seconds)):
        offset = 0.0
        if index and jitter_ms:
            offset = jitter_ms * _MS if index % 2 else -jitter_ms * _MS
        times.append(int(index * interval_ns + offset))
    return times


def _admitted(gate: FrameRateGate, series: list[int]) -> list[int]:
    return [index for index, captured_ns in enumerate(series) if gate.admits(captured_ns)]


def test_gate_admits_every_frame_at_exactly_the_cap() -> None:
    assert len(_admitted(FrameRateGate(30.0), _series(30.0, 1.0))) == 30


@pytest.mark.parametrize(
    ("max_fps", "expected"),
    [(30.0, range(28, 31)), (15.0, [15]), (10.0, [10]), (20.0, range(18, 22))],
)
def test_gate_tolerates_jitter_and_divides_the_camera_rate(max_fps, expected) -> None:
    admitted = _admitted(FrameRateGate(max_fps), _series(30.0, 1.0, jitter_ms=2.0))
    assert admitted[0] == 0
    assert len(admitted) in expected


@pytest.mark.parametrize("max_fps", [0.0, -1.0, math.inf, math.nan])
def test_gate_and_session_reject_invalid_rates(max_fps: float) -> None:
    with pytest.raises(ValueError, match="max_fps"):
        FrameRateGate(max_fps)
    with pytest.raises(ValueError, match="max_fps"):
        CaptureSession(GatedCamera(), max_fps=max_fps)


@pytest.mark.parametrize("pause", [-1.0, math.inf, math.nan])
def test_session_rejects_invalid_failed_read_pause(pause: float) -> None:
    with pytest.raises(ValueError, match="failed_read_pause_s"):
        CaptureSession(GatedCamera(), failed_read_pause_s=pause)


def test_session_accepts_zero_failed_read_pause() -> None:
    session = CaptureSession(GatedCamera(), failed_read_pause_s=0.0)
    assert not session.is_running


# ---------------------------------------------------------------- session


def test_consumer_takes_the_newest_frame_and_overwrites_are_dropped() -> None:
    camera = GatedCamera()
    session = CaptureSession(camera)
    session.start()
    camera.release(3)
    camera.wait_until_blocked(4)

    frame = session.next_frame()
    assert frame is not None
    assert frame.sequence == 3
    assert session.dropped_frames == 2
    _stop_while_blocked(session, camera)


def test_gate_runs_in_the_producer_so_skipped_frames_never_enter_the_slot() -> None:
    clock = deque([0, 10 * _MS, 33 * _MS, 66 * _MS])
    camera = GatedCamera()
    session = CaptureSession(
        camera,
        max_fps=30.0,
        clock_ns=lambda: clock.popleft() if clock else 10**12,
    )
    session.start()
    sequences: list[int] = []

    camera.release()
    frame = session.next_frame()
    assert frame is not None
    sequences.append(frame.sequence)
    camera.release()
    camera.wait_until_blocked(3)  # the 10 ms frame was skipped by the gate
    camera.release()
    frame = session.next_frame()
    assert frame is not None
    sequences.append(frame.sequence)
    camera.release()
    frame = session.next_frame()
    assert frame is not None
    sequences.append(frame.sequence)

    assert sequences == [1, 3, 4]
    assert session.dropped_frames == 0
    _stop_while_blocked(session, camera)


def test_failed_read_is_reported_as_none() -> None:
    camera = GatedCamera([None])
    session = CaptureSession(camera, failed_read_pause_s=0.0)
    session.start()
    camera.release()
    assert session.next_frame() is None
    _stop_while_blocked(session, camera)


def test_camera_error_reaches_the_consumer_and_persists() -> None:
    error = CameraError("unplugged")
    camera = GatedCamera([error])
    session = CaptureSession(camera)
    session.start()
    camera.release()
    with pytest.raises(CameraError) as caught:
        session.next_frame()
    assert caught.value is error
    with pytest.raises(CameraError):
        session.next_frame()
    session.stop()
    assert not session.is_running


def test_pending_frame_is_delivered_before_a_producer_error() -> None:
    camera = GatedCamera([np.zeros((48, 64, 3), dtype=np.uint8), CameraError("unplugged")])
    session = CaptureSession(camera)
    session.start()
    camera.release(2)
    producer = _capture_thread()
    assert producer is not None
    producer.join(SAFETY_TIMEOUT_S)
    assert not session.is_running

    frame = session.next_frame()
    assert frame is not None
    assert frame.sequence == 1
    with pytest.raises(CameraError):
        session.next_frame()
    session.stop()


def test_stop_wakes_a_blocked_consumer_with_capture_closed() -> None:
    camera = GatedCamera()
    session = CaptureSession(camera)
    session.start()
    outcome: list[object] = []

    def consume() -> None:
        try:
            outcome.append(session.next_frame())
        except CaptureClosed as exc:
            outcome.append(exc)

    consumer = _thread(consume)
    camera.wait_until_blocked(1)
    stopper = _thread(session.stop)
    consumer.join(SAFETY_TIMEOUT_S)
    assert not consumer.is_alive()
    assert isinstance(outcome[0], CaptureClosed)

    camera.release()
    stopper.join(SAFETY_TIMEOUT_S)
    assert not stopper.is_alive()
    session.stop()
    assert not session.is_running
    assert _capture_thread() is None


def test_concurrent_stops_both_return_and_the_thread_dies_once() -> None:
    camera = GatedCamera()
    session = CaptureSession(camera)
    session.start()
    camera.wait_until_blocked(1)
    stoppers = [_thread(session.stop) for _ in range(2)]
    with pytest.raises(CaptureClosed):
        session.next_frame()
    camera.release()
    for stopper in stoppers:
        stopper.join(SAFETY_TIMEOUT_S)
        assert not stopper.is_alive()
    assert not session.is_running
    assert _capture_thread() is None


def test_every_camera_read_is_accounted_for() -> None:
    camera = GatedCamera([None])
    session = CaptureSession(camera, failed_read_pause_s=0.0)
    session.start()
    taken = 0
    failures_seen = 0

    camera.release()
    camera.wait_until_blocked(2)
    assert session.next_frame() is None
    failures_seen += 1

    camera.release(2)
    camera.wait_until_blocked(4)
    frame = session.next_frame()
    assert frame is not None and frame.sequence == 2
    taken += 1

    camera.release()
    camera.wait_until_blocked(5)
    frame = session.next_frame()
    assert frame is not None and frame.sequence == 3
    taken += 1

    camera.release(3)
    camera.wait_until_blocked(8)
    frame = session.next_frame()
    assert frame is not None and frame.sequence == 6
    taken += 1

    taken += _stop_while_blocked(session, camera)
    discarded_at_stop = camera.reads - taken - session.dropped_frames - failures_seen
    assert session.dropped_frames == 3
    assert discarded_at_stop in {0, 1}
    assert camera.reads == taken + session.dropped_frames + failures_seen + discarded_at_stop


def test_stop_waits_for_the_in_flight_read() -> None:
    camera = GatedCamera()
    session = CaptureSession(camera)
    session.start()
    camera.wait_until_blocked(1)

    stopper = _thread(session.stop)
    with pytest.raises(CaptureClosed):
        session.next_frame()
    stopper.join(0.05)
    assert stopper.is_alive()
    assert camera.in_read

    camera.release()
    stopper.join(SAFETY_TIMEOUT_S)
    assert not stopper.is_alive()
    assert not camera.in_read
    assert not session.is_running


def test_failed_reads_are_paced_and_the_pause_is_stop_aware() -> None:
    camera = InstantFailureCamera()
    session = CaptureSession(camera, failed_read_pause_s=10.0)
    session.start()
    assert camera.first_read.wait(SAFETY_TIMEOUT_S)
    assert session.next_frame() is None
    assert camera.reads == 1

    stopper = _thread(session.stop)
    stopper.join(2.0)
    assert not stopper.is_alive()
    assert camera.reads == 1
    assert not session.is_running


def test_second_start_is_rejected() -> None:
    camera = GatedCamera()
    session = CaptureSession(camera)
    session.start()
    with pytest.raises(RuntimeError):
        session.start()
    camera.wait_until_blocked(1)
    _stop_while_blocked(session, camera)


# ---------------------------------------------------------------- --max-frames


def test_max_frames_counts_processed_iterations_not_camera_reads(monkeypatch, tmp_path) -> None:
    from fovea.webcam.engine import GazeSettings
    from fovea.webcam.event_source import WebcamEventSource

    class FreeRunningCamera:
        def __init__(self, *_args, **_kwargs) -> None:
            self.reads = 0

        def connect(self) -> None:
            return None

        def read(self):
            self.reads += 1
            return np.zeros((48, 64, 3), dtype=np.uint8)

        def disconnect(self) -> None:
            return None

    class FakeEstimator:
        def __init__(self, **_kwargs) -> None:
            return None

        def process(self, _frame):
            return None

        def close(self) -> None:
            return None

    cameras: list[FreeRunningCamera] = []

    def make_camera(*args, **kwargs) -> FreeRunningCamera:
        camera = FreeRunningCamera(*args, **kwargs)
        cameras.append(camera)
        return camera

    monkeypatch.setattr("fovea.webcam.event_source.Webcam", make_camera)
    monkeypatch.setattr("fovea.webcam.event_source.FaceLandmarkEstimator", FakeEstimator)

    source = WebcamEventSource(
        GazeSettings(calibration_path=str(tmp_path / "missing.json")),
        max_frames=3,
        max_fps=5.0,
        show_calibration=False,
    )
    events = list(source.events())
    tracking = [event for event in events if isinstance(event, TrackingState)]
    assert len(tracking) == 3
    assert cameras[0].reads > 3
