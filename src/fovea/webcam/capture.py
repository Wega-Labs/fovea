"""Threaded frame capture with capture timestamps and a latest-frame hand-off.

The producer thread owns only ``read()``; the caller keeps ``connect()`` and
``disconnect()``. The hand-off holds at most one frame, so a slow consumer
never queues stale frames: it always takes the newest admitted frame and the
frames it overwrote are counted as dropped.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    """One camera frame together with the clocks sampled when its read returned."""

    pixels: NDArray[np.uint8]
    # time.monotonic_ns() sampled right after read() returned: the latency clock.
    captured_ns: int
    # time.time_ns() sampled immediately after captured_ns: the wire timestamp.
    timestamp_ns: int
    # 1-based count of successful reads this session; gaps mark frames the fps gate skipped.
    sequence: int


class FrameCapture(Protocol):
    """What the producer needs from a camera; ``Webcam`` satisfies it structurally."""

    def read(self) -> NDArray[np.uint8] | None: ...


class CaptureClosed(Exception):
    """Raised by ``next_frame()`` once ``stop()`` has been called."""


class FrameRateGate:
    """Admit at most ``max_fps`` frames per second, skipping the rest.

    The gate tolerates a quarter interval of jitter so a camera running at about
    the cap passes nearly every frame, and it never accumulates debt: a late
    frame does not entitle the next one to arrive early.
    """

    def __init__(self, max_fps: float) -> None:
        if not math.isfinite(max_fps) or max_fps <= 0.0:
            raise ValueError("max_fps must be a finite number greater than zero")
        self._interval_ns = max(1, round(1e9 / max_fps))
        self._tolerance_ns = self._interval_ns // 4
        self._next_due_ns: int | None = None

    def admits(self, captured_ns: int) -> bool:
        if self._next_due_ns is None:
            self._next_due_ns = captured_ns + self._interval_ns
            return True
        if captured_ns < self._next_due_ns - self._tolerance_ns:
            return False
        self._next_due_ns = (
            max(self._next_due_ns, captured_ns - self._tolerance_ns) + self._interval_ns
        )
        return True


class LatestFrameSlot:
    """Single-item hand-off between the capture thread and the consumer.

    The item is one of: empty, a frame, or a read-failure marker. Two terminal
    flags exist: closed (intentional stop) and error (the producer raised).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._frame: CapturedFrame | None = None
        self._read_failed = False
        self._closed = False
        self._error: Exception | None = None
        self.dropped = 0

    def put_frame(self, frame: CapturedFrame) -> None:
        """Store ``frame``, replacing a pending frame (counted) or marker (not counted)."""
        with self._changed:
            if self._closed:
                return
            if self._frame is not None:
                self.dropped += 1
            self._frame = frame
            self._read_failed = False
            self._changed.notify()

    def put_read_failure(self) -> None:
        """Record that a read failed; never displaces a pending frame."""
        with self._changed:
            if self._closed or self._frame is not None:
                return
            self._read_failed = True
            self._changed.notify()

    def fail(self, exc: Exception) -> None:
        """Record a terminal producer error and wake waiters."""
        with self._changed:
            self._error = exc
            self._changed.notify_all()

    def close(self) -> None:
        """Terminate the hand-off, discarding any pending item, and wake waiters."""
        with self._changed:
            self._closed = True
            self._frame = None
            self._read_failed = False
            self._changed.notify_all()

    def take(self) -> CapturedFrame | None:
        """Block for the next item.

        Returns the pending frame, or ``None`` when the camera's read failed.
        Raises ``CaptureClosed`` once closed, and the stored producer error
        (on every call) once no item is pending.
        """
        with self._changed:
            while True:
                if self._closed:
                    raise CaptureClosed
                if self._frame is not None:
                    frame = self._frame
                    self._frame = None
                    return frame
                if self._read_failed:
                    self._read_failed = False
                    return None
                if self._error is not None:
                    raise self._error
                self._changed.wait()


class CaptureSession:
    """Run ``camera.read()`` on a daemon thread and hand the newest frame to the consumer."""

    def __init__(
        self,
        camera: FrameCapture,
        *,
        max_fps: float | None = None,
        failed_read_pause_s: float = 0.05,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        wall_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if not math.isfinite(failed_read_pause_s) or failed_read_pause_s < 0.0:
            raise ValueError("failed_read_pause_s must be a finite number zero or greater")
        self._camera = camera
        self._gate = None if max_fps is None else FrameRateGate(max_fps)
        self._failed_read_pause_s = failed_read_pause_s
        self._clock_ns = clock_ns
        self._wall_ns = wall_ns
        self._slot = LatestFrameSlot()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Spawn the capture thread; the camera must already be connected."""
        if self._thread is not None:
            raise RuntimeError("capture session already started")
        self._thread = threading.Thread(target=self._run, name="fovea-capture", daemon=True)
        self._thread.start()

    def next_frame(self) -> CapturedFrame | None:
        """Block for the newest frame; ``None`` means the camera's read failed."""
        return self._slot.take()

    def stop(self) -> None:
        """Stop the capture thread and wait for it; idempotent and thread-safe.

        The in-flight ``read()`` is allowed to return, so the camera is never
        released while a read is running. The camera itself is not touched.
        """
        self._stopping.set()
        self._slot.close()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    @property
    def dropped_frames(self) -> int:
        """Admitted frames overwritten before the consumer took them (cumulative)."""
        return self._slot.dropped

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def _run(self) -> None:
        sequence = 0
        while not self._stopping.is_set():
            try:
                pixels = self._camera.read()
            except Exception as exc:  # forwarded to the consumer verbatim
                self._slot.fail(exc)
                return
            captured_ns = self._clock_ns()
            timestamp_ns = self._wall_ns()
            if pixels is None:
                self._slot.put_read_failure()
                self._stopping.wait(self._failed_read_pause_s)
                continue
            sequence += 1
            if self._gate is not None and not self._gate.admits(captured_ns):
                continue
            self._slot.put_frame(CapturedFrame(pixels, captured_ns, timestamp_ns, sequence))
