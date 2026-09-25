from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from fovea import AsyncEventSource, EventPump
from fovea.events import Blink, Eye, FoveaEvent
from fovea.serialize import to_json
from fovea.webcam.engine import GazeSettings
from fovea.webcam.event_source import WebcamEventSource
from fovea.webcam.fixtures import ReplayEventSource

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic" / "frontal_30f.jsonl"
SAFETY_TIMEOUT_S = 5.0


def blink(index: int) -> Blink:
    return Blink(eye=Eye.BOTH, duration_ms=100.0, confidence=1.0, timestamp_ns=index)


class FakeSource:
    """A thread-safe source that can emit a fixed stream, fail, or block until closed."""

    def __init__(
        self,
        count: int = 0,
        *,
        fail_at: int | None = None,
        block_after: int | None = None,
    ) -> None:
        self.count = count
        self.fail_at = fail_at
        self.block_after = block_after
        self.produced = 0
        self.close_calls = 0
        self.closed = threading.Event()
        self.finished = threading.Event()

    def events(self) -> Iterator[FoveaEvent]:
        self.closed.clear()
        self.finished.clear()
        try:
            index = 0
            while not self.closed.is_set():
                if self.fail_at is not None and index == self.fail_at:
                    raise RuntimeError("camera failed")
                if self.block_after is not None and index >= self.block_after:
                    self.closed.wait(SAFETY_TIMEOUT_S)
                    return
                if self.block_after is None and index >= self.count:
                    return
                self.produced += 1
                yield blink(index)
                index += 1
        finally:
            self.finished.set()

    def close(self) -> None:
        self.close_calls += 1
        self.closed.set()


def replay_source(tmp_path: Path) -> ReplayEventSource:
    return ReplayEventSource(
        path=FIXTURE,
        settings=GazeSettings(calibration_path=str(tmp_path / "missing.json")),
        project_root=tmp_path,
    )


async def collect(source: AsyncEventSource) -> list[FoveaEvent]:
    return [event async for event in source]


def test_async_iteration_matches_sync_replay(tmp_path: Path) -> None:
    expected = [to_json(event) for event in replay_source(tmp_path).events()]

    events = asyncio.run(collect(AsyncEventSource(replay_source(tmp_path))))

    assert expected
    assert [to_json(event) for event in events] == expected


def test_async_iteration_preserves_order_under_backpressure() -> None:
    source = FakeSource(50)

    async def slow_consumer() -> list[FoveaEvent]:
        seen: list[FoveaEvent] = []
        async for event in AsyncEventSource(source, max_pending=2):
            # The producer may only run a bounded distance ahead of the consumer.
            assert source.produced - len(seen) <= 2 + 2
            seen.append(event)
            await asyncio.sleep(0)
        return seen

    events = asyncio.run(slow_consumer())

    assert [event.timestamp_ns for event in events] == list(range(50))


def test_async_break_with_aclose_stops_blocked_source() -> None:
    source = FakeSource(block_after=3)

    async def consume() -> list[FoveaEvent]:
        async with AsyncEventSource(source) as stream:
            seen = []
            async for event in stream:
                seen.append(event)
                if len(seen) == 3:
                    break
        return seen

    started = time.monotonic()
    events = asyncio.run(consume())

    assert len(events) == 3
    assert source.close_calls >= 1
    assert source.finished.is_set()
    assert time.monotonic() - started < SAFETY_TIMEOUT_S


def test_async_reraises_source_failure_and_closes() -> None:
    source = FakeSource(10, fail_at=2)

    async def consume() -> list[FoveaEvent]:
        seen = []
        with pytest.raises(RuntimeError, match="camera failed"):
            async for event in AsyncEventSource(source):
                seen.append(event)
        return seen

    events = asyncio.run(consume())

    assert len(events) == 2
    assert source.close_calls >= 1


def test_async_cancelled_wait_does_not_lose_events() -> None:
    source = FakeSource(block_after=1)

    async def consume() -> list[FoveaEvent]:
        async with AsyncEventSource(source) as stream:
            first = await anext(stream)
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(anext(stream), timeout=0.05)
            return [first]

    events = asyncio.run(consume())

    assert [event.timestamp_ns for event in events] == [0]
    assert source.finished.is_set()


def test_async_aclose_is_idempotent_and_ends_iteration() -> None:
    source = FakeSource(5)

    async def consume() -> None:
        stream = AsyncEventSource(source)
        await stream.aclose()
        await stream.aclose()
        assert [event async for event in stream] == []

    asyncio.run(consume())
    assert source.close_calls == 1


def test_async_rejects_empty_queue() -> None:
    with pytest.raises(ValueError, match="max_pending"):
        AsyncEventSource(FakeSource(), max_pending=0)


def test_pump_delivers_every_event_to_a_callback() -> None:
    source = FakeSource(20)
    received: list[FoveaEvent] = []
    pump = EventPump(source, received.append)

    pump.start()
    assert source.finished.wait(SAFETY_TIMEOUT_S)
    pump.stop(SAFETY_TIMEOUT_S)

    assert [event.timestamp_ns for event in received] == list(range(20))
    assert not pump.running


def test_pump_accepts_an_event_sink() -> None:
    class Sink:
        def __init__(self) -> None:
            self.events: list[FoveaEvent] = []

        def publish(self, event: FoveaEvent) -> None:
            self.events.append(event)

    sink = Sink()
    source = FakeSource(3)
    with EventPump(source, sink):
        assert source.finished.wait(SAFETY_TIMEOUT_S)

    assert len(sink.events) == 3


def test_pump_start_and_stop_are_idempotent_and_thread_safe() -> None:
    source = FakeSource(block_after=0)
    pump = EventPump(source, lambda _event: None)
    starters = [threading.Thread(target=pump.start) for _ in range(8)]
    for thread in starters:
        thread.start()
    for thread in starters:
        thread.join(SAFETY_TIMEOUT_S)
    assert pump.running
    assert sum(t.name == "fovea-event-pump" for t in threading.enumerate()) == 1

    stoppers = [threading.Thread(target=pump.stop, args=(SAFETY_TIMEOUT_S,)) for _ in range(8)]
    for thread in stoppers:
        thread.start()
    for thread in stoppers:
        thread.join(SAFETY_TIMEOUT_S)

    assert not pump.running
    assert source.finished.is_set()


def test_pump_stops_blocked_source_promptly() -> None:
    source = FakeSource(block_after=1)
    received: list[FoveaEvent] = []
    pump = EventPump(source, received.append)
    pump.start()
    deadline = time.monotonic() + SAFETY_TIMEOUT_S
    while not received and time.monotonic() < deadline:
        time.sleep(0.001)

    started = time.monotonic()
    pump.stop(SAFETY_TIMEOUT_S)

    assert len(received) == 1
    assert source.finished.is_set()
    assert time.monotonic() - started < SAFETY_TIMEOUT_S


def test_sink_may_stop_the_pump_without_deadlock() -> None:
    source = FakeSource(block_after=100)
    received: list[FoveaEvent] = []
    pump: EventPump

    def sink(event: FoveaEvent) -> None:
        received.append(event)
        if len(received) == 2:
            pump.stop()

    pump = EventPump(source, sink)
    pump.start()

    assert source.finished.wait(SAFETY_TIMEOUT_S)
    assert len(received) == 2


def test_pump_stop_reraises_sink_failure_and_closes_source() -> None:
    source = FakeSource(block_after=100)

    def sink(_event: FoveaEvent) -> None:
        raise ValueError("host rejected event")

    pump = EventPump(source, sink)
    pump.start()
    assert source.finished.wait(SAFETY_TIMEOUT_S)

    assert isinstance(pump.error, ValueError)
    with pytest.raises(ValueError, match="host rejected event"):
        pump.stop(SAFETY_TIMEOUT_S)
    assert source.close_calls >= 1
    pump.stop(SAFETY_TIMEOUT_S)  # The failure is reported once.


def test_pump_stop_reraises_source_failure() -> None:
    source = FakeSource(10, fail_at=1)
    pump = EventPump(source, lambda _event: None)
    pump.start()
    assert source.finished.wait(SAFETY_TIMEOUT_S)

    with pytest.raises(RuntimeError, match="camera failed"):
        pump.stop(SAFETY_TIMEOUT_S)


def test_pump_context_does_not_mask_the_body_exception() -> None:
    source = FakeSource(block_after=0)

    def sink(_event: FoveaEvent) -> None:
        raise AssertionError("unreachable")

    with pytest.raises(KeyError), EventPump(source, sink):
        raise KeyError("body failed")

    assert source.finished.wait(SAFETY_TIMEOUT_S)


def test_pump_can_restart_after_stop() -> None:
    source = FakeSource(4)
    received: list[FoveaEvent] = []
    pump = EventPump(source, received.append)

    for _ in range(2):
        pump.start()
        assert source.finished.wait(SAFETY_TIMEOUT_S)
        pump.stop(SAFETY_TIMEOUT_S)

    assert len(received) == 8


def test_replay_source_is_a_context_manager(tmp_path: Path) -> None:
    with replay_source(tmp_path) as source:
        first = next(source.events())
    assert first is not None
    assert source._closed


def test_webcam_source_context_manager_closes() -> None:
    source = WebcamEventSource(GazeSettings())
    source._closed = False

    with source as entered:
        assert entered is source

    assert source._closed


def test_package_logger_is_silent_by_default() -> None:
    handlers = logging.getLogger("fovea").handlers
    assert any(isinstance(handler, logging.NullHandler) for handler in handlers)
