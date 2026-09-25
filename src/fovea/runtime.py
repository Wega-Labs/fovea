"""Run a blocking EventSource from asyncio or on a managed background thread.

``EventSource.events()`` blocks the calling thread for the life of the camera
session. The adapters here move that loop onto one worker thread and hand the
events to an asyncio consumer (``AsyncEventSource``) or to a sink callback
(``EventPump``). Both close the source to interrupt a blocked ``events()`` call,
so the wrapped source must make ``close()`` safe to call from another thread.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol, Self

from fovea.events import FoveaEvent
from fovea.interfaces import EventSink

logger = logging.getLogger(__name__)

# How often a producer blocked on a full queue re-checks whether it was stopped.
_STOP_POLL_SECONDS = 0.05
DEFAULT_MAX_PENDING = 256


class ClosableEventSource(Protocol):
    """An EventSource whose ``close()`` may be called from any thread."""

    def events(self) -> Iterator[FoveaEvent]:
        """Yield events until the source is closed."""
        ...

    def close(self) -> None:
        """Stop the session and make a blocked ``events()`` call return."""
        ...


@dataclass(frozen=True, slots=True)
class _Failure:
    error: BaseException


class _End:
    pass


_END = _End()
_QueueItem = FoveaEvent | _Failure | _End


def _close_iterator(iterator: Iterator[FoveaEvent]) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


class AsyncEventSource:
    """Iterate a blocking EventSource with ``async for``.

    The source runs on one worker thread and feeds a bounded queue. When the
    consumer falls behind, the worker blocks instead of dropping events, so
    discrete events such as blinks are never lost; the webcam capture thread
    already keeps only the latest frame. A failure inside ``events()`` is
    re-raised from the consumer's ``async for``.

    Use ``async with`` or call ``aclose()`` so the camera is released when the
    consumer stops early::

        async with AsyncEventSource(WebcamEventSource(settings)) as source:
            async for event in source:
                ...
    """

    def __init__(
        self,
        source: ClosableEventSource,
        *,
        max_pending: int = DEFAULT_MAX_PENDING,
    ) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be at least 1")
        self._source = source
        self._max_pending = max_pending
        self._queue: asyncio.Queue[_QueueItem] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._closed = False

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> FoveaEvent:
        if self._closed:
            raise StopAsyncIteration
        queue = self._ensure_started()
        item = await queue.get()
        if isinstance(item, _End):
            await self.aclose()
            raise StopAsyncIteration
        if isinstance(item, _Failure):
            await self.aclose()
            raise item.error
        return item

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the source and wait for the worker thread to finish. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._source.close()
        thread = self._thread
        if thread is not None:
            await asyncio.to_thread(thread.join)
            logger.debug("async event source stopped")

    def _ensure_started(self) -> asyncio.Queue[_QueueItem]:
        if self._queue is not None:
            return self._queue
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self._max_pending)
        self._thread = threading.Thread(
            target=self._produce,
            name="fovea-async-event-source",
            daemon=True,
        )
        self._thread.start()
        logger.debug("async event source started")
        return self._queue

    def _produce(self) -> None:
        iterator = self._source.events()
        try:
            for event in iterator:
                if not self._deliver(event):
                    return
        except BaseException as exc:
            self._deliver(_Failure(exc))
            return
        finally:
            _close_iterator(iterator)
        self._deliver(_END)

    def _deliver(self, item: _QueueItem) -> bool:
        """Hand one item to the event loop; return False once stopped."""
        loop, queue = self._loop, self._queue
        assert loop is not None and queue is not None
        if self._stop.is_set():
            return False
        try:
            future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
        except RuntimeError:
            # The event loop closed without aclose(); nobody is left to consume.
            return False
        while True:
            try:
                future.result(timeout=_STOP_POLL_SECONDS)
                return True
            except concurrent.futures.TimeoutError:
                if self._stop.is_set():
                    future.cancel()
                    return False
            except concurrent.futures.CancelledError:
                return False


SinkCallback = Callable[[FoveaEvent], None]


class EventPump:
    """Run an EventSource on a background thread and deliver each event to a sink.

    ``start()`` and ``stop()`` are thread-safe and idempotent. ``stop()`` closes
    the source, waits for the worker thread, and re-raises the first failure
    from the source or the sink. A sink may call ``stop()`` itself; the pump
    then stops after the current event without waiting on its own thread::

        with EventPump(WebcamEventSource(settings), print):
            ...
    """

    def __init__(
        self,
        source: ClosableEventSource,
        sink: EventSink | SinkCallback,
    ) -> None:
        self._source = source
        self._publish: SinkCallback = sink if callable(sink) else sink.publish
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: BaseException | None = None

    @property
    def running(self) -> bool:
        """Whether the worker thread is alive."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def error(self) -> BaseException | None:
        """The first failure from the source or sink in the current run, if any."""
        return self._error

    def start(self) -> None:
        """Start delivering events. A no-op while already running."""
        with self._lock:
            if self.running:
                return
            self._stop.clear()
            self._error = None
            thread = threading.Thread(target=self._run, name="fovea-event-pump", daemon=True)
            self._thread = thread
            thread.start()
            logger.debug("event pump started")

    def stop(self, timeout: float | None = None) -> None:
        """Stop the pump, then re-raise the run's failure if there was one."""
        self._shutdown(timeout)
        error, self._error = self._error, None
        if error is not None:
            raise error

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.stop()
        else:
            # Do not mask the exception that is already propagating.
            self._shutdown(None)

    def _shutdown(self, timeout: float | None) -> None:
        with self._lock:
            thread = self._thread
            self._stop.set()
            self._source.close()
        if thread is None or thread is threading.current_thread():
            return
        thread.join(timeout)
        if thread.is_alive():
            logger.warning("event pump did not stop within %s seconds", timeout)

    def _run(self) -> None:
        iterator = self._source.events()
        try:
            for event in iterator:
                if self._stop.is_set():
                    break
                self._publish(event)
        except BaseException as exc:
            self._error = exc
            self._source.close()
        finally:
            _close_iterator(iterator)
            logger.debug("event pump stopped")
