"""Minimal boundaries for producers and consumers of Fovea events."""

from collections.abc import Iterator
from typing import Protocol

from fovea.events import FoveaEvent


class EventSource(Protocol):
    """A calibrated engine or adapter that produces Fovea events."""

    def events(self) -> Iterator[FoveaEvent]:
        """Yield events until the source is closed."""
        ...


class EventSink(Protocol):
    """An application or platform adapter that consumes Fovea events."""

    def publish(self, event: FoveaEvent) -> None:
        """Handle one immutable event."""
        ...
