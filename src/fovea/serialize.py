"""Stable NDJSON serialization for the public Fovea event contract."""

from __future__ import annotations

import json
import re
from dataclasses import asdict

from fovea.events import FoveaEvent


def event_type_name(cls: type[object]) -> str:
    """Convert an event class name such as ``GazePoint`` to ``gaze_point``."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()


def to_json(event: FoveaEvent) -> str:
    """Serialize one immutable event as a compact JSON object.

    ``StrEnum`` values are string subclasses, so the standard encoder writes
    their values without a custom, lossy fallback serializer.
    """
    payload = {"type": event_type_name(type(event)), **asdict(event)}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
