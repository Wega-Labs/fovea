"""Versioned NDJSON handshake, controls, and generated JSON Schema."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from enum import StrEnum
from types import UnionType
from typing import Any, Literal, get_args, get_origin, get_type_hints

from fovea import __version__
from fovea.events import (
    Blink,
    CalibrationCue,
    Diagnostics,
    Fixation,
    GazePoint,
    Gesture,
    Manipulation,
    TrackingState,
)
from fovea.serialize import event_type_name

PROTOCOL_VERSION = "1.0"
COORDINATE_SPACE = "display_normalized"


@dataclass(frozen=True, slots=True)
class CalibrateCommand:
    cmd: Literal["calibrate"] = "calibrate"


@dataclass(frozen=True, slots=True)
class TestCommand:
    cmd: Literal["test"] = "test"


@dataclass(frozen=True, slots=True)
class PauseCommand:
    cmd: Literal["pause"] = "pause"


@dataclass(frozen=True, slots=True)
class ResumeCommand:
    cmd: Literal["resume"] = "resume"


@dataclass(frozen=True, slots=True)
class QuitCommand:
    cmd: Literal["quit"] = "quit"


type Command = CalibrateCommand | TestCommand | PauseCommand | ResumeCommand | QuitCommand

EVENT_TYPES = (
    GazePoint,
    Fixation,
    Blink,
    Gesture,
    Manipulation,
    TrackingState,
    CalibrationCue,
    Diagnostics,
)
COMMAND_TYPES = (
    CalibrateCommand,
    TestCommand,
    PauseCommand,
    ResumeCommand,
    QuitCommand,
)


class ProtocolError(ValueError):
    """Raised for an invalid protocol control line."""


def hello_payload(backend: str = "mediapipe") -> dict[str, object]:
    """Build the required first line of a successful event stream."""
    return {
        "type": "hello",
        "protocol": PROTOCOL_VERSION,
        "fovea": __version__,
        "backend": backend,
        "coordinate_space": COORDINATE_SPACE,
        "indicator_required": True,
        "capabilities": ["calibration_cue", "diagnostics", "fixation", "blink"],
    }


def hello_json(backend: str = "mediapipe") -> str:
    return json.dumps(hello_payload(backend), ensure_ascii=False, separators=(",", ":"))


def parse_command_line(line: str) -> Command:
    """Parse a legacy bare command or its JSON control-message equivalent."""
    text = line.strip()
    if not text:
        raise ProtocolError("control line is empty")
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"invalid control JSON: {exc.msg}") from exc
        if not isinstance(payload, dict) or set(payload) != {"cmd"}:
            raise ProtocolError("control JSON must contain only a cmd field")
        raw = payload["cmd"]
        if not isinstance(raw, str):
            raise ProtocolError("control cmd must be a string")
        name = raw.lower()
    else:
        name = text.lower()

    for command_type in COMMAND_TYPES:
        command = command_type()
        if command.cmd == name:
            return command
    raise ProtocolError(f"unknown control command: {name}")


def _type_schema(annotation: object) -> dict[str, object]:
    origin = get_origin(annotation)
    if origin is Literal:
        values = list(get_args(annotation))
        if len(values) == 1:
            return {"const": values[0]}
        return {"enum": values}
    if origin is UnionType:
        return {"anyOf": [_type_schema(item) for item in get_args(annotation)]}
    if isinstance(annotation, type) and issubclass(annotation, StrEnum):
        return {"type": "string", "enum": [item.value for item in annotation]}
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    raise TypeError(f"unsupported protocol field type: {annotation!r}")


def _message_schema(message_type: type[Any], discriminator: str) -> dict[str, object]:
    hints = get_type_hints(message_type)
    message_fields = list(fields(message_type))
    properties: dict[str, object] = {}
    if discriminator == "type":
        properties["type"] = {"const": event_type_name(message_type)}
    for message_field in message_fields:
        properties[message_field.name] = _type_schema(hints[message_field.name])
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": (["type"] if discriminator == "type" else [])
        + [message_field.name for message_field in message_fields],
    }


def protocol_schema() -> dict[str, object]:
    """Generate protocol v1 schema directly from event and command dataclasses."""
    definitions: dict[str, object] = {}
    references: list[dict[str, str]] = []
    for event_type in EVENT_TYPES:
        name = event_type_name(event_type)
        definitions[name] = _message_schema(event_type, "type")
        references.append({"$ref": f"#/$defs/{name}"})
    for command_type in COMMAND_TYPES:
        command = command_type()
        name = f"command_{command.cmd}"
        definitions[name] = _message_schema(command_type, "cmd")
        references.append({"$ref": f"#/$defs/{name}"})
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/Wega-Labs/fovea/schema/fovea-protocol-v1.json",
        "title": "Fovea protocol v1 events and controls",
        "oneOf": references,
        "$defs": definitions,
    }


def protocol_schema_text() -> str:
    return json.dumps(protocol_schema(), ensure_ascii=False, indent=2) + "\n"
