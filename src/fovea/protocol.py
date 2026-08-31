"""Versioned NDJSON handshake, controls, and generated JSON Schema."""

from __future__ import annotations

import json
import math
from dataclasses import MISSING, dataclass, fields, is_dataclass
from enum import StrEnum
from types import NoneType, UnionType
from typing import Any, Literal, get_args, get_origin, get_type_hints

from fovea import __version__
from fovea.events import (
    Blink,
    CalibrationCue,
    CalibrationDone,
    CalibrationWarning,
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
class CalibrationTargetSpec:
    label: str
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class CalibrateCommand:
    cmd: Literal["calibrate"] = "calibrate"
    targets: tuple[CalibrationTargetSpec, ...] | None = None


@dataclass(frozen=True, slots=True)
class TestCommand:
    cmd: Literal["test"] = "test"
    targets: tuple[CalibrationTargetSpec, ...] | None = None


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
    CalibrationWarning,
    CalibrationDone,
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
        "capabilities": [
            "calibration_cue",
            "diagnostics",
            "fixation",
            "blink",
            "windowed_calibration",
        ],
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
        if not isinstance(payload, dict) or "cmd" not in payload:
            raise ProtocolError("control JSON must contain a cmd field")
        if set(payload) - {"cmd", "targets"}:
            raise ProtocolError("control JSON contains unsupported fields")
        raw = payload["cmd"]
        if not isinstance(raw, str):
            raise ProtocolError("control cmd must be a string")
        name = raw.lower()
        targets_present = "targets" in payload
        targets = _parse_targets(payload.get("targets"))
    else:
        name = text.lower()
        targets_present = False
        targets = None

    if name == "calibrate":
        return CalibrateCommand(targets=targets)
    if name == "test":
        return TestCommand(targets=targets)
    if targets_present:
        raise ProtocolError(f"targets are not valid for the {name} command")
    for command_type in (PauseCommand, ResumeCommand, QuitCommand):
        command = command_type()
        if command.cmd == name:
            return command
    raise ProtocolError(f"unknown control command: {name}")


def _parse_targets(value: object) -> tuple[CalibrationTargetSpec, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ProtocolError("control targets must be an array")
    targets: list[CalibrationTargetSpec] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"label", "x", "y"}:
            raise ProtocolError("each target must contain label, x, and y")
        label = item["label"]
        x = item["x"]
        y = item["y"]
        if not isinstance(label, str) or not label.strip():
            raise ProtocolError("target label must be a non-empty string")
        if (
            not isinstance(x, int | float)
            or isinstance(x, bool)
            or not isinstance(y, int | float)
            or isinstance(y, bool)
        ):
            raise ProtocolError("target coordinates must be numbers")
        coordinate_x = float(x)
        coordinate_y = float(y)
        if not math.isfinite(coordinate_x) or not math.isfinite(coordinate_y):
            raise ProtocolError("target coordinates must be finite")
        if not 0.0 <= coordinate_x <= 1.0 or not 0.0 <= coordinate_y <= 1.0:
            raise ProtocolError("target coordinates must be within [0, 1]")
        targets.append(CalibrationTargetSpec(label, coordinate_x, coordinate_y))
    return tuple(targets)


def _type_schema(annotation: object) -> dict[str, object]:
    origin = get_origin(annotation)
    if origin is Literal:
        values = list(get_args(annotation))
        if len(values) == 1:
            return {"const": values[0]}
        return {"enum": values}
    if origin is UnionType:
        return {"anyOf": [_type_schema(item) for item in get_args(annotation)]}
    if origin is tuple:
        item_types = get_args(annotation)
        if len(item_types) != 2 or item_types[1] is not Ellipsis:
            raise TypeError(f"unsupported protocol tuple type: {annotation!r}")
        schema: dict[str, object] = {
            "type": "array",
            "items": _type_schema(item_types[0]),
        }
        if item_types[0] is CalibrationTargetSpec:
            schema["minItems"] = 5
        return schema
    if annotation is NoneType:
        return {"type": "null"}
    if isinstance(annotation, type) and issubclass(annotation, StrEnum):
        return {"type": "string", "enum": [item.value for item in annotation]}
    if isinstance(annotation, type) and is_dataclass(annotation):
        return _plain_dataclass_schema(annotation)
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    raise TypeError(f"unsupported protocol field type: {annotation!r}")


def _plain_dataclass_schema(message_type: type[Any]) -> dict[str, object]:
    hints = get_type_hints(message_type)
    message_fields = list(fields(message_type))
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            message_field.name: _type_schema(hints[message_field.name])
            for message_field in message_fields
        },
        "required": [
            message_field.name
            for message_field in message_fields
            if message_field.default is MISSING and message_field.default_factory is MISSING
        ],
    }


def _message_schema(message_type: type[Any], discriminator: str) -> dict[str, object]:
    hints = get_type_hints(message_type)
    message_fields = list(fields(message_type))
    properties: dict[str, object] = {}
    if discriminator == "type":
        properties["type"] = {"const": event_type_name(message_type)}
    for message_field in message_fields:
        properties[message_field.name] = _type_schema(hints[message_field.name])
    if discriminator == "type":
        required = ["type", *(message_field.name for message_field in message_fields)]
    else:
        required = [
            "cmd",
            *(
                message_field.name
                for message_field in message_fields
                if message_field.name != "cmd"
                and message_field.default is MISSING
                and message_field.default_factory is MISSING
            ),
        ]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
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
