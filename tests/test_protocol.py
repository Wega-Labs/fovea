from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from fovea import __version__
from fovea.cli import main
from fovea.protocol import (
    COMMAND_TYPES,
    EVENT_TYPES,
    PROTOCOL_VERSION,
    ProtocolError,
    hello_payload,
    parse_command_line,
    protocol_schema_text,
)
from fovea.serialize import event_type_name

SCHEMA_PATH = Path(__file__).parents[1] / "schema" / "fovea-protocol-v1.json"


def test_hello_declares_protocol_backend_and_safety_requirement() -> None:
    hello = hello_payload()
    assert hello == {
        "type": "hello",
        "protocol": PROTOCOL_VERSION,
        "fovea": __version__,
        "backend": "mediapipe",
        "coordinate_space": "display_normalized",
        "indicator_required": True,
        "capabilities": ["calibration_cue", "diagnostics", "fixation", "blink"],
    }


@pytest.mark.parametrize("command_type", COMMAND_TYPES)
def test_bare_and_json_controls_are_equivalent(command_type: type[object]) -> None:
    command = command_type()
    assert parse_command_line(command.cmd) == command
    assert parse_command_line(json.dumps({"cmd": command.cmd})) == command


@pytest.mark.parametrize(
    "line",
    ["", "unknown", "[]", "{}", '{"cmd":1}', '{"cmd":"quit","extra":true}'],
)
def test_invalid_controls_are_rejected(line: str) -> None:
    with pytest.raises(ProtocolError):
        parse_command_line(line)


def test_committed_schema_matches_dataclasses() -> None:
    assert SCHEMA_PATH.read_text(encoding="utf-8") == protocol_schema_text()
    schema = json.loads(protocol_schema_text())
    definitions = schema["$defs"]
    for event_type in EVENT_TYPES:
        assert event_type_name(event_type) in definitions
    for command_type in COMMAND_TYPES:
        assert f"command_{command_type().cmd}" in definitions


def test_schema_command_prints_committed_schema(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert main(["schema"]) == 0
    assert capsys.readouterr().out == SCHEMA_PATH.read_text(encoding="utf-8")
