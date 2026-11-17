from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "python-minimal" / "main.py"


def load_example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fovea_python_minimal_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_python_minimal_example_replays_the_fixture(capsys: pytest.CaptureFixture[str]) -> None:
    counts = load_example().main([])

    output = capsys.readouterr().out
    assert counts["GazePoint"] > 0
    assert counts["TrackingState"] > 0
    assert "gaze: x=" in output
    assert output.rstrip().splitlines()[-1].startswith("events: ")
