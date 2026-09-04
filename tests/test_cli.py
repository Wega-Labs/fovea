"""Tests for the NDJSON process boundary."""

from __future__ import annotations

import io
import json
import time
from collections.abc import Iterator
from dataclasses import asdict
from typing import ClassVar

import pytest

from fovea.cli import main
from fovea.events import (
    Blink,
    CalibrationCue,
    CalibrationDone,
    CalibrationUpdated,
    CalibrationWarning,
    Diagnostics,
    Dwell,
    DwellProgress,
    Eye,
    Fixation,
    FoveaEvent,
    GazePoint,
    GazeTestDone,
    GazeTestPoint,
    Gesture,
    GesturePhase,
    Manipulation,
    TargetEnter,
    TargetLeave,
    TrackingState,
    TrackingStatus,
)
from fovea.protocol import hello_json
from fovea.serialize import event_type_name, to_json
from fovea.webcam.calibration import CalibrationTarget
from fovea.webcam.camera import CameraError
from fovea.webcam.engine import GazeSettings
from fovea.webcam.landmarks import MediaPipeUnavailableError
from fovea.webcam.targeting import TargetRect


def _all_event_types() -> list[FoveaEvent]:
    return [
        GazePoint(0.25, 0.75, 0.9, 1, "card-1", 0.3, 0.8),
        TargetEnter("card-1", 2),
        TargetLeave("card-1", 3),
        DwellProgress("card-1", 0.5, 4),
        Dwell("card-1", 5),
        Fixation(0.25, 0.75, 300.0, 0.8, 6),
        Blink(Eye.BOTH, 120.0, 0.7, 7),
        Gesture("pinch", GesturePhase.STARTED, 0.9, 8),
        Manipulation("card-1", GesturePhase.UPDATED, 1.0, 2.0, 1.1, 5.0, 0.8, 9),
        TrackingState(TrackingStatus.UNCERTAIN, 0.5, 10, "Move closer"),
        CalibrationCue("center", 0.5, 0.5, 0, 10, 3, 28, "Look", 11),
        CalibrationWarning("Target coverage is low", 0.2, 12),
        CalibrationDone(5, 0.76, 0.08, 13),
        CalibrationUpdated(5, 0.07, 14),
        GazeTestDone(
            1,
            0.04,
            0.04,
            0.04,
            (GazeTestPoint(0.5, 0.5, 0.53, 0.52, 0.04),),
            15,
        ),
        Diagnostics(30.0, 8.0, 0.2, 1.0, -2.0, 16),
    ]


@pytest.mark.parametrize("event", _all_event_types())
def test_every_event_serializes_to_ndjson(event: FoveaEvent) -> None:
    payload = json.loads(to_json(event))
    assert payload.pop("type") == event_type_name(type(event))
    expected = json.loads(json.dumps(asdict(event)))
    assert payload == expected
    assert "\n" not in to_json(event)


class FakeSource:
    instances: ClassVar[list[FakeSource]] = []

    def __init__(self, events: list[FoveaEvent]) -> None:
        self._events = events
        self.closed = False
        self.calibration_starts = 0
        self.test_starts = 0
        self.calibration_targets: tuple[CalibrationTarget, ...] | None = None
        self.test_targets: tuple[CalibrationTarget, ...] | None = None
        self.targets: tuple[TargetRect, ...] = ()
        self.observations: list[tuple[float, float, float, int | None]] = []
        FakeSource.instances.append(self)

    def events(self) -> Iterator[FoveaEvent]:
        yield from self._events

    def close(self) -> None:
        self.closed = True

    def start_calibration(self, targets: tuple[CalibrationTarget, ...] | None = None) -> None:
        if targets is not None and len(targets) < 5:
            raise ValueError("calibration requires at least 5 targets")
        self.calibration_starts += 1
        self.calibration_targets = targets

    def start_gaze_test(self, targets: tuple[CalibrationTarget, ...] | None = None) -> None:
        self.test_starts += 1
        self.test_targets = targets

    def set_targets(self, targets: tuple[TargetRect, ...]) -> None:
        self.targets = targets

    def observe(
        self,
        x: float,
        y: float,
        weight: float,
        timestamp_ns: int | None,
    ) -> None:
        self.observations.append((x, y, weight, timestamp_ns))


def test_fake_source_produces_expected_ndjson(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    events = [GazePoint(0.1, 0.2, 0.8, 1), TrackingState(TrackingStatus.ACTIVE, 0.9, 2)]
    source = FakeSource(events)

    exit_code = main(
        ["run", "--ndjson", "--no-display"],
        source_factory=lambda **_kwargs: source,
    )

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == [
        hello_json(),
        *(to_json(event) for event in events),
    ]
    assert source.closed


def test_max_frames_reaches_source_and_limits_fake_events(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    received: dict[str, object] = {}

    def factory(**kwargs: object) -> FakeSource:
        received.update(kwargs)
        count = int(kwargs["max_frames"])
        return FakeSource([GazePoint(0.5, 0.5, 1.0, index) for index in range(count)])

    exit_code = main(
        ["run", "--ndjson", "--max-frames", "3", "--no-display"],
        source_factory=factory,
    )

    assert exit_code == 0
    assert received["max_frames"] == 3
    assert len(capsys.readouterr().out.splitlines()) == 4


def test_backend_reaches_source_and_hello(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    received: dict[str, object] = {}

    def factory(**kwargs: object) -> FakeSource:
        received.update(kwargs)
        return FakeSource([])

    exit_code = main(
        ["run", "--ndjson", "--backend", "mediapipe"],
        source_factory=factory,
    )

    assert exit_code == 0
    assert received["backend"] == "mediapipe"
    assert capsys.readouterr().out.splitlines() == [hello_json("mediapipe")]


def test_quit_on_stdin_closes_source(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("quit\n"))

    class SlowSource(FakeSource):
        def events(self) -> Iterator[FoveaEvent]:
            while not self.closed:
                time.sleep(0.001)
                yield GazePoint(0.5, 0.5, 1.0, 1)

    source = SlowSource([])
    assert main(["run", "--ndjson"], source_factory=lambda **_kwargs: source) == 0
    assert source.closed
    assert capsys.readouterr().out.splitlines() == [hello_json()]


def test_stdin_controls_reach_source_between_events(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO('calibrate\n{"cmd":"test"}\n{"cmd":"pause"}\nresume\n'),
    )

    class SlowSource(FakeSource):
        def events(self) -> Iterator[FoveaEvent]:
            time.sleep(0.01)
            yield GazePoint(0.5, 0.5, 1.0, 1)

    source = SlowSource([])
    assert main(["run", "--ndjson"], source_factory=lambda **_kwargs: source) == 0
    assert source.calibration_starts == 1
    assert source.test_starts == 1
    assert capsys.readouterr().out.splitlines()[0] == hello_json()


def test_observe_control_reaches_source(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO('{"cmd":"observe","x":0.2,"y":0.8,"weight":0.5,"timestamp_ns":42}\n'),
    )

    class SlowSource(FakeSource):
        def events(self) -> Iterator[FoveaEvent]:
            time.sleep(0.01)
            yield GazePoint(0.5, 0.5, 1.0, 1)

    source = SlowSource([])
    assert main(["run", "--ndjson"], source_factory=lambda **_kwargs: source) == 0
    assert source.observations == [(0.2, 0.8, 0.5, 42)]
    assert capsys.readouterr().out.splitlines()[0] == hello_json()


def test_no_online_disables_setting(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    received: dict[str, object] = {}

    def factory(**kwargs: object) -> FakeSource:
        received.update(kwargs)
        return FakeSource([])

    assert main(["run", "--ndjson", "--no-online"], source_factory=factory) == 0
    settings = received["settings"]
    assert isinstance(settings, GazeSettings)
    assert settings.online_calibration is False
    assert capsys.readouterr().out.splitlines() == [hello_json()]


def test_custom_calibration_targets_reach_source(monkeypatch, capsys) -> None:
    targets = [
        {"label": "top-left", "x": 0.1, "y": 0.1},
        {"label": "top-right", "x": 0.9, "y": 0.1},
        {"label": "center", "x": 0.5, "y": 0.5},
        {"label": "bottom-left", "x": 0.1, "y": 0.9},
        {"label": "bottom-right", "x": 0.9, "y": 0.9},
    ]
    control = json.dumps({"cmd": "calibrate", "targets": targets})
    monkeypatch.setattr("sys.stdin", io.StringIO(f"{control}\n"))

    class SlowSource(FakeSource):
        def events(self) -> Iterator[FoveaEvent]:
            time.sleep(0.01)
            yield GazePoint(0.5, 0.5, 1.0, 1)

    source = SlowSource([])
    assert main(["run", "--ndjson"], source_factory=lambda **_kwargs: source) == 0
    assert source.calibration_targets is not None
    assert [target.label for target in source.calibration_targets] == [
        target["label"] for target in targets
    ]
    assert capsys.readouterr().out.splitlines()[0] == hello_json()


def test_too_few_calibration_targets_emit_error(monkeypatch, capsys) -> None:
    targets = [
        {"label": "one", "x": 0.1, "y": 0.1},
        {"label": "two", "x": 0.5, "y": 0.5},
        {"label": "three", "x": 0.9, "y": 0.9},
    ]
    control = json.dumps({"cmd": "calibrate", "targets": targets})
    monkeypatch.setattr("sys.stdin", io.StringIO(f"{control}\n"))

    class SlowSource(FakeSource):
        def events(self) -> Iterator[FoveaEvent]:
            time.sleep(0.01)
            yield GazePoint(0.5, 0.5, 1.0, 1)

    source = SlowSource([])
    assert main(["run", "--ndjson"], source_factory=lambda **_kwargs: source) == 2
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == hello_json()
    assert json.loads(lines[-1]) == {
        "type": "error",
        "message": "calibration requires at least 5 targets",
    }


def test_registered_targets_reach_source(monkeypatch, capsys) -> None:
    control = json.dumps(
        {
            "cmd": "targets",
            "items": [{"id": "save", "x": 0.7, "y": 0.8, "w": 0.2, "h": 0.1}],
            "space": "display_normalized",
        }
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(f"{control}\n"))

    class SlowSource(FakeSource):
        def events(self) -> Iterator[FoveaEvent]:
            time.sleep(0.01)
            yield GazePoint(0.5, 0.5, 1.0, 1)

    source = SlowSource([])
    assert main(["run", "--ndjson"], source_factory=lambda **_kwargs: source) == 0
    assert source.targets == (TargetRect("save", 0.7, 0.8, 0.2, 0.1),)
    assert capsys.readouterr().out.splitlines()[0] == hello_json()


@pytest.mark.parametrize(
    ("command", "flag"),
    [("calibrate", "force_calibrate"), ("test", "force_test")],
)
def test_calibrate_and_test_aliases_set_source_mode(
    command: str,
    flag: str,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    received: dict[str, object] = {}

    def factory(**kwargs: object) -> FakeSource:
        received.update(kwargs)
        return FakeSource([])

    assert main([command, "--no-display"], source_factory=factory) == 0
    assert received[flag] is True
    assert capsys.readouterr().out.splitlines() == [hello_json()]


def test_diagnostics_flag_reaches_source(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    received: dict[str, object] = {}

    def factory(**kwargs: object) -> FakeSource:
        received.update(kwargs)
        return FakeSource([])

    assert (
        main(
            [
                "run",
                "--ndjson",
                "--diagnostics",
                "--diagnostics-dir",
                str(tmp_path),
                "--diagnostics-retention",
                "24h",
            ],
            source_factory=factory,
        )
        == 0
    )
    assert received["diagnostics"] is True
    assert capsys.readouterr().out.splitlines() == [hello_json()]


def test_display_identity_arguments_reach_source(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    received: dict[str, object] = {}

    def factory(**kwargs: object) -> FakeSource:
        received.update(kwargs)
        return FakeSource([])

    args = [
        "run",
        "--ndjson",
        "--display-id",
        "studio",
        "--display-width",
        "2560",
        "--display-height",
        "1440",
    ]
    assert main(args, source_factory=factory) == 0
    assert received["display_id"] == "studio"
    assert received["display_width"] == 2560
    assert received["display_height"] == 1440
    assert capsys.readouterr().out.splitlines() == [hello_json()]


def test_bench_writes_json_report(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(
        "fovea.cli.run_live_benchmark",
        lambda *_args, **_kwargs: {"schema": "fovea-benchmark-v1", "accuracy": {}},
    )
    output = tmp_path / "report.json"
    source = FakeSource([])
    args = [
        "bench",
        "--screen-width-cm",
        "30",
        "--screen-height-cm",
        "20",
        "--camera-name",
        "integrated",
        "--lighting",
        "office",
        "--glasses",
        "none",
        "--diagnostics-dir",
        str(tmp_path / "diagnostics"),
        "--output",
        str(output),
        "--yes",
    ]

    assert main(args, source_factory=lambda **_kwargs: source) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema": "fovea-benchmark-v1",
        "accuracy": {},
    }
    assert json.loads(capsys.readouterr().out) == {
        "schema": "fovea-benchmark-v1",
        "accuracy": {},
    }
    assert source.closed


def test_invalid_camera_exits_two_with_one_json_error(capsys) -> None:
    exit_code = main(["run", "--ndjson", "--camera", "-1"])
    lines = capsys.readouterr().out.splitlines()
    assert exit_code == 2
    assert len(lines) == 1
    assert json.loads(lines[0])["type"] == "error"


def test_camera_error_exits_three(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    def broken_factory(**_kwargs: object) -> FakeSource:
        raise CameraError("camera unavailable")

    exit_code = main(["run", "--ndjson"], source_factory=broken_factory)
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert payload == {"type": "error", "message": "camera unavailable"}


def test_model_error_exits_four(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    def broken_factory(**_kwargs: object) -> FakeSource:
        raise MediaPipeUnavailableError("model unavailable")

    exit_code = main(["run", "--ndjson"], source_factory=broken_factory)
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 4
    assert payload == {"type": "error", "message": "model unavailable"}


def test_doctor_reports_versions_model_and_camera(monkeypatch, capsys) -> None:
    monkeypatch.setattr("fovea.cli.verify_face_landmarker", lambda _path: None)
    monkeypatch.setattr("fovea.cli.backend_available", lambda _backend: True)
    monkeypatch.setattr("fovea.cli._camera_count", lambda: 2)
    monkeypatch.setattr("fovea.cli._macos_camera_authorization", lambda: "authorized")

    assert main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert "fovea=" in output
    assert "backend=mediapipe" in output
    assert "backend_available=yes" in output
    assert "mediapipe=" in output
    assert "opencv=" in output
    assert "numpy=" in output
    assert "model_status=verified" in output
    assert "camera_count=2" in output
    assert "camera_authorization=authorized" in output
