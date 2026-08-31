"""Command-line process boundary for Fovea hosts."""

from __future__ import annotations

import argparse
import json
import math
import platform
import queue
import signal
import sys
import threading
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from types import FrameType
from typing import NoReturn

from fovea import __version__
from fovea.benchmark import BenchmarkConfig, run_live_benchmark
from fovea.interfaces import EventSource
from fovea.privacy import default_diagnostics_dir, parse_retention, purge_expired_diagnostics
from fovea.protocol import (
    CalibrateCommand,
    Command,
    ProtocolError,
    QuitCommand,
    TargetsCommand,
    TestCommand,
    hello_json,
    parse_command_line,
    protocol_schema_text,
)
from fovea.serialize import to_json
from fovea.webcam.backend import BACKEND_NAMES, backend_available
from fovea.webcam.calibration import CalibrationTarget
from fovea.webcam.camera import CameraError
from fovea.webcam.engine import GazeSettings
from fovea.webcam.event_source import WebcamEventSource
from fovea.webcam.landmarks import MediaPipeUnavailableError, resolve_model_path
from fovea.webcam.model import (
    DEFAULT_MODEL_PATH,
    ModelChecksumError,
    verify_face_landmarker,
)
from fovea.webcam.targeting import TargetRect

type SourceFactory = Callable[..., EventSource]
type SignalHandler = Callable[[int, FrameType | None], object] | int | None


class CliUsageError(ValueError):
    """Raised for command-line configuration errors that map to exit code 2."""


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise CliUsageError(message)


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_int(value: str) -> int:
    parsed = _nonnegative_int(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite number zero or greater")
    return parsed


def _retention_seconds(value: str) -> float:
    try:
        return parse_retention(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _add_capture_arguments(parser: argparse.ArgumentParser, *, require_ndjson: bool) -> None:
    parser.add_argument("--ndjson", action="store_true", required=require_ndjson)
    parser.add_argument("--camera", type=_nonnegative_int, default=0, metavar="N")
    parser.add_argument("--width", type=_positive_int, default=640, metavar="W")
    parser.add_argument("--height", type=_positive_int, default=480, metavar="H")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--calibration-path", type=Path, metavar="P")
    parser.add_argument("--model", type=Path, metavar="P")
    parser.add_argument("--backend", choices=BACKEND_NAMES, default="mediapipe")
    parser.add_argument("--max-frames", type=_positive_int, metavar="N")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument(
        "--diagnostics-retention",
        type=_retention_seconds,
        default=_retention_seconds("24h"),
        metavar="DURATION",
    )
    parser.add_argument("--diagnostics-dir", type=Path, metavar="P")
    parser.add_argument("--display-id", metavar="ID")
    parser.add_argument("--display-width", type=_positive_int, default=1280, metavar="W")
    parser.add_argument("--display-height", type=_positive_int, default=720, metavar="H")


def _build_parser() -> _JsonArgumentParser:
    parser = _JsonArgumentParser(prog="fovea", description="Local webcam gaze event engine")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="stream gaze events")
    _add_capture_arguments(run, require_ndjson=True)
    run.add_argument("--calibrate", action="store_true")

    calibrate = commands.add_parser("calibrate", help="run the calibration wizard")
    _add_capture_arguments(calibrate, require_ndjson=False)

    test = commands.add_parser("test", help="run the calibrated gaze test")
    _add_capture_arguments(test, require_ndjson=False)

    bench = commands.add_parser("bench", help="run the guided live benchmark")
    _add_capture_arguments(bench, require_ndjson=False)
    bench.set_defaults(diagnostics=True)
    bench.add_argument("--screen-width-cm", type=_positive_float, required=True)
    bench.add_argument("--screen-height-cm", type=_positive_float, required=True)
    bench.add_argument("--camera-name", required=True)
    bench.add_argument("--lighting", required=True)
    bench.add_argument("--glasses", required=True)
    bench.add_argument("--machine", default=platform.platform())
    bench.add_argument("--fixation-seconds", type=_positive_float, default=2.0)
    bench.add_argument("--yaw-seconds", type=_nonnegative_float, default=2.0)
    bench.add_argument("--drift-seconds", type=_nonnegative_float, default=600.0)
    bench.add_argument("--output", type=Path, default=Path("fovea-benchmark.json"))
    bench.add_argument("--yes", action="store_true", help="skip phase confirmation prompts")

    doctor = commands.add_parser("doctor", help="print environment and camera diagnostics")
    doctor.add_argument("--backend", choices=BACKEND_NAMES, default="mediapipe")
    commands.add_parser("schema", help="print the protocol JSON Schema")
    return parser


def _emit_error(message: str) -> None:
    print(
        json.dumps(
            {"type": "error", "message": message}, ensure_ascii=False, separators=(",", ":")
        ),
        flush=True,
    )


def _read_stdin(commands: queue.Queue[Command]) -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            commands.put(parse_command_line(line))
        except ProtocolError as exc:
            print(f"Ignoring invalid control command: {exc}", file=sys.stderr, flush=True)


def _source_method(source: EventSource, name: str, *args: object) -> None:
    method = getattr(source, name, None)
    if callable(method):
        method(*args)
    else:
        print(f"Control command unavailable: {name}", file=sys.stderr, flush=True)


def _drain_commands(
    source: EventSource,
    commands: queue.Queue[Command],
    paused: bool,
) -> tuple[bool, bool]:
    should_quit = False
    while True:
        try:
            command = commands.get_nowait()
        except queue.Empty:
            break
        if isinstance(command, CalibrateCommand):
            if command.targets is None:
                _source_method(source, "start_calibration")
            else:
                calibration_targets = tuple(
                    CalibrationTarget(target.label, target.x, target.y)
                    for target in command.targets
                )
                _source_method(source, "start_calibration", calibration_targets)
        elif isinstance(command, TestCommand):
            if command.targets is None:
                _source_method(source, "start_gaze_test")
            else:
                calibration_targets = tuple(
                    CalibrationTarget(target.label, target.x, target.y)
                    for target in command.targets
                )
                _source_method(source, "start_gaze_test", calibration_targets)
        elif isinstance(command, TargetsCommand):
            registered_targets = tuple(
                TargetRect(target.id, target.x, target.y, target.w, target.h)
                for target in command.items
            )
            _source_method(source, "set_targets", registered_targets)
        elif command.cmd == "pause":
            paused = True
        elif command.cmd == "resume":
            paused = False
        elif command.cmd == "quit":
            should_quit = True
    return paused, should_quit


def _close_source(source: EventSource) -> None:
    close = getattr(source, "close", None)
    if callable(close):
        close()


def _stream(source: EventSource, backend: str = "mediapipe") -> int:
    print(hello_json(backend), flush=True)
    commands: queue.Queue[Command] = queue.Queue()
    reader = threading.Thread(target=_read_stdin, args=(commands,), daemon=True)
    reader.start()

    def request_stop(_signum: int, _frame: FrameType | None) -> None:
        commands.put(QuitCommand())

    previous_handlers: dict[signal.Signals, SignalHandler] = {}
    if threading.current_thread() is threading.main_thread():
        for stop_signal in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[stop_signal] = signal.signal(stop_signal, request_stop)

    paused = False
    iterator = iter(source.events())
    try:
        while True:
            paused, should_quit = _drain_commands(source, commands, paused)
            if should_quit:
                return 0
            try:
                event = next(iterator)
            except StopIteration:
                return 0
            paused, should_quit = _drain_commands(source, commands, paused)
            if should_quit:
                return 0
            if not paused:
                print(to_json(event), flush=True)
    finally:
        for stop_signal, handler in previous_handlers.items():
            signal.signal(stop_signal, handler)


def _settings(args: argparse.Namespace) -> GazeSettings:
    settings = GazeSettings()
    if args.calibration_path is not None:
        settings.calibration_path = str(args.calibration_path)
    return settings


def _make_source(args: argparse.Namespace, factory: SourceFactory) -> EventSource:
    return factory(
        settings=_settings(args),
        project_root=Path.cwd(),
        device_index=args.camera,
        width=args.width,
        height=args.height,
        backend=args.backend,
        model_path=args.model,
        max_frames=args.max_frames,
        force_calibrate=(args.command == "calibrate" or bool(getattr(args, "calibrate", False))),
        force_test=args.command == "test",
        show_calibration=not args.no_display,
        diagnostics=args.diagnostics,
        display_id=args.display_id,
        display_width=args.display_width,
        display_height=args.display_height,
    )


def _camera_count(limit: int = 5) -> int:
    import cv2

    count = 0
    for index in range(limit):
        capture = cv2.VideoCapture(index)
        try:
            if capture.isOpened():
                count += 1
        finally:
            capture.release()
    return count


def _macos_camera_authorization() -> str | None:
    if sys.platform != "darwin":
        return None
    try:
        from AVFoundation import (  # type: ignore[import-not-found]
            AVCaptureDevice,
            AVMediaTypeVideo,
        )
    except ImportError:
        return "unknown (install fovea-input[macos])"
    status = int(AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo))
    return {0: "not_determined", 1: "restricted", 2: "denied", 3: "authorized"}.get(
        status, f"unknown ({status})"
    )


def _package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not installed"


def _doctor(backend: str = "mediapipe") -> int:
    model_path = DEFAULT_MODEL_PATH
    try:
        verify_face_landmarker(model_path)
        model_status = "verified"
    except (FileNotFoundError, ModelChecksumError) as exc:
        model_status = str(exc)

    print(f"fovea={__version__}")
    print(f"python={platform.python_version()}")
    print(f"backend={backend}")
    print(f"backend_available={'yes' if backend_available(backend) else 'no'}")
    print(f"mediapipe={_package_version('mediapipe')}")
    print(f"opencv={_package_version('opencv-contrib-python')}")
    print(f"numpy={_package_version('numpy')}")
    print(f"model_path={model_path}")
    print(f"model_status={model_status}")
    print(f"camera_count={_camera_count()}")
    authorization = _macos_camera_authorization()
    if authorization is not None:
        print(f"camera_authorization={authorization}")
    return 0


def _benchmark_prompt(message: str, *, assume_yes: bool) -> None:
    print(message, file=sys.stderr, flush=True)
    if assume_yes:
        return
    print("Press Enter when ready.", file=sys.stderr, flush=True)
    if sys.stdin.readline() == "":
        raise CliUsageError("benchmark input ended; pass --yes for non-interactive operation")


def _benchmark_config(args: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        screen_width_cm=args.screen_width_cm,
        screen_height_cm=args.screen_height_cm,
        capture_width=args.width,
        capture_height=args.height,
        camera_name=args.camera_name,
        lighting=args.lighting,
        glasses=args.glasses,
        backend=args.backend,
        camera_index=args.camera,
        fovea_version=__version__,
        machine=args.machine,
        fixation_seconds=args.fixation_seconds,
        yaw_seconds=args.yaw_seconds,
        drift_seconds=args.drift_seconds,
    )


def main(argv: list[str] | None = None, source_factory: SourceFactory | None = None) -> int:
    """Run the CLI and return a stable process exit code."""
    try:
        args = _build_parser().parse_args(argv)
    except CliUsageError as exc:
        _emit_error(str(exc))
        return 2

    if args.command == "doctor":
        return _doctor(args.backend)
    if args.command == "schema":
        print(protocol_schema_text(), end="")
        return 0

    source: EventSource | None = None
    try:
        if args.diagnostics:
            diagnostics_dir = args.diagnostics_dir or default_diagnostics_dir()
            purge_expired_diagnostics(diagnostics_dir, args.diagnostics_retention)
        if source_factory is None:
            verify_face_landmarker(resolve_model_path(args.model))
        source = _make_source(args, source_factory or WebcamEventSource)
        if args.command == "bench":
            report = run_live_benchmark(
                source,
                _benchmark_config(args),
                prompt=lambda message: _benchmark_prompt(message, assume_yes=args.yes),
            )
            rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            args.output.write_text(rendered, encoding="utf-8")
            print(rendered, end="")
            return 0
        return _stream(source, args.backend)
    except CameraError as exc:
        _emit_error(str(exc))
        return 3
    except (FileNotFoundError, MediaPipeUnavailableError, ModelChecksumError) as exc:
        _emit_error(str(exc))
        return 4
    except RuntimeError as exc:
        _emit_error(str(exc))
        return 4
    except (OSError, ValueError) as exc:
        _emit_error(str(exc))
        return 2
    finally:
        if source is not None:
            _close_source(source)
