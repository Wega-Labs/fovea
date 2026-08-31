"""NDJSON CLI for the Fovea gaze engine."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from fovea import GazeSettings, WebcamEventSource


def _snake_case(name: str) -> str:
    """Convert CamelCase to snake_case."""
    result = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            result.append("_")
        result.append(ch.lower())
    return "".join(result)


def main() -> None:
    parser = argparse.ArgumentParser(prog="fovea", description="Fovea gaze engine CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the gaze engine and emit NDJSON events.")
    run.add_argument(
        "--ndjson",
        action="store_true",
        help="Emit events as NDJSON (one JSON object per line).",
    )
    run.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0).",
    )
    run.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera width (default: 640).",
    )
    run.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera height (default: 480).",
    )
    run.add_argument(
        "--calibrate",
        action="store_true",
        help="Force calibration before tracking.",
    )
    run.add_argument(
        "--no-display",
        action="store_true",
        help="Hide the calibration display window.",
    )
    run.add_argument(
        "--calibration-path",
        type=str,
        default="data/gaze_calibration.json",
        help="Path for the calibration data file.",
    )
    run.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the face-landmark model file.",
    )
    run.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after processing N frames.",
    )
    run.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print FPS and latency diagnostics to stderr.",
    )

    args = parser.parse_args()

    if args.command != "run":
        parser.print_help()
        sys.exit(1)

    settings = GazeSettings(calibration_path=args.calibration_path)
    project_root = Path(__file__).resolve().parent.parent

    source = WebcamEventSource(
        settings=settings,
        project_root=project_root,
        device_index=args.camera,
        width=args.width,
        height=args.height,
        model_path=args.model,
        max_frames=args.max_frames,
        force_calibrate=args.calibrate,
        show_calibration=not args.no_display,
    )

    last_time = time.perf_counter()
    frame_count = 0
    fps = 0.0
    t0 = time.perf_counter()

    try:
        for event in source.events():
            now = time.perf_counter()
            dt = max(1e-3, now - last_time)
            last_time = now
            frame_count += 1
            elapsed = now - t0
            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                t0 = now

            if args.ndjson:
                d = dataclasses.asdict(event)
                d["type"] = _snake_case(type(event).__name__)
                print(json.dumps(d), flush=True)

            if args.diagnostics:
                print(
                    f"[fovea] fps={fps:.1f} latency_ms={(dt * 1000):.1f}",
                    file=sys.stderr,
                )
    finally:
        source.close()


if __name__ == "__main__":
    main()
