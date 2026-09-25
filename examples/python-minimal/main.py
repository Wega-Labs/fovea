"""Minimal in-process Fovea host.

By default it replays the checked-in synthetic landmark fixture, so it needs no camera, face,
model download, or network. Pass --webcam to read the local camera instead (requires the
downloaded model and camera permission).
"""

from __future__ import annotations

import argparse
import tempfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from fovea import FoveaEvent, GazePoint, GazeSettings, TrackingState, WebcamEventSource
from fovea.webcam import ReplayEventSource

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "synthetic" / "frontal_30f.jsonl"


def replay_events(fixture: Path, calibration_dir: Path) -> Iterator[FoveaEvent]:
    # A scratch calibration path keeps the example away from the user's real calibration.
    settings = GazeSettings(calibration_path=str(calibration_dir / "calibration.json"))
    source = ReplayEventSource(path=fixture, settings=settings, project_root=calibration_dir)
    try:
        yield from source.events()
    finally:
        source.close()


def webcam_events(max_frames: int) -> Iterator[FoveaEvent]:
    source = WebcamEventSource(GazeSettings(), max_frames=max_frames, show_calibration=True)
    try:
        yield from source.events()
    finally:
        source.close()


def summarize(events: Iterator[FoveaEvent]) -> Counter[str]:
    """Print tracking changes and gaze points, and count every event type."""
    counts: Counter[str] = Counter()
    for event in events:
        counts[type(event).__name__] += 1
        if isinstance(event, TrackingState):
            detail = f" {event.detail}" if event.detail else ""
            print(f"tracking: {event.status} confidence={event.confidence:.2f}{detail}")
        elif isinstance(event, GazePoint):
            print(f"gaze: x={event.x:.3f} y={event.y:.3f} confidence={event.confidence:.2f}")
    return counts


def main(argv: list[str] | None = None) -> Counter[str]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", type=Path, default=DEFAULT_FIXTURE, help="landmark JSONL to replay"
    )
    parser.add_argument(
        "--webcam", action="store_true", help="read the local camera instead of a fixture"
    )
    parser.add_argument("--max-frames", type=int, default=300, help="frames to read with --webcam")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="fovea-example-") as scratch:
        events = (
            webcam_events(args.max_frames)
            if args.webcam
            else replay_events(args.fixture, Path(scratch))
        )
        counts = summarize(events)
    print("events: " + ", ".join(f"{name}={count}" for name, count in sorted(counts.items())))
    return counts


if __name__ == "__main__":
    main()
