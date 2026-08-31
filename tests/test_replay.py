"""Privacy and determinism tests for landmark fixtures and replay."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from fovea.cli import main
from fovea.events import GazePoint, TrackingState
from fovea.serialize import to_json
from fovea.webcam.calibration import CALIBRATION_LAYOUT, load_model
from fovea.webcam.engine import GazeSettings
from fovea.webcam.features import extract_features
from fovea.webcam.fixtures import (
    LandmarkFrame,
    RecordedLandmark,
    ReplayEventSource,
    frame_to_json,
    read_landmark_frames,
    record_landmarks,
)
from tests.synth import SyntheticLandmark, synthetic_landmarks

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic" / "frontal_30f.jsonl"


def _frame(points: list[SyntheticLandmark], index: int) -> LandmarkFrame:
    return LandmarkFrame(
        ts_ns=1_000_000_000 + index * 33_333_333,
        w=640,
        h=480,
        landmarks=tuple(RecordedLandmark(point.x, point.y, point.z) for point in points),
        blendshapes={},
    )


def _write_fixture(path: Path, frames: list[LandmarkFrame]) -> None:
    path.write_text("".join(f"{frame_to_json(frame)}\n" for frame in frames), encoding="utf-8")


def _replay(tmp_path: Path) -> list[str]:
    source = ReplayEventSource(
        path=FIXTURE,
        settings=GazeSettings(calibration_path=str(tmp_path / "missing.json")),
        project_root=tmp_path,
    )
    return [to_json(event) for event in source.events()]


def test_synthetic_fixture_is_small_and_contains_no_pixels() -> None:
    assert FIXTURE.stat().st_size <= 200 * 1024
    frames = list(read_landmark_frames(FIXTURE))
    assert len(frames) == 30
    assert all(len(frame.landmarks) == 478 for frame in frames)
    raw = FIXTURE.read_text(encoding="utf-8")
    assert '"pixels"' not in raw
    assert '"image"' not in raw


def test_fixture_features_have_frontal_pose_and_centered_iris() -> None:
    frame = next(read_landmark_frames(FIXTURE))
    features = extract_features(
        frame.landmarks,
        float(frame.w),
        float(frame.h),
        blink_ear=0.16,
        min_face_width=0.12,
        max_yaw_deg=35.0,
        blendshapes=frame.blendshapes,
    )
    assert features.tracking == "GOOD"
    assert abs(features.pitch_deg) < 2.0
    assert abs(features.yaw_deg) < 2.0
    assert abs(features.iris_nx - 0.5) < 0.01
    assert abs(features.iris_ny - 0.5) < 0.01


def test_replay_is_byte_deterministic_and_emits_gaze(tmp_path) -> None:
    first = _replay(tmp_path / "first")
    second = _replay(tmp_path / "second")
    assert first == second
    decoded = [json.loads(line) for line in first]
    assert any(item["type"] == "tracking_state" for item in decoded)
    assert any(item["type"] == "gaze_point" for item in decoded)


def test_replay_cli_is_byte_deterministic(monkeypatch, capsys) -> None:
    outputs: list[str] = []
    for _ in range(2):
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        assert main(["replay", str(FIXTURE), "--ndjson"]) == 0
        outputs.append(capsys.readouterr().out)
    assert outputs[0] == outputs[1]


def test_replay_drives_calibration_fit(tmp_path) -> None:
    fixture = tmp_path / "calibration.jsonl"
    frames = [
        _frame(
            synthetic_landmarks(
                gaze_dx=(target.x - 0.5) * 0.45,
                gaze_dy=(target.y - 0.5) * 0.45,
            ),
            index,
        )
        for index, target in enumerate(CALIBRATION_LAYOUT)
    ]
    _write_fixture(fixture, frames)
    calibration_path = tmp_path / "calibration.json"
    source = ReplayEventSource(
        path=fixture,
        settings=GazeSettings(
            calibration_path=str(calibration_path),
            samples_per_point=1,
            min_good_samples=1,
            settle_frames=0,
        ),
        project_root=tmp_path,
        force_calibrate=True,
    )
    list(source.events())
    model = load_model(calibration_path)
    assert model is not None
    assert len(model.samples) == len(CALIBRATION_LAYOUT)


def test_replay_preserves_smoothing_regression(tmp_path) -> None:
    fixture = tmp_path / "step.jsonl"
    _write_fixture(
        fixture,
        [
            _frame(synthetic_landmarks(gaze_dx=-0.15), 0),
            _frame(synthetic_landmarks(gaze_dx=0.15), 1),
        ],
    )

    def gaze_x(alpha: float, name: str) -> list[float]:
        source = ReplayEventSource(
            path=fixture,
            settings=GazeSettings(
                calibration_path=str(tmp_path / f"missing-{name}.json"),
                smoothing_alpha=alpha,
            ),
            project_root=tmp_path,
        )
        return [event.x for event in source.events() if isinstance(event, GazePoint)]

    unsmoothed = gaze_x(1.0, "raw")
    smoothed = gaze_x(0.25, "smooth")
    assert len(unsmoothed) == len(smoothed) == 2
    assert abs(smoothed[1] - smoothed[0]) < abs(unsmoothed[1] - unsmoothed[0])


def test_record_writes_landmarks_without_pixels(monkeypatch, tmp_path) -> None:
    class FakeCamera:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def connect(self) -> None:
            pass

        def read(self) -> np.ndarray:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        def disconnect(self) -> None:
            pass

    class FakeEstimator:
        def __init__(self, **_kwargs) -> None:
            pass

        def process(self, _pixels: np.ndarray) -> object:
            return type(
                "Observation",
                (),
                {"landmarks": synthetic_landmarks(), "blendshapes": {}},
            )()

        def close(self) -> None:
            pass

    monkeypatch.setattr("fovea.webcam.fixtures.Webcam", FakeCamera)
    monkeypatch.setattr("fovea.webcam.fixtures.FaceLandmarkEstimator", FakeEstimator)
    path = tmp_path / "recording.jsonl"
    assert record_landmarks(path, seconds=1.0, max_frames=2) == 2
    frames = list(read_landmark_frames(path))
    assert len(frames) == 2
    assert all(len(frame.landmarks) == 478 for frame in frames)
    assert '"pixels"' not in path.read_text(encoding="utf-8")


def test_synthetic_generator_controls_pose_gaze_and_blink() -> None:
    points = synthetic_landmarks(pitch=15.0, yaw=-20.0, gaze_dx=0.2, gaze_dy=-0.1)
    features = extract_features(points, 640, 480, 0.16, 0.12, 35.0)
    assert abs(features.pitch_deg - 15.0) < 2.0
    assert abs(features.yaw_deg + 20.0) < 2.0
    assert features.iris_nx > 0.65
    assert features.iris_ny < 0.45
    blink = extract_features(synthetic_landmarks(blink=True), 640, 480, 0.16, 0.12, 35.0)
    assert blink.blink


def test_replay_event_types_are_public_contract(tmp_path) -> None:
    source = ReplayEventSource(
        path=FIXTURE,
        settings=GazeSettings(calibration_path=str(tmp_path / "missing.json")),
        project_root=tmp_path,
        max_frames=1,
    )
    events = list(source.events())
    assert any(isinstance(event, TrackingState) for event in events)
    assert any(isinstance(event, GazePoint) for event in events)
