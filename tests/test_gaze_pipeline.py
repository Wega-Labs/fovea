"""Tests for the webcam gaze pipeline (features → calibration → engine)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from fovea.webcam.calibration import (
    CalibrationIdentity,
    CalibrationModel,
    CalibrationTarget,
    fit_ridge,
    load_model,
    save_model,
    uncalibrated_map,
)
from fovea.webcam.engine import GazeEngine, GazeSettings
from fovea.webcam.features import (
    FEATURE_NAMES,
    EyeBox,
    GazeFeatures,
    extract_features,
)
from fovea.webcam.sampler import PointCollector
from fovea.webcam.smoothing import OneEuroPoint, ema


def lm(x: float, y: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y)


def _face_with_iris(
    left_nx: float = 0.5,
    left_ny: float = 0.5,
    right_nx: float = 0.5,
    right_ny: float = 0.5,
    face_width: float = 0.44,
) -> list[SimpleNamespace]:
    points = [lm(0.5, 0.5) for _ in range(478)]
    points[1] = lm(0.50, 0.52)
    points[10] = lm(0.50, 0.18)
    points[152] = lm(0.50, 0.90)
    points[234] = lm(0.5 - face_width / 2, 0.52)
    points[454] = lm(0.5 + face_width / 2, 0.52)
    points[291] = lm(0.58, 0.70)
    points[61] = lm(0.42, 0.70)

    points[33] = lm(0.30, 0.40)
    points[133] = lm(0.40, 0.40)
    for i in (159, 158, 157, 173, 160):
        points[i] = lm(0.35, 0.36)
    for i in (145, 144, 163, 153):
        points[i] = lm(0.35, 0.44)
    rx = 0.30 + right_nx * 0.10
    ry = 0.36 + right_ny * 0.08
    for i in (468, 469, 470, 471, 472):
        points[i] = lm(rx, ry)

    points[263] = lm(0.70, 0.40)
    points[362] = lm(0.60, 0.40)
    for i in (386, 385, 384, 398, 387):
        points[i] = lm(0.65, 0.36)
    for i in (374, 373, 380, 382):
        points[i] = lm(0.65, 0.44)
    lx = 0.60 + left_nx * 0.10
    ly = 0.36 + left_ny * 0.08
    for i in (473, 474, 475, 476, 477):
        points[i] = lm(lx, ly)
    return points


def _features(
    nx: float,
    ny: float,
    blend_x: float = 0.0,
    blend_y: float = 0.0,
    yaw: float = 0.0,
    pitch: float = 0.0,
) -> GazeFeatures:
    eye = EyeBox(0, 0, 1, 1, nx, ny, nx, ny, 0.35, True)
    return GazeFeatures(
        left=eye,
        right=eye,
        iris_nx=nx,
        iris_ny=ny,
        blend_x=blend_x,
        blend_y=blend_y,
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=0.0,
        face_width=0.30,
        blink=False,
        both_eyes=True,
        tracking="GOOD",
        message="",
    )


def test_normalized_iris_inside_eye_box() -> None:
    face = _face_with_iris(left_nx=0.25, left_ny=0.75, right_nx=0.25, right_ny=0.75)
    feats = extract_features(face, 640, 480, blink_ear=0.16, min_face_width=0.15, max_yaw_deg=40)
    assert feats.both_eyes
    assert 0.15 < feats.iris_nx < 0.35
    assert 0.65 < feats.iris_ny < 0.85


def test_face_distance_gate_uses_short_frame_side() -> None:
    landscape = extract_features(
        _face_with_iris(face_width=0.091),
        640,
        480,
        blink_ear=0.16,
        min_face_width=0.12,
        max_yaw_deg=40,
    )
    widescreen = extract_features(
        _face_with_iris(face_width=0.06825),
        1280,
        720,
        blink_ear=0.16,
        min_face_width=0.12,
        max_yaw_deg=40,
    )
    assert landscape.tracking == widescreen.tracking == "GOOD"


def test_poor_samples_advance_with_half_weight() -> None:
    collector = PointCollector(needed=2, min_good=2)
    vector = _features(0.5, 0.5).vector()
    collector.add(vector, "POOR", blink=False)
    collector.add(vector, "POOR", blink=False)
    assert collector.done()
    assert collector.count == 2
    assert collector.weighted_count == 1.0
    assert collector.quality() == "POOR"


def test_normal_desk_face_completes_calibration(tmp_path) -> None:
    identity = CalibrationIdentity(None, 1280, 720, 0, 640, 480)
    engine = GazeEngine(
        GazeSettings(
            calibration_path=str(tmp_path / "desk.json"),
            samples_per_point=1,
            min_good_samples=1,
            settle_frames=0,
        ),
        tmp_path,
        identity,
    )
    engine.start_calibration()
    face = _face_with_iris(face_width=0.15)
    for _ in range(10):
        engine.process(face, 640, 480, 1 / 30, 30.0)
    assert engine.wizard is None
    assert engine.model is not None
    assert load_model(tmp_path / "desk.json", expect=identity) is not None


def test_custom_five_target_calibration_completes_and_persists(tmp_path) -> None:
    identity = CalibrationIdentity("window", 1440, 900, 0, 640, 480)
    targets = (
        CalibrationTarget("top-left", 0.1, 0.1),
        CalibrationTarget("top-right", 0.9, 0.1),
        CalibrationTarget("center", 0.5, 0.5),
        CalibrationTarget("bottom-left", 0.1, 0.9),
        CalibrationTarget("bottom-right", 0.9, 0.9),
    )
    path = tmp_path / "custom.json"
    engine = GazeEngine(
        GazeSettings(
            calibration_path=str(path),
            samples_per_point=1,
            min_good_samples=1,
            settle_frames=0,
        ),
        tmp_path,
        identity,
    )
    engine.start_calibration(targets)
    face = _face_with_iris(face_width=0.15)
    for _target in targets:
        engine.process(face, 640, 480, 1 / 30, 30.0)

    assert engine.wizard is None
    assert engine.model is not None
    assert engine.last_calibration_report["n_points"] == 5
    assert engine.last_calibration_report["coverage"] == 0.8
    assert np.isfinite(engine.last_calibration_report["loo_error"])
    loaded = load_model(path, expect=identity)
    assert loaded is not None
    assert loaded.targets == targets


def test_calibration_rejects_fewer_than_five_targets(tmp_path) -> None:
    engine = GazeEngine(GazeSettings(calibration_path=str(tmp_path / "c.json")), tmp_path)
    targets = tuple(CalibrationTarget(str(index), 0.5, 0.5) for index in range(3))
    with np.testing.assert_raises_regex(ValueError, "at least 5"):
        engine.start_calibration(targets)


def test_low_coverage_calibration_queues_warning(tmp_path) -> None:
    from fovea.events import CalibrationWarning
    from fovea.webcam.event_source import WebcamEventSource

    targets = tuple(
        CalibrationTarget(str(index), 0.40 + index * 0.02, 0.45 + index * 0.01)
        for index in range(5)
    )
    settings = GazeSettings(calibration_path=str(tmp_path / "c.json"))
    source = WebcamEventSource(settings, tmp_path, show_calibration=False)
    source._engine = GazeEngine(settings, tmp_path)
    source.start_calibration(targets)

    warning = source._pending_events.popleft()
    assert isinstance(warning, CalibrationWarning)
    assert warning.coverage < 0.4


def test_feature_vector_avoids_multicollinearity() -> None:
    rows = []
    for nx, ny, dnx, dny, bx, by, yaw, pitch, roll in [
        (0.50, 0.50, 0.00, 0.00, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.35, 0.40, 0.04, -0.02, -0.3, -0.1, 5.0, -2.0, 1.0),
        (0.65, 0.40, -0.03, 0.01, 0.3, -0.1, -4.0, -1.0, -1.0),
        (0.40, 0.65, 0.02, 0.05, -0.2, 0.4, 2.0, 6.0, 0.5),
        (0.60, 0.65, -0.02, -0.04, 0.2, 0.4, -3.0, 5.0, -0.5),
        (0.50, 0.70, 0.01, 0.03, 0.0, 0.5, 1.0, 8.0, 0.2),
    ]:
        left = EyeBox(0, 0, 1, 1, nx + dnx, ny + dny, nx + dnx, ny + dny, 0.35, True)
        right = EyeBox(0, 0, 1, 1, nx - dnx, ny - dny, nx - dnx, ny - dny, 0.35, True)
        rows.append(
            GazeFeatures(
                left, right, nx, ny, bx, by, yaw, pitch, roll, 0.3, False, True, "GOOD", ""
            ).vector()
        )
    mat = np.vstack(rows)
    assert mat.shape[1] == len(FEATURE_NAMES)
    iris_block = mat[:, [1, 2, 3, 4]]
    assert np.linalg.matrix_rank(iris_block) >= 3


def test_ridge_calibration_maps_corners() -> None:
    rows = []
    xy = []
    layout = [
        (0.5, 0.5, 0.5, 0.5, 0.0, 0.0),
        (0.12, 0.12, 0.32, 0.32, -0.4, -0.3),
        (0.50, 0.12, 0.50, 0.32, 0.0, -0.3),
        (0.88, 0.12, 0.68, 0.32, 0.4, -0.3),
        (0.12, 0.50, 0.32, 0.50, -0.4, 0.0),
        (0.50, 0.50, 0.50, 0.50, 0.0, 0.0),
        (0.88, 0.50, 0.68, 0.50, 0.4, 0.0),
        (0.12, 0.88, 0.32, 0.68, -0.4, 0.5),
        (0.50, 0.88, 0.50, 0.68, 0.0, 0.5),
        (0.88, 0.88, 0.68, 0.68, 0.4, 0.5),
    ]
    for sx, sy, nx, ny, bx, by in layout:
        rows.append(_features(nx, ny, bx, by).vector())
        xy.append((sx, sy))
    model = fit_ridge(rows, xy, {"n": len(rows)}, {"n": "GOOD"})
    assert max(abs(c) for c in model.coef_x) < 50
    assert max(abs(c) for c in model.coef_y) < 50
    pred = model.predict(_features(0.68, 0.68, 0.4, 0.5))
    assert pred[0] > 0.7
    assert pred[1] > 0.7
    pred_c = model.predict(_features(0.5, 0.5, 0.0, 0.0))
    assert abs(pred_c[0] - 0.5) < 0.12
    assert abs(pred_c[1] - 0.5) < 0.12


def test_v1_calibration_is_rejected(tmp_path) -> None:
    path = tmp_path / "old.json"
    path.write_text(
        '{"version":1,"coef_x":[1,0,0,0,0,0,0,0,0,0],'
        '"coef_y":[1,0,0,0,0,0,0,0,0,0],"feature_names":[]}',
        encoding="utf-8",
    )
    assert load_model(path) is None


def test_v2_calibration_is_valid_but_unlabeled(tmp_path) -> None:
    path = tmp_path / "v2.json"
    path.write_text(
        '{"version":2,"coef_x":[1],"coef_y":[1],"feature_names":[],"samples":{},'
        '"quality":{},"created":""}',
        encoding="utf-8",
    )
    model = load_model(path)
    assert model is not None
    assert model.identity is None


def test_calibration_identity_roundtrip_and_mismatch(tmp_path) -> None:
    identity = CalibrationIdentity("display-1", 1920, 1080, 0, 640, 480)
    other_display = CalibrationIdentity("display-2", 1920, 1080, 0, 640, 480)
    rows = [
        _features(0.5, 0.5).vector(),
        _features(0.3, 0.3).vector(),
        _features(0.7, 0.7).vector(),
    ]
    model = fit_ridge(
        rows,
        [(0.5, 0.5), (0.2, 0.2), (0.8, 0.8)],
        {"n": 3},
        {"n": "GOOD"},
        identity=identity,
    )
    path = tmp_path / "identified.json"
    save_model(model, path)
    loaded = load_model(path, expect=identity)
    assert loaded is not None
    assert loaded.identity == identity
    assert load_model(path, expect=other_display) is None


def test_uncalibrated_look_down_moves_toward_bottom() -> None:
    _sx, sy = uncalibrated_map(_features(0.5, 0.55, blend_y=0.55))
    assert sy > 0.65


def test_ema_and_one_euro_smoothing() -> None:
    assert abs(ema(0.0, 1.0, 0.25) - 0.25) < 1e-9
    filt = OneEuroPoint(1.2, 0.02)
    x, y = 0.5, 0.5
    for _ in range(5):
        x, y = filt.filter(0.52, 0.48, 1 / 30)
    assert abs(x - 0.5) < 0.05
    assert abs(y - 0.5) < 0.05


def test_blink_frames_do_not_update_engine_screen(tmp_path) -> None:
    engine = GazeEngine(
        GazeSettings(calibration_path=str(tmp_path / "c.json"), smoothing_alpha=1.0),
        tmp_path,
    )
    face = _face_with_iris(0.7, 0.7, 0.7, 0.7)
    out = engine.process(face, 640, 480, 1 / 30, 30.0, blendshapes={"eyelookdownleft": 0.5})
    assert out.valid
    frozen = out.screen
    closed = _face_with_iris()
    for i in (159, 158, 157, 173, 160, 386, 385, 384, 398, 387):
        closed[i] = lm(closed[i].x, 0.395)
    for i in (145, 144, 163, 153, 374, 373, 380, 382):
        closed[i] = lm(closed[i].x, 0.405)
    lost = engine.process(closed, 640, 480, 1 / 30, 30.0)
    assert not lost.valid
    assert lost.frozen
    assert lost.screen == frozen


def test_calibration_roundtrip(tmp_path) -> None:
    rows = [
        _features(0.5, 0.5).vector(),
        _features(0.3, 0.3, -0.3, -0.2).vector(),
        _features(0.7, 0.7, 0.3, 0.4).vector(),
    ]
    xy = [(0.5, 0.5), (0.2, 0.2), (0.8, 0.8)]
    model = fit_ridge(rows, xy, {"a": 3}, {"a": "GOOD"})
    path = tmp_path / "gaze_calibration.json"
    save_model(model, path)
    loaded = load_model(path)
    assert loaded is not None
    assert isinstance(loaded, CalibrationModel)
    assert loaded.version >= 2
    p1 = model.predict(_features(0.5, 0.5))
    p2 = loaded.predict(_features(0.5, 0.5))
    assert abs(p1[0] - p2[0]) < 1e-9


def test_webcam_event_source_yields_typed_events(monkeypatch, tmp_path) -> None:
    from fovea.events import Diagnostics, GazePoint, TrackingState, TrackingStatus
    from fovea.webcam.calibration import fit_ridge, save_model
    from fovea.webcam.event_source import WebcamEventSource

    face = _face_with_iris(face_width=0.08)
    cal_path = tmp_path / "gaze_calibration.json"
    rows = [
        _features(0.5, 0.5).vector(),
        _features(0.3, 0.3).vector(),
        _features(0.7, 0.7).vector(),
    ]
    model = fit_ridge(
        rows,
        [(0.5, 0.5), (0.2, 0.2), (0.8, 0.8)],
        {"n": 3},
        {"n": "GOOD"},
        identity=CalibrationIdentity(None, 1280, 720, 0, 640, 480),
    )
    save_model(model, cal_path)

    class FakeCamera:
        calls = 0

        def __init__(self, *_args, **_kwargs) -> None:
            return None

        def connect(self) -> None:
            return None

        def read(self):
            if FakeCamera.calls == 0:
                FakeCamera.calls += 1
                return np.zeros((480, 640, 3), dtype=np.uint8)
            return None

        def disconnect(self) -> None:
            return None

    class FakeEstimator:
        def __init__(self, **_kwargs) -> None:
            return None

        def process(self, _frame):
            return type("Obs", (), {"landmarks": face, "blendshapes": {}})()

        def close(self) -> None:
            return None

    monkeypatch.setattr("fovea.webcam.event_source.Webcam", FakeCamera)
    monkeypatch.setattr("fovea.webcam.event_source.FaceLandmarkEstimator", FakeEstimator)

    source = WebcamEventSource(
        GazeSettings(calibration_path=str(cal_path), smoothing_alpha=1.0),
        tmp_path,
        max_frames=2,
        force_calibrate=False,
        show_calibration=False,
        diagnostics=True,
    )
    events = list(source.events())
    assert any(isinstance(event, GazePoint) for event in events)
    assert any(isinstance(event, TrackingState) for event in events)
    diagnostics = [event for event in events if isinstance(event, Diagnostics)]
    assert len(diagnostics) == 1
    assert diagnostics[0].face_width > 0.0
    poor_states = [
        event
        for event in events
        if isinstance(event, TrackingState) and event.status is TrackingStatus.UNCERTAIN
    ]
    assert poor_states
    assert poor_states[0].detail == "Face too far from camera"


def test_diagnostics_rate_limit_is_two_hz() -> None:
    from fovea.webcam.event_source import _diagnostics_due

    assert _diagnostics_due(None, 10.0)
    assert not _diagnostics_due(10.0, 10.499)
    assert _diagnostics_due(10.0, 10.5)


def test_gaze_test_report_converts_to_typed_event() -> None:
    from fovea.webcam.event_source import _gaze_test_event

    report: dict[str, object] = {
        "n": 1,
        "mean_error": 0.05,
        "median_error": 0.05,
        "max_error": 0.05,
        "points": [
            {
                "expected": [0.5, 0.5],
                "predicted": [0.54, 0.53],
                "error": 0.05,
            }
        ],
    }
    event = _gaze_test_event(report, 123)
    assert event is not None
    assert event.n_points == 1
    assert event.points[0].expected_x == 0.5
    assert event.points[0].predicted_y == 0.53


def test_calibration_emits_layout_cue(monkeypatch, tmp_path) -> None:
    from fovea.events import CalibrationCue
    from fovea.webcam.calibration import CALIBRATION_LAYOUT
    from fovea.webcam.event_source import WebcamEventSource

    class FakeCamera:
        def __init__(self, *_args, **_kwargs) -> None:
            return None

        def connect(self) -> None:
            return None

        def read(self):
            return np.zeros((480, 640, 3), dtype=np.uint8)

        def disconnect(self) -> None:
            return None

    class FakeEstimator:
        def __init__(self, **_kwargs) -> None:
            return None

        def process(self, _frame):
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr("fovea.webcam.event_source.Webcam", FakeCamera)
    monkeypatch.setattr("fovea.webcam.event_source.FaceLandmarkEstimator", FakeEstimator)

    source = WebcamEventSource(
        GazeSettings(calibration_path=str(tmp_path / "missing.json")),
        tmp_path,
        max_frames=1,
        force_calibrate=True,
        show_calibration=False,
    )
    cues = [event for event in source.events() if isinstance(event, CalibrationCue)]
    assert cues
    first = CALIBRATION_LAYOUT[0]
    assert cues[0].label == first.label
    assert cues[0].x == first.x
    assert cues[0].y == first.y
    assert cues[0].total == len(CALIBRATION_LAYOUT)
