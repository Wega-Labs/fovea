"""Online self-calibration persistence, quarantine, and transaction tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fovea.events import CalibrationUpdated, GazePoint
from fovea.webcam.calibration import (
    CALIBRATION_LAYOUT,
    CALIBRATION_VERSION,
    IDENTITY_VERSION,
    CalibrationIdentity,
    CalibrationModel,
    fit_ridge,
    load_model,
    online_refit,
    save_model,
    weighted_leave_one_out_error,
)
from fovea.webcam.engine import ADMISSION_CAP, GazeEngine, GazeSettings
from fovea.webcam.features import EyeBox, GazeFeatures, extract_features
from fovea.webcam.fixtures import (
    LandmarkFrame,
    RecordedLandmark,
    ReplayEventSource,
    frame_to_json,
)
from fovea.webcam.frame_processor import drain_online_events
from tests.synth import synthetic_landmarks


def _features(nx: float, ny: float, *, blink: bool = False) -> GazeFeatures:
    eye = EyeBox(0.0, 0.0, 1.0, 1.0, nx, ny, nx, ny, 0.35, True)
    return GazeFeatures(
        left=eye,
        right=eye,
        iris_nx=nx,
        iris_ny=ny,
        blend_x=(nx - 0.5) * 0.4,
        blend_y=(ny - 0.5) * 0.4,
        yaw_deg=(nx - 0.5) * 8.0,
        pitch_deg=(ny - 0.5) * 8.0,
        roll_deg=0.0,
        face_width=0.3,
        blink=blink,
        both_eyes=True,
        tracking="GOOD",
        message="",
    )


def _calibration_model() -> CalibrationModel:
    rows = [_features(target.x, target.y).vector() for target in CALIBRATION_LAYOUT]
    xy = [(target.x, target.y) for target in CALIBRATION_LAYOUT]
    return fit_ridge(
        rows,
        xy,
        {str(index): 20 for index in range(len(rows))},
        {str(index): "GOOD" for index in range(len(rows))},
        identity=CalibrationIdentity(None, 1280, 720, 0, 640, 480),
        targets=CALIBRATION_LAYOUT,
    )


def _engine(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    online: bool = True,
    clock: object | None = None,
) -> GazeEngine:
    save_model(_calibration_model(), path)
    feature = _features(0.62, 0.58)
    monkeypatch.setattr("fovea.webcam.engine.extract_features", lambda *_args, **_kwargs: feature)
    kwargs = {} if clock is None else {"clock": clock}
    return GazeEngine(
        GazeSettings(
            calibration_path=str(path),
            smoothing_alpha=1.0,
            online_calibration=online,
        ),
        **kwargs,
    )


def _admit(
    engine: GazeEngine,
    timestamp_ns: int,
    dx: float,
    dy: float,
    *,
    target: tuple[float, float] | None = None,
) -> None:
    output = engine.process([object()], 640.0, 480.0, 1 / 30, 30.0, timestamp_ns=timestamp_ns)
    assert output.features is not None
    predicted = engine.model.predict(output.features) if engine.model is not None else (0.5, 0.5)
    observed = target or (predicted[0] + dx, predicted[1] + dy)
    engine.observe(observed[0], observed[1], timestamp_ns=timestamp_ns)


def test_online_refit_and_weighted_leave_one_out_are_finite() -> None:
    model = _calibration_model()
    rows = [np.asarray(anchor.row) for anchor in model.anchors]
    xy = [anchor.xy for anchor in model.anchors]
    weights = [1.0] * len(rows)
    coef_x, coef_y = online_refit(rows, xy, weights, model.ridge)
    assert np.all(np.isfinite(coef_x))
    assert np.all(np.isfinite(coef_y))
    assert weighted_leave_one_out_error(rows, xy, weights, model.ridge) >= 0.0


def test_v4_roundtrip_persists_anchors_and_online_state(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    model = _calibration_model()
    save_model(model, path)
    loaded = load_model(path)
    assert loaded == model
    assert loaded is not None
    assert loaded.version == CALIBRATION_VERSION
    assert len(loaded.anchors) == len(CALIBRATION_LAYOUT)


def test_v3_keeps_identity_but_has_no_online_anchors(tmp_path: Path) -> None:
    model = replace(
        _calibration_model(),
        version=IDENTITY_VERSION,
        anchors=(),
        baseline_anchor_error=None,
    )
    path = tmp_path / "v3.json"
    save_model(model, path)
    loaded = load_model(path)
    assert loaded is not None
    assert loaded.identity == model.identity
    assert loaded.anchors == ()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("baseline_anchor_error", float("nan")),
        ("n", True),
        ("commit_seq", -1),
    ],
)
def test_v4_rejects_invalid_online_scalars(tmp_path: Path, field: str, value: object) -> None:
    path = tmp_path / "bad.json"
    data = _calibration_model().to_dict()
    data[field] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_model(path) is None


def test_v4_rejects_bad_vector_weight_seq_and_anchor_target(tmp_path: Path) -> None:
    valid = _calibration_model().to_dict()
    variants: list[dict[str, object]] = []
    bad_row = json.loads(json.dumps(valid))
    bad_row["anchors"][0]["row"] = [1.0]
    variants.append(bad_row)
    bad_target = json.loads(json.dumps(valid))
    bad_target["anchors"][0]["xy"] = [0.1, 0.1]
    variants.append(bad_target)
    bad_weight = json.loads(json.dumps(valid))
    bad_weight.update(
        {
            "observations": [
                {
                    "row": bad_weight["anchors"][0]["row"],
                    "xy": [0.5, 0.5],
                    "host_weight": 0.0,
                    "commit_seq": 1,
                }
            ],
            "n": 1,
            "commit_seq": 1,
        }
    )
    variants.append(bad_weight)
    bad_seq = json.loads(json.dumps(valid))
    bad_seq.update(
        {
            "observations": [
                {
                    "row": bad_seq["anchors"][0]["row"],
                    "xy": [0.5, 0.5],
                    "host_weight": 1.0,
                    "commit_seq": True,
                }
            ],
            "n": 1,
            "commit_seq": 1,
        }
    )
    variants.append(bad_seq)
    for index, data in enumerate(variants):
        path = tmp_path / f"bad-{index}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        assert load_model(path) is None


def test_atomic_save_failure_preserves_old_file_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "calibration.json"
    old_model = _calibration_model()
    save_model(old_model, path)
    before = path.read_bytes()

    def fail_replace(*_args: object) -> None:
        raise OSError

    monkeypatch.setattr("fovea.webcam.calibration.os.replace", fail_replace)
    with pytest.raises(OSError):
        save_model(
            replace(old_model, coef_x=tuple(value + 0.01 for value in old_model.coef_x)), path
        )
    assert path.read_bytes() == before
    assert load_model(path) == old_model
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob(".*.tmp")) == []


def test_two_refits_queue_ordered_fake_clock_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticks = iter((101, 202))
    engine = _engine(tmp_path / "calibration.json", monkeypatch, clock=ticks.__next__)
    original = engine.model
    assert original is not None
    first = _features(0.38, 0.42)
    second = _features(0.72, 0.66)
    for batch, feature in enumerate((first, second)):
        target = tuple(coordinate + 0.045 for coordinate in original.predict(feature))
        monkeypatch.setattr(
            "fovea.webcam.engine.extract_features", lambda *_args, value=feature, **_kwargs: value
        )
        for index in range(5):
            timestamp = 1_000_000_000 + (batch * 5 + index) * 10_000_000
            _admit(engine, timestamp, 0.0, 0.0, target=target)
    reports = drain_online_events(engine)
    assert [report.n for report in reports] == [5, 10]
    assert [report.timestamp_ns for report in reports] == [101, 202]
    assert all(np.isfinite(report.loo_error) for report in reports)


def test_robust_cluster_excludes_sub_cap_outlier_without_changing_clean_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean = _engine(tmp_path / "clean.json", monkeypatch)
    for index in range(5):
        _admit(clean, 1_000_000_000 + index * 10_000_000, 0.08, 0.06)
    clean_model = clean.model
    clean_report = clean.drain_online_reports()

    contaminated = _engine(tmp_path / "contaminated.json", monkeypatch)
    _admit(contaminated, 1_000_000_000, 0.08, 0.06)
    _admit(contaminated, 1_010_000_000, 0.08, 0.06)
    _admit(contaminated, 1_020_000_000, -0.49, 0.0)
    for index in range(2, 5):
        _admit(contaminated, 1_010_000_000 + index * 10_000_000, 0.08, 0.06)
    contaminated_model = contaminated.model
    contaminated_report = contaminated.drain_online_reports()

    assert clean_model is not None and contaminated_model is not None
    assert len(contaminated_model.observations) == 5
    assert all(observation.xy[0] > 0.2 for observation in contaminated_model.observations)
    assert np.allclose(contaminated_model.coef_x, clean_model.coef_x)
    assert np.allclose(contaminated_model.coef_y, clean_model.coef_y)
    assert contaminated_report[0].loo_error == pytest.approx(clean_report[0].loo_error)


def test_gross_cap_and_near_zero_do_not_promote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path / "calibration.json", monkeypatch)
    output = engine.process([object()], 640.0, 480.0, 1 / 30, 30.0, timestamp_ns=1_000_000_000)
    assert engine.model is not None and output.features is not None
    predicted = engine.model.predict(output.features)
    farthest = max(
        ((0.0, 0.0), (1.0, 1.0)), key=lambda point: np.hypot(*(np.subtract(point, predicted)))
    )
    assert np.hypot(*(np.subtract(farthest, predicted))) > ADMISSION_CAP
    engine.observe(*farthest, timestamp_ns=1_000_000_000)
    assert engine._admission_epoch == 0
    for index in range(3):
        _admit(engine, 2_000_000_000 + index, 0.0, 0.0)
    assert engine._promoted == []
    assert engine.drain_online_reports() == ()


def test_coherent_wide_residual_group_keeps_every_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path / "wide.json", monkeypatch)
    residuals = ((0.04, 0.07), (0.06, 0.10), (0.09, 0.05), (0.11, 0.08), (0.07, 0.04))
    for index, (dx, dy) in enumerate(residuals):
        _admit(engine, 1_000_000_000 + index, dx, dy)
    assert engine.model is not None
    assert len(engine.model.observations) == len(residuals)
    assert engine.model.n == len(residuals)
    assert engine._last_screen is None
    expected = engine.model.predict(_features(0.62, 0.58))
    first = engine.process([object()], 640.0, 480.0, 1 / 30, 30.0, timestamp_ns=2_000_000_000)
    assert first.screen is not None
    assert (first.screen.x, first.screen.y) == pytest.approx(expected)


def test_transaction_rollback_preserves_model_file_and_counters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "calibration.json"
    engine = _engine(path, monkeypatch)
    original = engine.model
    before = path.read_bytes()

    def harmful_refit(*_args: object, **_kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        return np.full(10, 100.0), np.full(10, -100.0)

    monkeypatch.setattr("fovea.webcam.engine.online_refit", harmful_refit)
    for index in range(5):
        _admit(engine, 1_000_000_000 + index, 0.08, 0.06)

    assert engine.model == original
    assert engine._commit_seq == 0
    assert engine._online_n == 0
    assert engine._promoted_count == 0
    assert engine.drain_online_reports() == ()
    assert path.read_bytes() == before


def test_stale_disabled_and_pre_v4_observations_are_noops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    disabled = _engine(tmp_path / "disabled.json", monkeypatch, online=False)
    _admit(disabled, 1_000_000_000, 0.08, 0.06)
    assert disabled._admission_epoch == 0
    assert disabled._trusted == []

    path = tmp_path / "v3.json"
    v3 = replace(
        _calibration_model(),
        version=IDENTITY_VERSION,
        anchors=(),
        baseline_anchor_error=None,
    )
    save_model(v3, path)
    before = path.read_bytes()
    engine = GazeEngine(GazeSettings(calibration_path=str(path)))
    for index in range(10):
        timestamp = 1_000_000_000 + index
        engine.process([object()], 640.0, 480.0, 1 / 30, 30.0, timestamp_ns=timestamp)
        engine.observe(0.5, 0.5, timestamp_ns=timestamp)
    assert engine.drain_online_reports() == ()
    assert engine.model == v3
    assert path.read_bytes() == before

    active = _engine(tmp_path / "stale.json", monkeypatch)
    active.process([object()], 640.0, 480.0, 1 / 30, 30.0, timestamp_ns=1_000_000_000)
    active.observe(0.5, 0.5, timestamp_ns=1_300_000_001)
    assert active._admission_epoch == 0


def test_blink_lost_and_wizard_paths_do_not_associate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path / "calibration.json", monkeypatch)
    monkeypatch.setattr(
        "fovea.webcam.engine.extract_features",
        lambda *_args, **_kwargs: _features(0.62, 0.58, blink=True),
    )
    engine.process([object()], 640.0, 480.0, 1 / 30, 30.0, timestamp_ns=1)
    engine.observe(0.6, 0.6, timestamp_ns=1)
    assert engine._admission_epoch == 0

    monkeypatch.setattr(
        "fovea.webcam.engine.extract_features",
        lambda *_args, **_kwargs: _features(0.62, 0.58),
    )
    engine.process([object()], 640.0, 480.0, 1 / 30, 30.0, timestamp_ns=2)
    engine.process(None, 640.0, 480.0, 1 / 30, 30.0, timestamp_ns=3)
    engine.observe(0.6, 0.6, timestamp_ns=2)
    assert engine._admission_epoch == 0

    engine.process([object()], 640.0, 480.0, 1 / 30, 30.0, timestamp_ns=4)
    engine.start_gaze_test()
    engine.observe(0.6, 0.6, timestamp_ns=4)
    assert engine._admission_epoch == 0
    engine.start_calibration()
    assert engine.model is None
    engine.observe(0.6, 0.6, timestamp_ns=4)
    assert engine._admission_epoch == 0


def test_replay_observations_converge_and_persist_loaded_v4_model(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.json"
    fixture_path = tmp_path / "offset.jsonl"

    def offset_features(target_x: float, target_y: float) -> tuple[GazeFeatures, list[object]]:
        points = synthetic_landmarks(
            gaze_dx=(target_x - 0.5) * 0.45 + 0.025,
            gaze_dy=(target_y - 0.5) * 0.45 - 0.018,
        )
        features = extract_features(points, 640.0, 480.0, 0.16, 0.12, 35.0)
        return features, list(points)

    anchor_rows = []
    anchor_xy = []
    for target in CALIBRATION_LAYOUT:
        points = synthetic_landmarks(
            gaze_dx=(target.x - 0.5) * 0.45,
            gaze_dy=(target.y - 0.5) * 0.45,
        )
        anchor_rows.append(extract_features(points, 640.0, 480.0, 0.16, 0.12, 35.0).vector())
        anchor_xy.append((target.x, target.y))
    base = fit_ridge(
        anchor_rows,
        anchor_xy,
        {str(index): 20 for index in range(len(anchor_rows))},
        {str(index): "GOOD" for index in range(len(anchor_rows))},
        identity=CalibrationIdentity(None, 1280, 720, 0, 640, 480),
        targets=CALIBRATION_LAYOUT,
    )
    save_model(base, calibration_path)
    selected = tuple(CALIBRATION_LAYOUT[index] for index in (1, 3, 7, 9, 0))
    expected_by_timestamp: dict[int, tuple[float, float]] = {}
    frames: list[LandmarkFrame] = []
    for index in range(20):
        target = selected[index % len(selected)]
        _features_at_target, points = offset_features(target.x, target.y)
        timestamp = 1_000_000_000 + index * 33_000_000
        expected_by_timestamp[timestamp] = (target.x, target.y)
        frames.append(
            LandmarkFrame(
                timestamp,
                640,
                480,
                tuple(RecordedLandmark(point.x, point.y, point.z) for point in points),
                {},
            )
        )
    fixture_path.write_text(
        "".join(f"{frame_to_json(frame)}\n" for frame in frames),
        encoding="utf-8",
    )

    def mean_error(model: CalibrationModel) -> float:
        errors = []
        for target in selected:
            feature, _points = offset_features(target.x, target.y)
            predicted = model.predict(feature)
            errors.append(float(np.hypot(predicted[0] - target.x, predicted[1] - target.y)))
        return float(np.mean(errors))

    before = mean_error(base)
    source = ReplayEventSource(
        fixture_path,
        GazeSettings(calibration_path=str(calibration_path), smoothing_alpha=1.0),
        tmp_path,
    )
    updates: list[CalibrationUpdated] = []
    for event in source.events():
        if isinstance(event, GazePoint):
            source.observe(
                *expected_by_timestamp[event.timestamp_ns],
                timestamp_ns=event.timestamp_ns,
            )
        elif isinstance(event, CalibrationUpdated):
            updates.append(event)

    loaded = load_model(calibration_path)
    assert loaded is not None
    after = mean_error(loaded)
    assert loaded.version == CALIBRATION_VERSION
    assert loaded.n <= 20
    assert updates
    assert after < 0.06
    assert after <= before * 0.6
