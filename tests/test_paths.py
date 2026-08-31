from pathlib import Path

import pytest

from fovea.paths import default_calibration_path, default_data_path, default_model_path
from fovea.webcam.engine import GazeEngine, GazeSettings
from fovea.webcam.event_source import WebcamEventSource
from fovea.webcam.landmarks import resolve_model_path


def test_environment_overrides_keep_runtime_files_out_of_checkout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "cache" / "face_landmarker.task"
    data_path = tmp_path / "data"
    monkeypatch.setenv("FOVEA_MODEL_PATH", str(model_path))
    monkeypatch.setenv("FOVEA_DATA_DIR", str(data_path))

    assert default_model_path() == model_path
    assert resolve_model_path(None) == model_path
    assert default_data_path() == data_path
    assert default_calibration_path() == data_path / "calibration" / "default.json"
    assert GazeEngine(GazeSettings()).path == default_calibration_path()


def test_calibration_path_must_be_absolute(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute or None"):
        GazeEngine(GazeSettings(calibration_path="relative.json"), tmp_path)


def test_project_root_is_accepted_with_deprecation_warning(tmp_path: Path) -> None:
    with pytest.warns(DeprecationWarning, match="project_root"):
        source = WebcamEventSource(
            GazeSettings(calibration_path=tmp_path / "calibration.json"),
            tmp_path,
        )
    source.close()
