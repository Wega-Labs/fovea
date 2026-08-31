"""Platform-native storage paths with explicit environment overrides."""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_cache_dir, user_data_dir

APP_NAME = "fovea"


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value).expanduser()


def default_model_path() -> Path:
    """Return the model file override or Fovea's platform cache location."""
    override = _environment_path("FOVEA_MODEL_PATH")
    if override is not None:
        return override
    return Path(user_cache_dir(APP_NAME)) / "models" / "face_landmarker.task"


def default_data_path() -> Path:
    """Return the persistent application-data root."""
    override = _environment_path("FOVEA_DATA_DIR")
    if override is not None:
        return override
    return Path(user_data_dir(APP_NAME))


def default_calibration_path(key: str = "default") -> Path:
    """Return the platform data path for one named calibration."""
    if not key or Path(key).name != key:
        raise ValueError("calibration key must be a single non-empty path component")
    return default_data_path() / "calibration" / f"{key}.json"
