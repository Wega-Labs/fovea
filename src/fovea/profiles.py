"""Named calibration profiles, auto-selected by display and camera identity.

A profile is a named set of calibrations, for example ``default`` and
``glasses``. Inside a profile each calibration is stored per display/camera
identity, so moving between a laptop panel and an external monitor reuses the
matching calibration instead of overwriting a single file::

    <data>/calibration/<profile>/<identity key>.json

The pre-profile file ``<data>/calibration/default.json`` is still used by the
``default`` profile while it matches the current identity.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fovea.paths import default_data_path
from fovea.webcam.calibration import CalibrationIdentity, load_model

DEFAULT_PROFILE = "default"
_ACTIVE_PROFILE_FILE = "active-profile"
_PROFILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_SLUG_UNSAFE = re.compile(r"[^a-z0-9]+")


def calibration_root() -> Path:
    return default_data_path() / "calibration"


def validate_profile_name(name: str) -> str:
    """Return ``name`` if it is a safe profile name, else raise ``ValueError``."""
    if not _PROFILE_NAME.fullmatch(name):
        raise ValueError(
            "profile name must be 1-64 letters, digits, '.', '_' or '-' "
            "and start with a letter or digit"
        )
    return name


def identity_key(identity: CalibrationIdentity) -> str:
    """Return a readable, collision-resistant file stem for one identity."""
    display = _SLUG_UNSAFE.sub("-", (identity.display_id or "display").lower()).strip("-")
    readable = (
        f"{display[:32] or 'display'}-{identity.display_width}x{identity.display_height}"
        f"-camera{identity.camera_index}-{identity.frame_width}x{identity.frame_height}"
    )
    canonical = json.dumps(identity.to_dict(), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
    return f"{readable}-{digest}"


def active_profile() -> str:
    """Return the profile chosen with ``fovea profiles use``, or ``default``."""
    try:
        stored = (default_data_path() / _ACTIVE_PROFILE_FILE).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return DEFAULT_PROFILE
    try:
        return validate_profile_name(stored)
    except ValueError:
        return DEFAULT_PROFILE


def set_active_profile(name: str) -> None:
    validate_profile_name(name)
    path = default_data_path() / _ACTIVE_PROFILE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{name}\n", encoding="utf-8")


def _profile_files(profile: str) -> list[Path]:
    directory = calibration_root() / profile
    if not directory.is_dir():
        return []
    files = [path for path in directory.glob("*.json") if path.is_file()]
    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)


def resolve_calibration_path(profile: str | None, identity: CalibrationIdentity) -> Path:
    """Pick the calibration file for ``identity`` in ``profile`` (or the active profile).

    Returns the newest stored calibration that matches the identity, then the
    legacy ``default.json`` for the default profile, and otherwise the path a
    new calibration for this identity should be saved to.
    """
    name = validate_profile_name(profile) if profile is not None else active_profile()
    for path in _profile_files(name):
        if _matching(path, identity):
            return path
    if name == DEFAULT_PROFILE:
        legacy = calibration_root() / f"{DEFAULT_PROFILE}.json"
        if _matching(legacy, identity):
            return legacy
    return calibration_root() / name / f"{identity_key(identity)}.json"


def _matching(path: Path, identity: CalibrationIdentity) -> bool:
    try:
        return load_model(path, expect=identity) is not None
    except (OSError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class StoredCalibration:
    profile: str
    path: Path
    identity: CalibrationIdentity | None
    updated: datetime


def list_calibrations() -> list[StoredCalibration]:
    """Return every stored calibration, grouped by profile name."""
    root = calibration_root()
    entries: list[StoredCalibration] = []
    candidates: list[tuple[str, Path]] = []
    legacy = root / f"{DEFAULT_PROFILE}.json"
    if legacy.is_file():
        candidates.append((DEFAULT_PROFILE, legacy))
    if root.is_dir():
        for directory in sorted(root.iterdir()):
            if directory.is_dir() and _PROFILE_NAME.fullmatch(directory.name):
                candidates.extend((directory.name, path) for path in _profile_files(directory.name))
    for profile, path in candidates:
        try:
            model = load_model(path)
        except (OSError, ValueError):
            continue
        if model is None:
            continue
        updated = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        entries.append(StoredCalibration(profile, path, model.identity, updated))
    return sorted(entries, key=lambda entry: (entry.profile, -entry.updated.timestamp()))
