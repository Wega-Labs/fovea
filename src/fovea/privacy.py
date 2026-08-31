"""Privacy-preserving paths and retention for opt-in diagnostic artifacts."""

from __future__ import annotations

import math
import os
import re
import sys
import time
from pathlib import Path

_RETENTION_PATTERN = re.compile(r"^(?P<value>\d+(?:\.\d+)?)(?P<unit>[smhd])$")
_UNIT_SECONDS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
_DIAGNOSTIC_PREFIX = "fovea-diagnostics-"
_DIAGNOSTIC_SUFFIXES = {".json", ".jsonl", ".ndjson"}


def parse_retention(value: str) -> float:
    """Parse a compact retention duration such as ``24h`` into seconds."""
    match = _RETENTION_PATTERN.fullmatch(value.strip().lower())
    if match is None:
        raise ValueError("retention must be a number followed by s, m, h, or d")
    seconds = float(match.group("value")) * _UNIT_SECONDS[match.group("unit")]
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("retention must be greater than zero")
    return seconds


def default_diagnostics_dir() -> Path:
    """Return Fovea's platform-native directory for opt-in diagnostic artifacts."""
    data_override = os.environ.get("FOVEA_DATA_DIR")
    if data_override:
        return Path(data_override).expanduser() / "diagnostics"
    if sys.platform == "darwin":
        data_root = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        data_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        data_root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return data_root / "fovea" / "diagnostics"


def purge_expired_diagnostics(
    directory: Path,
    retention_seconds: float,
    *,
    now: float | None = None,
) -> tuple[Path, ...]:
    """Delete expired diagnostic JSON files and return the paths that were removed.

    Cleanup is deliberately non-recursive and ignores unknown suffixes, directories,
    and files newer than the cutoff.
    """
    if not math.isfinite(retention_seconds) or retention_seconds <= 0.0:
        raise ValueError("retention_seconds must be greater than zero")
    if not directory.is_dir():
        return ()
    cutoff = (time.time() if now is None else now) - retention_seconds
    removed: list[Path] = []
    for path in directory.iterdir():
        if (
            not path.name.startswith(_DIAGNOSTIC_PREFIX)
            or path.suffix.lower() not in _DIAGNOSTIC_SUFFIXES
            or not path.is_file()
        ):
            continue
        try:
            if path.stat().st_mtime >= cutoff:
                continue
            path.unlink()
        except FileNotFoundError:
            continue
        removed.append(path)
    return tuple(removed)
