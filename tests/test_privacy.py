from __future__ import annotations

import os
from pathlib import Path

import pytest

from fovea.privacy import default_diagnostics_dir, parse_retention, purge_expired_diagnostics


def test_retention_parser_supports_compact_units() -> None:
    assert parse_retention("30m") == 1800.0
    assert parse_retention("24h") == 86400.0
    assert parse_retention("7d") == 604800.0


@pytest.mark.parametrize("value", ["", "24", "0h", "-1h", "tomorrow"])
def test_invalid_retention_is_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="retention"):
        parse_retention(value)


def test_expired_diagnostics_are_deleted_without_touching_other_files(tmp_path: Path) -> None:
    now = 2_000_000_000.0
    old = tmp_path / "fovea-diagnostics-old.ndjson"
    fresh = tmp_path / "fovea-diagnostics-fresh.json"
    unrelated = tmp_path / "notes.txt"
    unrelated_json = tmp_path / "unrelated.json"
    nested = tmp_path / "nested"
    for path in (old, fresh, unrelated, unrelated_json):
        path.write_text("{}\n", encoding="utf-8")
    nested.mkdir()
    (nested / "old.json").write_text("{}\n", encoding="utf-8")
    os.utime(old, (now - 25 * 3600, now - 25 * 3600))
    os.utime(fresh, (now - 23 * 3600, now - 23 * 3600))
    os.utime(unrelated, (now - 25 * 3600, now - 25 * 3600))
    os.utime(unrelated_json, (now - 25 * 3600, now - 25 * 3600))

    removed = purge_expired_diagnostics(tmp_path, parse_retention("24h"), now=now)

    assert removed == (old,)
    assert not old.exists()
    assert fresh.exists()
    assert unrelated.exists()
    assert unrelated_json.exists()
    assert (nested / "old.json").exists()


def test_data_directory_override_scopes_diagnostics(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FOVEA_DATA_DIR", str(tmp_path))
    assert default_diagnostics_dir() == tmp_path / "diagnostics"
