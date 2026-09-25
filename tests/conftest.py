from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep a developer's stored calibrations and active profile out of test results."""
    monkeypatch.setenv("FOVEA_DATA_DIR", str(tmp_path / "fovea-data"))
