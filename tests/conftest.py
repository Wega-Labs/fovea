from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep a developer's real config.toml from changing CLI test results."""
    monkeypatch.setenv("FOVEA_CONFIG", str(tmp_path / "no-config.toml"))
