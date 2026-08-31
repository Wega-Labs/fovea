from __future__ import annotations

from pathlib import Path

from scripts.check_no_network import find_network_usage


def test_network_policy_flags_runtime_imports_but_not_schema_identifiers(
    tmp_path: Path,
) -> None:
    safe = tmp_path / "safe.py"
    unsafe = tmp_path / "unsafe.py"
    safe.write_text('SCHEMA = "https://example.test/schema"\n', encoding="utf-8")
    unsafe.write_text("import socket\n", encoding="utf-8")

    violations = find_network_usage(tmp_path, allowed=frozenset())

    assert len(violations) == 1
    assert violations[0].path == Path("unsafe.py")
    assert violations[0].detail == "socket"


def test_pinned_model_downloader_is_the_only_allowed_network_module(tmp_path: Path) -> None:
    model = tmp_path / "fovea" / "webcam" / "model.py"
    model.parent.mkdir(parents=True)
    model.write_text("import urllib.request\n", encoding="utf-8")
    assert find_network_usage(tmp_path) == ()
