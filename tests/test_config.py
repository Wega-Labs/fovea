from __future__ import annotations

import argparse
import io
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from fovea.cli import _capture_actions, main
from fovea.config import default_config_path, load_config, render_config, resolve_defaults
from fovea.events import FoveaEvent


@pytest.fixture(autouse=True)
def _empty_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))


class EmptySource:
    def events(self) -> Iterator[FoveaEvent]:
        return iter(())

    def close(self) -> None:
        return None


def run_with_config(argv: list[str]) -> dict[str, object]:
    captured: dict[str, object] = {}

    def factory(**kwargs: object) -> EmptySource:
        captured.update(kwargs)
        return EmptySource()

    assert main(argv, source_factory=factory) == 0
    return captured


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_default_path_honors_environment_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FOVEA_CONFIG", str(tmp_path / "custom.toml"))
    assert default_config_path() == tmp_path / "custom.toml"

    monkeypatch.delenv("FOVEA_CONFIG")
    assert default_config_path().name == "config.toml"
    assert "fovea" in default_config_path().parts


def test_missing_file_is_empty_unless_named(tmp_path: Path) -> None:
    assert load_config(tmp_path / "missing.toml", required=False) == {}
    with pytest.raises(ValueError, match="not found"):
        load_config(tmp_path / "missing.toml", required=True)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("camera = ", "invalid config file"),
        ("[capture]\ncamera = 1\n", "must be a string, number, or boolean"),
        ("camera = [1, 2]\n", "must be a string, number, or boolean"),
    ],
)
def test_rejects_malformed_files(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(write(tmp_path / "c.toml", text), required=True)


def test_resolves_values_through_flag_validation() -> None:
    defaults = resolve_defaults(
        {
            "camera": 2,
            "fps": 24,
            "no-display": True,
            "diagnostics-retention": "2d",
            "backend": "mediapipe",
            "model": "~/models/face.task",
        },
        _capture_actions(),
    )

    assert defaults == {
        "camera": 2,
        "fps": 24.0,
        "no_display": True,
        "diagnostics_retention": 172800.0,
        "backend": "mediapipe",
        "model": Path("~/models/face.task").expanduser(),
    }


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"camra": 1}, "unknown config key 'camra'"),
        ({"ndjson": True}, "unknown config key 'ndjson'"),
        ({"config": "other.toml"}, "unknown config key 'config'"),
        ({"width": 0}, "config key 'width': must be greater than zero"),
        ({"fps": "fast"}, "config key 'fps': must be a number"),
        ({"no-display": "yes"}, "must be true or false"),
        ({"camera": True}, "must not be a boolean"),
        ({"backend": "coreml"}, "must be one of: mediapipe"),
        ({"diagnostics-retention": "2 weeks"}, "retention must be"),
    ],
)
def test_rejects_invalid_values(values: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_defaults(values, _capture_actions())  # type: ignore[arg-type]


def test_config_file_sets_defaults_and_flags_win(tmp_path: Path) -> None:
    config = write(tmp_path / "c.toml", "camera = 2\nwidth = 1280\nno-display = true\n")

    from_file = run_with_config(["run", "--ndjson", "--config", str(config)])
    overridden = run_with_config(["run", "--ndjson", "--config", str(config), "--camera", "3"])

    assert from_file["device_index"] == 2
    assert from_file["width"] == 1280
    assert from_file["show_calibration"] is False
    assert overridden["device_index"] == 3
    assert overridden["width"] == 1280


def test_default_location_is_read_without_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = write(tmp_path / "config.toml", "display-width = 1920\n")
    monkeypatch.setenv("FOVEA_CONFIG", str(config))

    assert run_with_config(["run", "--ndjson"])["display_width"] == 1920


def test_invalid_config_exits_with_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write(tmp_path / "c.toml", "height = -1\n")

    code = main(["run", "--ndjson", "--config", str(config)], source_factory=EmptySource)

    assert code == 2
    error = json.loads(capsys.readouterr().out)
    assert error["type"] == "error"
    assert "height" in error["message"]


def test_non_capture_commands_ignore_a_broken_default_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FOVEA_CONFIG", str(write(tmp_path / "c.toml", "camera = \n")))

    assert main(["schema"]) == 0
    assert capsys.readouterr().out


def test_show_round_trips(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = write(
        tmp_path / "c.toml",
        "camera = 2\n"
        "fps = 12.5\n"
        "no-display = true\n"
        'diagnostics-retention = "36h"\n'
        "display-id = 'studio \"left\" monitor'\n"
        'calibration-path = "~/fovea/cal.json"\n',
    )

    assert main(["config", "show", "--config", str(config)]) == 0
    shown = capsys.readouterr().out
    reloaded = write(tmp_path / "shown.toml", shown)
    assert main(["config", "show", "--config", str(reloaded)]) == 0

    assert capsys.readouterr().out == shown
    reloaded_defaults = resolve_defaults(load_config(reloaded, required=True), _capture_actions())
    assert reloaded_defaults["camera"] == 2
    assert reloaded_defaults["fps"] == 12.5
    assert reloaded_defaults["no_display"] is True
    assert reloaded_defaults["diagnostics_retention"] == 36 * 3600.0
    assert reloaded_defaults["display_id"] == 'studio "left" monitor'
    assert reloaded_defaults["calibration_path"] == Path("~/fovea/cal.json").expanduser()


def test_render_skips_unset_values() -> None:
    capture = _capture_actions()
    namespace = argparse.Namespace(camera=1, fps=None, no_display=False, display_id=None)
    rendered = render_config(namespace, capture)

    assert "camera = 1\n" in rendered
    assert "no-display = false\n" in rendered
    assert "fps" not in rendered
    assert "display-id" not in rendered


def test_config_path_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FOVEA_CONFIG", str(tmp_path / "x.toml"))

    assert main(["config", "path"]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "x.toml")
