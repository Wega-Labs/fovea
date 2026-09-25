from __future__ import annotations

import io
import os
import re
from dataclasses import replace
from pathlib import Path

import pytest

from fovea.cli import main
from fovea.paths import default_data_path
from fovea.profiles import (
    DEFAULT_PROFILE,
    active_profile,
    calibration_root,
    identity_key,
    list_calibrations,
    resolve_calibration_path,
    set_active_profile,
    validate_profile_name,
)
from fovea.webcam.calibration import CalibrationIdentity, load_model, save_model
from fovea.webcam.engine import GazeEngine, GazeSettings
from tests.test_online_calibration import _calibration_model

LAPTOP = CalibrationIdentity(None, 1280, 720, 0, 640, 480)
MONITOR = CalibrationIdentity("DELL U2720Q", 2560, 1440, 0, 640, 480)


def store(profile: str, identity: CalibrationIdentity, *, mtime: float | None = None) -> Path:
    path = calibration_root() / profile / f"{identity_key(identity)}.json"
    save_model(replace(_calibration_model(), identity=identity), path)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


@pytest.mark.parametrize("name", ["default", "glasses", "Desk-2", "a.b_c", "x" * 64])
def test_accepts_safe_profile_names(name: str) -> None:
    assert validate_profile_name(name) == name


@pytest.mark.parametrize("name", ["", ".hidden", "../up", "a/b", "a b", "x" * 65, "-flag"])
def test_rejects_unsafe_profile_names(name: str) -> None:
    with pytest.raises(ValueError, match="profile name"):
        validate_profile_name(name)


def test_identity_key_is_readable_stable_and_distinct() -> None:
    key = identity_key(MONITOR)

    assert key == identity_key(replace(MONITOR))
    assert key.startswith("dell-u2720q-2560x1440-camera0-640x480-")
    assert re.fullmatch(r"[a-z0-9-]+", key)
    assert identity_key(LAPTOP) != key
    assert identity_key(replace(LAPTOP, camera_index=1)) != identity_key(LAPTOP)


def test_new_identity_gets_its_own_file_in_the_active_profile() -> None:
    path = resolve_calibration_path(None, LAPTOP)

    assert path == calibration_root() / DEFAULT_PROFILE / f"{identity_key(LAPTOP)}.json"
    assert not path.exists()


def test_switching_displays_keeps_each_calibration() -> None:
    laptop = store(DEFAULT_PROFILE, LAPTOP)
    monitor = store(DEFAULT_PROFILE, MONITOR)

    assert resolve_calibration_path(None, LAPTOP) == laptop
    assert resolve_calibration_path(None, MONITOR) == monitor
    assert laptop.is_file()
    assert monitor.is_file()


def test_newest_matching_calibration_wins() -> None:
    older = store(DEFAULT_PROFILE, MONITOR, mtime=1_000)
    newer = calibration_root() / DEFAULT_PROFILE / "copied-by-hand.json"
    save_model(replace(_calibration_model(), identity=MONITOR), newer)
    os.utime(newer, (2_000, 2_000))

    assert resolve_calibration_path(None, MONITOR) == newer
    assert older.is_file()


def test_unlabeled_display_matches_a_stored_display_id() -> None:
    stored = store(DEFAULT_PROFILE, MONITOR)

    assert resolve_calibration_path(None, replace(MONITOR, display_id=None)) == stored


def test_profiles_are_isolated() -> None:
    store(DEFAULT_PROFILE, LAPTOP)

    glasses = resolve_calibration_path("glasses", LAPTOP)

    assert glasses.parent == calibration_root() / "glasses"
    assert not glasses.exists()


def test_legacy_default_file_is_reused_only_when_it_matches() -> None:
    legacy = calibration_root() / "default.json"
    save_model(replace(_calibration_model(), identity=LAPTOP), legacy)

    assert resolve_calibration_path(None, LAPTOP) == legacy
    assert resolve_calibration_path(None, MONITOR).parent == calibration_root() / DEFAULT_PROFILE
    assert resolve_calibration_path("glasses", LAPTOP).parent == calibration_root() / "glasses"


def test_corrupt_calibration_is_skipped() -> None:
    broken = calibration_root() / DEFAULT_PROFILE / "broken.json"
    broken.parent.mkdir(parents=True)
    broken.write_text("{not json", encoding="utf-8")

    assert resolve_calibration_path(None, LAPTOP).name == f"{identity_key(LAPTOP)}.json"


def test_active_profile_round_trips_and_falls_back() -> None:
    assert active_profile() == DEFAULT_PROFILE

    set_active_profile("glasses")
    assert active_profile() == "glasses"
    assert resolve_calibration_path(None, LAPTOP).parent == calibration_root() / "glasses"

    (default_data_path() / "active-profile").write_text("../../etc\n", encoding="utf-8")
    assert active_profile() == DEFAULT_PROFILE


def test_engine_auto_selects_and_saves_into_the_profile() -> None:
    monitor = store("glasses", MONITOR)

    engine = GazeEngine(GazeSettings(profile="glasses"), identity=MONITOR)
    fresh = GazeEngine(GazeSettings(profile="glasses"), identity=LAPTOP)

    assert engine.path == monitor
    assert engine.model is not None
    assert fresh.path.parent == calibration_root() / "glasses"
    assert fresh.model is None


def test_explicit_calibration_path_bypasses_profiles(tmp_path: Path) -> None:
    store(DEFAULT_PROFILE, LAPTOP)
    explicit = tmp_path / "explicit.json"

    engine = GazeEngine(GazeSettings(calibration_path=explicit, profile="glasses"), identity=LAPTOP)

    assert engine.path == explicit


def test_list_calibrations_reports_each_profile() -> None:
    store(DEFAULT_PROFILE, LAPTOP)
    store("glasses", MONITOR)
    save_model(replace(_calibration_model(), identity=LAPTOP), calibration_root() / "default.json")

    entries = list_calibrations()

    assert [(entry.profile, entry.identity) for entry in entries].count(("glasses", MONITOR)) == 1
    assert sum(entry.profile == DEFAULT_PROFILE for entry in entries) == 2
    assert all(load_model(entry.path) is not None for entry in entries)


def test_cli_profiles_use_and_list(capsys: pytest.CaptureFixture[str]) -> None:
    store("glasses", MONITOR)

    assert main(["profiles", "use", "glasses"]) == 0
    assert capsys.readouterr().out == "active_profile=glasses\n"

    assert main(["profiles", "list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "active_profile=glasses"
    assert lines[1].startswith(
        "profile=glasses display=2560x1440 display_id=DELL U2720Q camera=0 frame=640x480 updated="
    )


def test_cli_rejects_unsafe_profile_name(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["profiles", "use", "../x"]) == 2
    assert "profile name" in capsys.readouterr().out


def test_cli_passes_profile_to_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    captured: dict[str, object] = {}

    class EmptySource:
        def events(self):  # type: ignore[no-untyped-def]
            return iter(())

        def close(self) -> None:
            return None

    def factory(**kwargs: object) -> EmptySource:
        captured.update(kwargs)
        return EmptySource()

    assert main(["run", "--ndjson", "--profile", "glasses"], source_factory=factory) == 0
    settings = captured["settings"]
    assert isinstance(settings, GazeSettings)
    assert settings.profile == "glasses"


def test_cli_profile_and_calibration_path_are_exclusive(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["run", "--ndjson", "--profile", "glasses", "--calibration-path", "/tmp/c.json"])

    assert code == 2
    assert "not allowed with argument" in capsys.readouterr().out
