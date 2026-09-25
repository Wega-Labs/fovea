"""TOML defaults for the capture commands.

Keys are the long capture flags without the leading dashes, for example
``camera = 1`` or ``display-width = 1920``. Command-line flags override the file,
and the file overrides built-in defaults. Every value passes through the same
argparse conversion and validation as the matching flag.
"""

from __future__ import annotations

import argparse
import math
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path

from platformdirs import user_config_dir

from fovea.paths import APP_NAME, _environment_path

type ConfigValue = str | int | float | bool


def default_config_path() -> Path:
    """Return the ``FOVEA_CONFIG`` override or Fovea's platform config location."""
    override = _environment_path("FOVEA_CONFIG")
    if override is not None:
        return override
    return Path(user_config_dir(APP_NAME)) / "config.toml"


def load_config(path: Path, *, required: bool) -> dict[str, ConfigValue]:
    """Read a flat TOML table of flag defaults.

    A missing file is an error only when ``required`` is true, i.e. when the
    user named it explicitly.
    """
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError:
        if required:
            raise ValueError(f"config file not found: {path}") from None
        return {}
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid config file {path}: {exc}") from exc
    values: dict[str, ConfigValue] = {}
    for key, value in data.items():
        if not isinstance(value, str | int | float | bool):
            raise ValueError(f"config key {key!r} must be a string, number, or boolean")
        values[key] = value
    return values


def _flag_actions(actions: Iterable[argparse.Action]) -> dict[str, argparse.Action]:
    """Map each configurable long flag name to its action."""
    by_key: dict[str, argparse.Action] = {}
    for action in actions:
        # --ndjson selects the process protocol, so a host must always request it explicitly.
        if action.required or action.dest in {"help", "config", "ndjson"}:
            continue
        for option in action.option_strings:
            if option.startswith("--"):
                by_key[option[2:]] = action
    return by_key


def _convert(key: str, action: argparse.Action, value: ConfigValue) -> object:
    is_switch = action.nargs == 0
    if is_switch:
        if not isinstance(value, bool):
            raise ValueError(f"config key {key!r} must be true or false")
        return value
    if isinstance(value, bool):
        raise ValueError(f"config key {key!r} must not be a boolean")
    converter = action.type if callable(action.type) else str
    try:
        converted = converter(str(value))
    except (argparse.ArgumentTypeError, TypeError, ValueError) as exc:
        raise ValueError(f"config key {key!r}: {exc}") from exc
    if action.choices is not None and converted not in action.choices:
        choices = ", ".join(str(choice) for choice in action.choices)
        raise ValueError(f"config key {key!r} must be one of: {choices}")
    if isinstance(converted, Path):
        return converted.expanduser()
    return converted


def resolve_defaults(
    values: Mapping[str, ConfigValue],
    actions: Iterable[argparse.Action],
) -> dict[str, object]:
    """Validate config values against flag actions and return argparse defaults by dest."""
    by_key = _flag_actions(actions)
    defaults: dict[str, object] = {}
    for key, value in values.items():
        action = by_key.get(key)
        if action is None:
            known = ", ".join(sorted(by_key))
            raise ValueError(f"unknown config key {key!r}; expected one of: {known}")
        defaults[action.dest] = _convert(key, action, value)
    return defaults


def _toml_value(value: object) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value) if math.isfinite(value) else None
    if isinstance(value, str | Path):
        text = str(value)
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return None


def _seconds(value: object) -> object:
    """Render parsed retention seconds in the ``<number>s`` form the flag accepts."""
    if isinstance(value, float) and math.isfinite(value):
        return f"{value:f}".rstrip("0").rstrip(".") + "s"
    return value


# Flags whose parsed value is not in the form the flag accepts.
_RENDER_AS_FLAG = {"diagnostics_retention": _seconds}


def render_config(namespace: argparse.Namespace, actions: Iterable[argparse.Action]) -> str:
    """Render the effective configurable flag values as TOML that loads back identically."""
    lines = []
    for key, action in sorted(_flag_actions(actions).items()):
        value = getattr(namespace, action.dest, None)
        if action.nargs == 0 and not isinstance(value, bool):
            continue
        as_flag = _RENDER_AS_FLAG.get(action.dest)
        rendered = _toml_value(value if as_flag is None else as_flag(value))
        if rendered is not None:
            lines.append(f"{key} = {rendered}")
    return "".join(f"{line}\n" for line in lines)
