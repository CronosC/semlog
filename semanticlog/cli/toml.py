"""Reading a TOML config file, and rendering a resolved value back as TOML."""

from __future__ import annotations

import difflib
import tomllib
from pathlib import Path
from typing import Any

from semanticlog.cli.settings_table import BUILD_DEFAULTS
from semanticlog.core.enums import BuildEnum
from semanticlog.core.errors import ConfigError

# `extends` is read by the loader and is not a build setting.
_KNOWN_KEYS: tuple[str, ...] = (*sorted(BUILD_DEFAULTS), "extends")


def _check_keys(raw: dict[str, Any], source: Path) -> None:
    """Reject unknown settings and suggest the nearest one by spelling.

    Raises:
        ConfigError: If the file names a setting that does not exist.
    """
    unknown = sorted(set(raw) - set(_KNOWN_KEYS))
    if not unknown:
        return
    hints = []
    for key in unknown:
        close = difflib.get_close_matches(key, _KNOWN_KEYS, n=1)
        hints.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
    raise ConfigError(
        f"{source}: unknown setting(s): {', '.join(hints)}. Check the spelling, "
        "or run 'semlog build --help' for every setting."
    )


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Read a TOML config, following `extends` if it has one.

    Returns:
        The settings it names, with its own entries winning over inherited
        ones.

    Raises:
        ConfigError: If the file is missing, is not valid TOML, names an
            unknown setting, or the chain of `extends` forms a cycle.
    """
    return _load_config_file(Path(path), [])


def _load_config_file(path: Path, chain: list[Path]) -> dict[str, Any]:
    """Read one config. `chain` holds the files followed so far, for cycle detection.

    Raises:
        ConfigError: As `load_config_file`.
    """
    resolved = path.resolve()
    if resolved in chain:
        trail = " -> ".join(str(p) for p in [*chain, resolved])
        raise ConfigError(
            f"Config 'extends' forms a cycle: {trail}. Remove one 'extends'."
        )
    if not resolved.is_file():
        raise ConfigError(f"Config file not found: {path}. Check the path.")

    try:
        with resolved.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}.") from exc

    extends = raw.pop("extends", None)
    _check_keys(raw, path)
    if extends is None:
        return raw
    if not isinstance(extends, str):
        raise ConfigError(f"{path}: 'extends' must be a path string.")
    base = _load_config_file(resolved.parent / extends, [*chain, resolved])
    return {**base, **raw}


# Backslash first, so the escape it writes is not escaped again.
_TOML_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def toml_value(value: Any) -> str:
    """Render a resolved setting as TOML."""
    if isinstance(value, BuildEnum):
        return toml_value(value.value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    text = str(value)
    for char, escape in _TOML_ESCAPES.items():
        text = text.replace(char, escape)
    return f'"{text}"'
