"""Writing a config file, which only the CLI tests need."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def write_config(tmp_path: Path):
    """Return a factory that writes a TOML config into the test's directory."""

    def _write(name: str, body: str) -> Path:
        path = tmp_path / name
        path.write_text(body, encoding="utf-8")
        return path

    return _write
