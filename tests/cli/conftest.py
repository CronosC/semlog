"""Shared fixtures for the CLI tests."""

from pathlib import Path

import pytest

FIXTURE_OCEL = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline.jsonocel"


@pytest.fixture
def ocel() -> Path:
    """The small committed OCEL log every CLI test builds from."""
    return FIXTURE_OCEL


@pytest.fixture
def write_config(tmp_path: Path):
    """Write a TOML config into ``tmp_path`` and return its path."""

    def _write(name: str, body: str) -> Path:
        path = tmp_path / name
        path.write_text(body, encoding="utf-8")
        return path

    return _write
