"""Writing a config file and a flat log, which only the CLI tests need."""

from __future__ import annotations

from pathlib import Path

import pytest

_FLAT_CSV = """case_id,activity,timestamp,resource
o1,place_order,2024-01-01 08:00:00,alice
o1,pay,2024-01-02 09:00:00,bob
o2,place_order,2024-01-03 08:00:00,alice
o2,ship_order,2024-01-04 10:00:00,bob
"""


@pytest.fixture
def flat_csv(tmp_path: Path) -> Path:
    """A flat single-case log, under column names that are not the XES ones."""
    path = tmp_path / "flat.csv"
    path.write_text(_FLAT_CSV, encoding="utf-8")
    return path


@pytest.fixture
def write_config(tmp_path: Path):
    """Return a factory that writes a TOML config into the test's directory."""

    def _write(name: str, body: str) -> Path:
        path = tmp_path / name
        path.write_text(body, encoding="utf-8")
        return path

    return _write
