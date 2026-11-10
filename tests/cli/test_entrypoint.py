"""The console script and the `-m` entry point, run as subprocesses."""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO / "examples" / "pipeline.toml"

pytestmark = pytest.mark.slow


def run(*argv: str) -> subprocess.CompletedProcess:
    """Run the console script installed beside this interpreter.

    Skips when semlog is not installed into the interpreter running the tests.
    Fails when it is installed but has no console script.
    """
    try:
        distribution("semlog")
    except PackageNotFoundError:
        pytest.skip("semlog is not installed into this interpreter")
    executable = Path(sys.executable).parent / "semlog"
    assert executable.is_file(), f"semlog is installed but {executable} is missing"
    return subprocess.run(  # noqa: S603
        [str(executable), *argv], capture_output=True, text=True, cwd=REPO, check=False
    )


def test_the_committed_example_builds(tmp_path):
    finished = run(
        "build", "--config", str(EXAMPLE_CONFIG), "--output", str(tmp_path), "-q"
    )

    assert finished.returncode == 0, finished.stderr
    assert (tmp_path / "pipeline.ofn").is_file()


def test_the_module_runs_as_a_script():
    finished = subprocess.run(
        [sys.executable, "-m", "semanticlog.cli.main", "--version"],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )

    assert finished.returncode == 0
    assert finished.stdout.startswith("semlog ")
