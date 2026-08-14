"""The installed command, run as a real process.

Everything else calls `main(argv)` in the test process, which never touches
the console script declared in pyproject.toml or the `__main__` guard. These
do, so a broken entry point fails here rather than for the first user.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO / "examples" / "pipeline.toml"

pytestmark = pytest.mark.slow


def run(*argv: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run the installed command and return the finished process."""
    executable = shutil.which("semlog")
    if executable is None:
        pytest.skip("semlog is not installed in this environment")
    return subprocess.run(  # noqa: S603
        [executable, *argv],
        capture_output=True,
        text=True,
        cwd=cwd or REPO,
        check=False,
    )


def test_the_console_script_reports_its_version():
    finished = run("--version")

    assert finished.returncode == 0
    assert finished.stdout.startswith("semlog ")


def test_the_console_script_prints_help_and_exits_two_with_no_command():
    finished = run()

    assert finished.returncode == 2
    assert "COMMAND" in finished.stdout


def test_the_module_can_be_run_directly():
    finished = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "semanticlog.cli.main", "--version"],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )

    assert finished.returncode == 0
    assert finished.stdout.startswith("semlog ")


def test_the_console_script_builds_an_ontology(ocel, tmp_path):
    finished = run("build", str(ocel), "--output", str(tmp_path), "-q")

    assert finished.returncode == 0, finished.stderr
    assert (tmp_path / "pipeline.owl").is_file()


def test_the_committed_example_builds(tmp_path):
    """The example in the README has to work from a fresh clone."""
    finished = run(
        "build", "--config", str(EXAMPLE_CONFIG), "--output", str(tmp_path), "-q"
    )

    assert finished.returncode == 0, finished.stderr
    assert (tmp_path / "pipeline.owl").is_file()
