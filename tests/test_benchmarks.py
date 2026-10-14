"""The benchmark scripts. Each runs as a subprocess at a tiny size and exits cleanly."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCHMARKS = REPO / "benchmarks"

pytestmark = pytest.mark.slow


def run(script: str, size: str) -> subprocess.CompletedProcess:
    """Run one benchmark and return the finished process."""
    return subprocess.run(  # noqa: S603
        [sys.executable, str(BENCHMARKS / script), size],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )


@pytest.mark.parametrize(
    ("script", "size", "reports"),
    [
        ("sink_throughput.py", "500", "assert_individuals"),
        ("streaming_memory.py", "500", "axioms"),
    ],
)
def test_the_benchmark_runs_and_reports(script, size, reports):
    finished = run(script, size)

    assert finished.returncode == 0, finished.stderr
    assert reports in finished.stdout
