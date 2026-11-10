"""What importing the package loads, and the names it exports."""

from __future__ import annotations

import subprocess
import sys

import pytest

HEAVY = ("pm4py", "owlapy", "jpype")

# A statement, and the heavy packages it may load. Each runs in a fresh process.
LOADS = {
    "import semlog": set(),
    "from semlog import OCEDStore": set(),
    "from semlog import Reasoner": {"owlapy", "jpype"},
    "from semlog.cli import main\n"
    "from contextlib import redirect_stdout, suppress\n"
    "with redirect_stdout(None), suppress(SystemExit):\n"
    "    main(['--help'])": set(),
}


def _loaded(statement: str) -> set[str]:
    script = f"import sys\n{statement}\nprint(' '.join(m for m in {HEAVY!r} if m in sys.modules))"
    finished = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    return set(finished.stdout.split())


@pytest.mark.slow
def test_only_the_reasoner_loads_the_owl_api_the_java_bridge_or_pm4py():
    assert {statement: _loaded(statement) for statement in LOADS} == LOADS


def test_every_exported_name_is_listed_and_reachable():
    import semlog

    assert set(semlog.__all__) == set(semlog._EXPORTS) | {"__version__"}
    assert [n for n in semlog.__all__ if getattr(semlog, n) is None] == []


def test_an_unknown_name_raises_attribute_error():
    import semlog

    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        getattr(semlog, "nope")  # noqa: B009
