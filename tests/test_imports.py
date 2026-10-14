"""What importing the package loads, and the names it exports."""

from __future__ import annotations

import subprocess
import sys

import pytest

HEAVY = ("pm4py", "owlapy", "jpype")

# A statement, and the heavy packages it may load. Each runs in a fresh process.
LOADS = {
    "import semanticlog": set(),
    "from semanticlog import OCEDStore": set(),
    "from semanticlog import Reasoner": {"owlapy", "jpype"},
}


def _loaded(statement: str) -> set[str]:
    script = f"import sys\n{statement}\nprint(' '.join(m for m in {HEAVY!r} if m in sys.modules))"
    finished = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    return set(finished.stdout.split())


@pytest.mark.slow
def test_only_the_reasoner_loads_the_owl_api_and_the_java_bridge():
    assert {statement: _loaded(statement) for statement in LOADS} == LOADS


def test_every_exported_name_is_listed_and_reachable():
    import semanticlog

    assert set(semanticlog.__all__) == set(semanticlog._EXPORTS) | {"__version__"}
    assert [n for n in semanticlog.__all__ if getattr(semanticlog, n) is None] == []


def test_an_unknown_name_raises_attribute_error():
    import semanticlog

    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        getattr(semanticlog, "nope")  # noqa: B009
