"""What importing the package costs.

pm4py and the Java OWL API are between them most of the cost of loading
semlog, and both are needed only once a build or a reasoner actually runs.
These run in a fresh process, since `sys.modules` in the test process already
holds everything the suite has imported.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow

HEAVY = ("pm4py", "owlapy", "jpype")


def loaded(statement: str) -> set[str]:
    """Return which of the heavy packages a statement pulls in."""
    script = (
        "import sys\n"
        f"{statement}\n"
        f"print(' '.join(m for m in {HEAVY!r} if m in sys.modules))"
    )
    finished = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    return set(finished.stdout.split())


def test_importing_the_package_starts_nothing_heavy():
    assert loaded("import semanticlog") == set()


def test_the_public_names_still_resolve():
    statement = (
        "from semanticlog import OCEDStore, OCEDOntoBuilder, OntoBuildSettings, "
        "PM4PyLoader, ClassicalLoader, Reasoner, DeclareResult, clean_oced_data\n"
        "assert OCEDStore().get_event_types() == []"
    )

    assert loaded(statement) >= {"pm4py", "owlapy"}


def test_the_reasoner_is_what_starts_the_java_bridge():
    assert "jpype" not in loaded("from semanticlog import OCEDStore")
    assert "jpype" in loaded("from semanticlog import Reasoner")


def test_every_exported_name_is_listed_and_reachable():
    """`__all__` and the lazy lookup table are two lists of the same names."""
    import semanticlog

    assert set(semanticlog.__all__) == set(semanticlog._EXPORTS) | {"__version__"}
    for name in semanticlog.__all__:
        assert getattr(semanticlog, name) is not None


def test_an_unknown_name_is_still_an_attribute_error():
    import semanticlog

    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        getattr(semanticlog, "nope")  # noqa: B009
