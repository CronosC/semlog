"""The fixtures the whole suite shares.

The stores come from `tests/support/stores.py`, the build configurations from `tests/support/settings.py` and the build helper from `tests/support/build.py`. `built` caches each configuration per sink for the session.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path

import pytest

from semanticlog.core.enums import OWLProfile
from semanticlog.core.errors import SemlogError
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.data.oced import OCEDStore
from tests.support import stores
from tests.support.build import STREAMING, SinkCounts, emit_through
from tests.support.emitted import Emitted
from tests.support.recording import RecordingSink
from tests.support.settings import CONFIGURATIONS

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)


@pytest.fixture(autouse=True, scope="session")
def semlog_errors_are_one_line():
    """Fail every test that raises a `SemlogError` whose message holds a line break.

    The CLI prints such an error as one line.
    """
    original = SemlogError.__init__

    def one_line(self, *args):
        assert not any("\n" in str(arg) for arg in args), (
            f"{type(self).__name__} message holds a line break: {args!r}"
        )
        original(self, *args)

    SemlogError.__init__ = one_line
    yield
    SemlogError.__init__ = original


# --------------------------------------------------------------------------
# Skipping the tests that need a JVM
# --------------------------------------------------------------------------


def _jvm_available() -> bool:
    """Report whether a JVM is present, without starting one."""
    try:
        import jpype
    except ImportError:
        return False
    try:
        jpype.getDefaultJVMPath()
    except Exception:
        return False
    return True


# Every xdist worker that runs a reasoner test starts a JVM, which takes
# seconds, so they share one group. tryfirst sets the group before xdist reads it.
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items) -> None:
    """Put the reasoner tests in one xdist group, and skip them without a JVM."""
    reasoner = [item for item in items if item.get_closest_marker("reasoner")]
    if not reasoner:
        return
    for item in reasoner:
        item.add_marker(pytest.mark.xdist_group("jvm"))
    if _jvm_available():
        return
    skip = pytest.mark.skip(reason="no JVM found")
    for item in reasoner:
        item.add_marker(skip)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


@pytest.fixture
def tiny_store() -> OCEDStore:
    """Three events of one activity, on one order."""
    return stores.build("tiny")


@pytest.fixture
def pipeline_store() -> OCEDStore:
    """The committed OCEL 2.0 log, loaded and cleaned."""
    return stores.build("pipeline")


@pytest.fixture
def pipeline_raw_store() -> OCEDStore:
    """The committed OCEL 2.0 log as the loader produced it, before cleaning."""
    return stores.pipeline_raw()


@pytest.fixture
def awkward_store() -> OCEDStore:
    """Twelve orders and four items, carrying every hard case."""
    return stores.build("awkward")


@pytest.fixture
def store() -> Callable[[str], OCEDStore]:
    """Return a factory for a fresh copy of any of the three stores."""
    return stores.build


@pytest.fixture
def ocel() -> Path:
    """The committed OCEL 2.0 file itself."""
    return stores.PIPELINE_OCEL


# --------------------------------------------------------------------------
# Sinks
# --------------------------------------------------------------------------


@pytest.fixture
def recording_sink() -> Callable[..., RecordingSink]:
    """Return a factory for a sink that records each call."""

    def make(
        profile: OWLProfile = OWLProfile.OWL2DL, iri: str = DEFAULT_IRI
    ) -> RecordingSink:
        return RecordingSink(profile, iri)

    return make


# --------------------------------------------------------------------------
# Builds over the configuration matrix
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def matrix(tmp_path_factory) -> Callable[..., tuple[Emitted, Path, SinkCounts]]:
    """Return one configuration as the ontology it emits, the file it wrote and the counts of its sink.

    Builds are cached for the session and shared with `built`, `built_path`
    and `built_counts`.
    """
    root = tmp_path_factory.mktemp("matrix")
    cache: dict[tuple[str, str], tuple[Emitted, Path, SinkCounts]] = {}

    def get(name: str, backend: str = STREAMING) -> tuple[Emitted, Path, SinkCounts]:
        key = (name, backend)
        if key not in cache:
            suffix = "ofn" if backend == STREAMING else "owl"
            path = root / f"{name}.{backend}.{suffix}"
            emitted, sink = emit_through(CONFIGURATIONS[name](), path, backend)
            counts = SinkCounts(sink.declared_by_kind(), sink.axiom_count)
            cache[key] = (emitted, path, counts)
        return cache[key]

    return get


@pytest.fixture(scope="session")
def built(matrix) -> Callable[..., Emitted]:
    """Return the ontology one configuration emits, read back from the file."""

    def get(name: str, backend: str = STREAMING) -> Emitted:
        return matrix(name, backend)[0]

    return get


@pytest.fixture(scope="session")
def built_path(matrix) -> Callable[..., Path]:
    """Return the file one configuration wrote, for a reader that needs the file itself."""

    def get(name: str, backend: str = STREAMING) -> Path:
        return matrix(name, backend)[1]

    return get


@pytest.fixture(scope="session")
def built_counts(matrix) -> Callable[..., SinkCounts]:
    """Return what the sink of one configuration counted while it built."""

    def get(name: str, backend: str = STREAMING) -> SinkCounts:
        return matrix(name, backend)[2]

    return get
