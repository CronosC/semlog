"""The fixtures the whole suite shares.

The stores come from `tests/support/stores.py` and the build configurations
from `tests/support/settings.py`. Everything that builds an ontology from the
matrix goes through `built`, which runs each configuration once per session.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path

import pytest

from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import StreamingSink
from tests.support import stores
from tests.support.ontology import Emitted, read_functional, read_rdf
from tests.support.recording import RecordingSink
from tests.support.settings import CONFIGURATIONS

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)

STREAMING = "streaming"
IN_MEMORY = "in_memory"
BACKENDS = (STREAMING, IN_MEMORY)


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


def pytest_collection_modifyitems(config, items) -> None:
    """Skip the reasoner tests when the machine has no JVM to run them on."""
    if not any(item.get_closest_marker("reasoner") for item in items):
        return
    if _jvm_available():
        return
    skip = pytest.mark.skip(reason="no JVM found, so the OWL API cannot start")
    for item in items:
        if item.get_closest_marker("reasoner"):
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
    """Return a factory for a sink that records calls instead of writing axioms."""

    def make(settings: OntoBuildSettings, iri: str = DEFAULT_IRI) -> RecordingSink:
        return RecordingSink(settings, iri)

    return make


# --------------------------------------------------------------------------
# Builds over the configuration matrix
# --------------------------------------------------------------------------


def emit(
    settings: OntoBuildSettings,
    path: Path,
    backend: str = STREAMING,
    data: OCEDStore | None = None,
    source: str = "matrix",
) -> Emitted:
    """Build one ontology, write it, and read it back.

    Args:
        settings: What to build.
        path: File to write. The extension is left as given.
        backend: Which sink to build through.
        data: Store to build from, defaulting to the matrix store.
        source: Value recorded as the source log annotation.

    Returns:
        The ontology that was written, read back from the file.
    """
    data = stores.build(stores.MATRIX_STORE) if data is None else data
    if backend == STREAMING:
        sink = StreamingSink(str(path), DEFAULT_IRI, settings)
        OCEDOntoBuilder(DEFAULT_IRI, settings, sink=sink).build_ontology(
            data, source=source
        )
        return read_functional(path)

    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
    builder.build_ontology(data, source=source)
    builder.sink.save(str(path))
    return read_rdf(path, DEFAULT_IRI)


@pytest.fixture(scope="session")
def built(tmp_path_factory) -> Callable[..., Emitted]:
    """Return the ontology one configuration emits, building it once per session.

    Every test that reads a whole build goes through this, so the matrix is
    built once rather than once per assertion.
    """
    root = tmp_path_factory.mktemp("matrix")
    cache: dict[tuple[str, str], Emitted] = {}

    def get(name: str, backend: str = STREAMING) -> Emitted:
        key = (name, backend)
        if key not in cache:
            suffix = "ofn" if backend == STREAMING else "owl"
            path = root / f"{name}.{backend}.{suffix}"
            cache[key] = emit(CONFIGURATIONS[name](), path, backend)
        return cache[key]

    return get
