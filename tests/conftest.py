"""The fixtures the whole suite shares.

The stores come from `tests/support/stores.py`, the build configurations from `tests/support/settings.py` and the build helper from `tests/support/build.py`. `built` caches each configuration per sink for the session.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path

import pytest

from semanticlog.core.enums import OWLProfile
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.data.oced import OCEDStore
from tests.support import stores
from tests.support.build import STREAMING, emit
from tests.support.emitted import Emitted
from tests.support.recording import RecordingSink
from tests.support.settings import CONFIGURATIONS

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)


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
    skip = pytest.mark.skip(reason="no JVM found")
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
def awkward_shared_store() -> OCEDStore:
    """The awkward store with attributes several types carry kept under one name."""
    return stores.build("awkward_shared")


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
def matrix(tmp_path_factory) -> Callable[..., tuple[Emitted, Path]]:
    """Return one configuration as the ontology it emits and the file it wrote.

    Builds are cached for the session and shared with `built` and `built_path`.
    """
    root = tmp_path_factory.mktemp("matrix")
    cache: dict[tuple[str, str], tuple[Emitted, Path]] = {}

    def get(name: str, backend: str = STREAMING) -> tuple[Emitted, Path]:
        key = (name, backend)
        if key not in cache:
            suffix = "ofn" if backend == STREAMING else "owl"
            path = root / f"{name}.{backend}.{suffix}"
            cache[key] = (emit(CONFIGURATIONS[name](), path, backend), path)
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
