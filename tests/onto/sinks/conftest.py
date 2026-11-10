"""Fixtures for the sink tests: a recording sink, and each real sink in turn."""

from __future__ import annotations

from pathlib import Path

import pytest

from semanticlog.core.enums import OWLProfile
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.onto.sinks import AxiomSink, InMemorySink, StreamingSink
from tests.support.build import BACKENDS, STREAMING
from tests.support.emitted import Emitted
from tests.support.functional import read_functional
from tests.support.rdf import read_rdf


@pytest.fixture
def sink(recording_sink):
    return recording_sink(OWLProfile.OWL2DL)


@pytest.fixture(params=BACKENDS)
def real_sink(request, tmp_path):
    """Return a factory for a sink of each backend in turn. Open streaming sinks are aborted on teardown."""
    streaming: list[StreamingSink] = []

    def make(profile: OWLProfile = OWLProfile.OWL2DL) -> AxiomSink:
        if request.param == STREAMING:
            sink = StreamingSink(str(tmp_path / "streamed.ofn"), DEFAULT_IRI, profile)
            streaming.append(sink)
            return sink
        return InMemorySink(DEFAULT_IRI, profile)

    yield make
    for sink in streaming:
        sink.abort()


@pytest.fixture
def read_back(tmp_path):
    """Return a function that finishes a real sink and reads its file."""

    def read(sink: AxiomSink) -> Emitted:
        if isinstance(sink, StreamingSink):
            sink.close()
            return read_functional(Path(sink.path))
        path = tmp_path / "memory.owl"
        sink.save(str(path))
        return read_rdf(path, DEFAULT_IRI, sink.tbox_iri)

    return read
