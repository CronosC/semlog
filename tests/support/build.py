"""Building one configuration through either sink and reading it back.

`emit` is the one build helper the suite shares, and `emit_through` also returns the sink. `STREAMING` and `IN_MEMORY` name the two sinks.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from semlog.core.vocabulary import DEFAULT_IRI
from semlog.data.oced import OCEDStore
from semlog.onto.builder import OCEDOntoBuilder
from semlog.onto.settings import OntoBuildSettings
from semlog.onto.sinks import AxiomSink, StreamingSink
from tests.support import stores
from tests.support.emitted import Emitted
from tests.support.functional import read_functional
from tests.support.rdf import read_rdf
from tests.support.settings import Configuration, configured

STREAMING = "streaming"
IN_MEMORY = "in_memory"
BACKENDS = (STREAMING, IN_MEMORY)


class SinkCounts(NamedTuple):
    """What a sink counted while it built."""

    declared_by_kind: dict[str, int]
    axiom_count: int


def emit_through(
    spec: Configuration | OntoBuildSettings,
    path: Path,
    backend: str = STREAMING,
    data: OCEDStore | None = None,
) -> tuple[Emitted, AxiomSink]:
    """Build one ontology, write it, and read it back.

    Args:
        spec: What to build, as a configuration or as bare settings with the default inputs.
        path: File to write. The extension is left as given.
        backend: Which sink to build through.
        data: Store to build from. Defaults to the matrix store.

    Returns:
        The ontology that was written, read back from the file, and the closed sink that wrote it.
    """
    config = configured(spec)
    data = stores.build(stores.MATRIX_STORE) if data is None else data
    settings = config.settings
    if backend == STREAMING:
        sink = StreamingSink(
            str(path), DEFAULT_IRI, settings.profile, settings.tbox_iri
        )
        OCEDOntoBuilder(DEFAULT_IRI, settings, sink=sink).build_ontology(
            data, config.inputs
        )
        return read_functional(path), sink

    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
    builder.build_ontology(data, config.inputs)
    builder.sink.save(str(path))
    return read_rdf(path, DEFAULT_IRI, settings.tbox_iri), builder.sink


def emit(
    spec: Configuration | OntoBuildSettings,
    path: Path,
    backend: str = STREAMING,
    data: OCEDStore | None = None,
) -> Emitted:
    """Build one ontology, write it, and return it as read back from the file."""
    return emit_through(spec, path, backend, data)[0]
