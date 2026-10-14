"""Building one configuration through either sink and reading it back.

`emit` is the one build helper the suite shares. `STREAMING` and `IN_MEMORY` name the two sinks.
"""

from __future__ import annotations

from pathlib import Path

from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import StreamingSink
from tests.support import stores
from tests.support.emitted import Emitted
from tests.support.functional import read_functional
from tests.support.rdf import read_rdf
from tests.support.settings import Configuration, configured

STREAMING = "streaming"
IN_MEMORY = "in_memory"
BACKENDS = (STREAMING, IN_MEMORY)


def emit(
    spec: Configuration | OntoBuildSettings,
    path: Path,
    backend: str = STREAMING,
    data: OCEDStore | None = None,
    source: str = "matrix",
) -> Emitted:
    """Build one ontology, write it, and read it back.

    Args:
        spec: What to build, as a configuration or as bare settings with the default inputs.
        path: File to write. The extension is left as given.
        backend: Which sink to build through.
        data: Store to build from. Defaults to the matrix store.
        source: Value recorded as the source log annotation.

    Returns:
        The ontology that was written, read back from the file.
    """
    config = configured(spec)
    data = stores.build(stores.MATRIX_STORE) if data is None else data
    if backend == STREAMING:
        sink = StreamingSink(str(path), DEFAULT_IRI, config.settings.profile)
        OCEDOntoBuilder(DEFAULT_IRI, config.settings, sink=sink).build_ontology(
            data, config.inputs, source=source
        )
        return read_functional(path)

    builder = OCEDOntoBuilder(DEFAULT_IRI, config.settings)
    builder.build_ontology(data, config.inputs, source=source)
    builder.sink.save(str(path))
    return read_rdf(path, DEFAULT_IRI)
