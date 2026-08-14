"""Building the upper schema on its own, with no data in it.

What the schema holds is decided by the settings alone, so the tests that
check it build it this way rather than through a whole log. This lives with
the tests because nothing in the pipeline builds a schema without data.
"""

from __future__ import annotations

from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.result import BuildResult
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import AxiomSink


def construct_tbox(
    iri: str,
    settings: OntoBuildSettings,
    path: str | None = None,
    attribute_history: bool | None = None,
    sink: AxiomSink | None = None,
) -> BuildResult:
    """Emit only the TBox.

    Args:
        iri: Base IRI for the ontology.
        settings: Settings for this build.
        path: Where to save it. Nothing is written when this is None.
        attribute_history: Whether to include the attribute change schema.
            Defaults to what the settings ask for.
        sink: Where to write. Defaults to an in-memory sink.

    Returns:
        The finished build.
    """
    builder = OCEDOntoBuilder(iri, settings, sink=sink)
    builder.tbox.emit(attribute_history)
    if path is not None:
        builder.sink.save(path)
    return builder._result()
