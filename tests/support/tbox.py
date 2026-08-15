"""Building the upper schema on its own, with no data in it.

What the schema holds is decided by the settings alone, so the tests that
check it build it this way rather than through a whole log. This lives with
the tests because nothing in the pipeline builds a schema without data.
"""

from __future__ import annotations

from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.emitted import EmittedIndividuals
from semanticlog.onto.emitters import TBoxEmitter
from semanticlog.onto.result import BuildResult
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import AxiomSink, InMemorySink


def construct_tbox(
    iri: str,
    settings: OntoBuildSettings,
    path: str | None = None,
    attribute_history: bool | None = None,
    sink: AxiomSink | None = None,
) -> BuildResult:
    """Emit only the TBox.

    The TBox emitter is driven directly rather than through a builder, since
    a builder makes its emitters from a store and there is none here.

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
    sink = InMemorySink(iri, settings.profile) if sink is None else sink
    TBoxEmitter(
        sink,
        settings,
        AnnotationWriter(sink, settings, {}),
        EmittedIndividuals(),
    ).emit(attribute_history)
    if path is not None:
        sink.save(path)
    sink.close()
    onto, written = sink.result_target()
    return BuildResult(onto=onto, settings=settings, iri=sink.iri, path=written)
