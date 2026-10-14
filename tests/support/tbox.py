"""Building the upper schema on its own, with no data in it.

The settings alone decide what the schema holds.
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
