"""What every emitter is given to work with."""

from __future__ import annotations

from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import AxiomSink


class Emitter:
    """Base for a build step that writes axioms.

    An emitter holds no data of its own beyond what it is given here, so it
    can be tested against a recording sink with no ontology involved.

    Attributes:
        sink: Where axioms go.
        settings: Settings of this build.
        annotations: Writer for the labels and comments that go with the
            axioms. It does nothing unless the build asked for annotations.
    """

    def __init__(
        self,
        sink: AxiomSink,
        settings: OntoBuildSettings,
        annotations: AnnotationWriter,
    ) -> None:
        self.sink = sink
        self.settings = settings
        self.annotations = annotations
