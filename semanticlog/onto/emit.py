"""Base for the build steps that turn store contents into axioms.

An emitter is given what it needs and nothing else: somewhere to send axioms,
the settings that decide what to send, and the writer that phrases the labels.
It holds no ontology and no build state, so each step can be exercised on its
own against a recording sink.
"""

from __future__ import annotations

from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sink import AxiomSink


class Emitter:
    """A single step of a build.

    Args:
        sink: Where axioms go.
        settings: Build configuration.
        annotations: Writer for the derived labels and comments. It decides for
            itself whether anything is emitted, so emitters call it
            unconditionally.
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
