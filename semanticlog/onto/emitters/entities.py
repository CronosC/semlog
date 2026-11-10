"""Events and objects, as one class per type and one individual per row.

`EntityEmitter` handles both kinds. They differ in their parent class and in
which columns are structural.
"""

from __future__ import annotations

from semanticlog.core import vocabulary as vocab
from semanticlog.data.readable import ReadableStore
from semanticlog.data.tables import EntityKind
from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.emitted import EmittedIndividuals
from semanticlog.onto.emitters.attributes import AttributeEmitter
from semanticlog.onto.emitters.base import Emitter
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import AxiomSink

_PARENTS = {EntityKind.EVENT: vocab.EVENT, EntityKind.OBJECT: vocab.OBJECT}


class EntityEmitter(Emitter):
    """Emits the type classes of events or objects, and their members."""

    def __init__(
        self,
        sink: AxiomSink,
        settings: OntoBuildSettings,
        annotations: AnnotationWriter,
        emitted: EmittedIndividuals,
        attributes: AttributeEmitter,
    ) -> None:
        super().__init__(sink, settings, annotations, emitted)
        self.attributes = attributes

    def emit(self, data: ReadableStore, kind: EntityKind) -> None:
        """Emit every type of one kind, its entities and their attributes.

        All classes are declared before any individual is asserted. Every
        non-structural column is emitted as an attribute. Event timestamps are
        left to `TimeEmitter`.
        """
        parent = _PARENTS[kind]
        views = [v for v in data.tables(changes=False) if v.kind is kind]

        for view in views:
            self.sink.add_class(view.name, parent, source=view.origin)
        self.sink.add_disjoint_classes([view.name for view in views])
        for view in views:
            self.annotations.entity_class(view.name, kind)
        for view in views:
            ids = view.frame[view.id_column]
            self.sink.assert_individuals(view.name, ids, kind.individual)
            self.annotations.entity_individuals(view.name, ids, kind)
            self.attributes.emit_columns(
                view.frame.drop(columns=view.structural),
                str(view.id_column),
                kind.individual,
            )
