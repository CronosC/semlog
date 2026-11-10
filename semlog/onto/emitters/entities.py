"""Events and objects, as one class per type and one individual per row.

`EntityEmitter` handles both kinds. They differ in their parent class and in
which columns are structural.
"""

from __future__ import annotations

from semlog.core import vocabulary as vocab
from semlog.data.readable import ReadableStore
from semlog.data.tables import EntityKind
from semlog.onto.annotations import AnnotationWriter
from semlog.onto.emitted import EmittedIndividuals
from semlog.onto.emitters.attributes import AttributeEmitter
from semlog.onto.emitters.base import Emitter
from semlog.onto.settings import OntoBuildSettings
from semlog.onto.sinks import AxiomSink

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
