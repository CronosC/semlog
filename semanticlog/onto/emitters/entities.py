"""Events and objects, as one class per type and one individual per row.

The two halves of a store differ only in what their class hangs below and in
which columns carry structure rather than an attribute, so one emitter covers
both and the builder says which half it is emitting.
"""

from __future__ import annotations

from semanticlog.core import vocabulary as vocab
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind
from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.emitters.attributes import AttributeEmitter
from semanticlog.onto.emitters.base import Emitter
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import AxiomSink

_PARENTS = {
    EntityKind.EVENT: (vocab.EVENT, vocab.EVENT_ATTRIBUTE),
    EntityKind.OBJECT: (vocab.OBJECT, vocab.OBJECT_ATTRIBUTE),
}


class EntityEmitter(Emitter):
    """Emits the type classes of one half of a store, and their members."""

    def __init__(
        self,
        sink: AxiomSink,
        settings: OntoBuildSettings,
        annotations: AnnotationWriter,
        attributes: AttributeEmitter,
    ) -> None:
        super().__init__(sink, settings, annotations)
        self.attributes = attributes

    def emit(self, data: OCEDStore, kind: EntityKind) -> None:
        """Emit every type of one kind, its entities and their attributes.

        All classes are declared before any individual is asserted, so that a
        name used for both is caught while the sink can still say which two
        sources claimed it. An event timestamp is left to the time emitter,
        and every column that carries no structure is an attribute.

        Args:
            data: Store to emit from.
            kind: Whether to emit the activities or the object types.
        """
        self.annotations.bind(data.backmap)
        parent, attribute_parent = _PARENTS[kind]
        views = [v for v in data.tables(changes=False) if v.kind is kind]

        for view in views:
            self.sink.add_class(view.name, parent, source=view.origin)
        for view in views:
            self.annotations.entity_class(view.name, kind)
        for view in views:
            ids = view.frame[view.id_column]
            self.sink.assert_individuals(list(ids), view.name)
            self.annotations.entity_individuals(view.name, ids)
            self.attributes.emit_columns(
                view.frame.drop(columns=view.structural),
                str(view.id_column),
                attribute_parent,
            )
