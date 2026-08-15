"""The relations of the log, as properties between individuals.

Each named relation becomes a property of its own below the schema property
for its family, so the log keeps its own vocabulary while a query can still
ask for every relation of a kind.
"""

from __future__ import annotations

from semanticlog.core import vocabulary as vocab
from semanticlog.data.readable import ReadableStore
from semanticlog.data.tables import RelationKind
from semanticlog.onto.emitters.base import Emitter


class RelationEmitter(Emitter):
    """Emits both relation families."""

    def emit_event_to_object(self, data: ReadableStore) -> None:
        """Emit the relations from events to objects."""
        self._emit(data, RelationKind.E2O, vocab.EVENT_OBJECT_RELATION)

    def emit_object_to_object(self, data: ReadableStore) -> None:
        """Emit the relations between objects."""
        self._emit(data, RelationKind.O2O, vocab.OBJECT_OBJECT_RELATION)

    def _emit(
        self,
        data: ReadableStore,
        relation_filter: RelationKind,
        superprop: str,
    ) -> None:
        """Emit one relation family, each relation with its domain and range.

        The domain of an event-to-object relation is always the event class.
        For an object-to-object relation it is the source types the store
        records, falling back to the object class when it records none.
        """
        for relation in data.get_relation_types(relation_filter):
            table = data.get_relation(relation)
            from_types = table.from_type or []
            if relation_filter is RelationKind.O2O:
                domain = from_types or [vocab.OBJECT]
            else:
                domain = [vocab.EVENT]
            self.sink.add_property(
                relation,
                superprop=superprop,
                domain_class=domain,
                range_class=list(table.to_type),
                create_inverse=self.settings.has_inverses,
                axiomatic_inverse=self.settings.axiomatic_inverses,
                source=f"relation {relation!r}",
            )
            self.annotations.relation(relation, from_types, table.to_type)
            self.assert_relations(relation, table.data, table.from_id, table.to_id)
