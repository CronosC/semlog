"""The relations of the log, as properties between individuals.

Each named relation becomes a sub-property of the schema property for its
family.
"""

from __future__ import annotations

from semanticlog.core import vocabulary as vocab
from semanticlog.data.readable import ReadableStore
from semanticlog.data.tables import EntityKind, RelationKind
from semanticlog.onto.emitters.base import Emitter

_SUPERPROPERTY = {
    RelationKind.E2O: vocab.EVENT_OBJECT_RELATION,
    RelationKind.O2O: vocab.OBJECT_OBJECT_RELATION,
}


class RelationEmitter(Emitter):
    """Emits one relation family per call."""

    def emit(self, data: ReadableStore, kind: RelationKind) -> None:
        """Emit one relation family, each relation with its domain and range.

        The domain of an event-to-object relation is the activities of its
        source events. For an object-to-object relation it is the source types
        the store records. Either falls back to the event or object class when
        it is empty, and under OWL 2 EL when it is a union. A range of several
        object types falls back to the object class under OWL 2 EL.

        Raises:
            ValueError: If `kind` is `RelationKind.ALL`.
        """
        superprop = _SUPERPROPERTY.get(kind)
        if superprop is None:
            raise ValueError(
                f"RelationEmitter takes one relation family, got {kind!r}, which "
                "names both relation families. Emit RelationKind.E2O and "
                "RelationKind.O2O separately."
            )
        for relation in data.get_relation_types(kind):
            table = data.get_relation(relation)
            from_types = table.from_type or []
            if kind is RelationKind.O2O:
                parent = vocab.OBJECT
                domain = from_types or [parent]
            else:
                parent = vocab.EVENT
                sources = table.data[table.from_id].dropna().unique()
                domain = data.types_holding(EntityKind.EVENT, sources) or [parent]
            self.sink.add_property(
                relation,
                superprop=superprop,
                domain_class=domain,
                range_class=list(table.to_type),
                create_inverse=self.settings.has_inverses,
                axiomatic_inverse=self.settings.axiomatic_inverses,
                domain_fallback=parent,
                range_fallback=vocab.OBJECT,
                source=f"relation {relation!r}",
            )
            self.annotations.relation(relation, from_types, table.to_type)
            self.assert_relations(relation, table.data, table.from_id, table.to_id)
