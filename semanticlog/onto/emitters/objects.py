"""Objects, as one class per object type and one individual per object."""

from __future__ import annotations

from semanticlog.core import vocabulary as vocab
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.emitters.base import _EntityEmitter


class ObjectEmitter(_EntityEmitter):
    """Emits the object type classes and the objects themselves."""

    def emit(self, data: OCEDStore) -> None:
        """Emit every object type and its objects, with their attributes."""
        self.annotations.bind(data.backmap)
        object_types = data.get_object_types()
        for obj_type in object_types:
            self.sink.add_class(
                obj_type, vocab.OBJECT, source=f"object type {obj_type!r}"
            )
        for obj_type in object_types:
            self.annotations.object_class(obj_type)
        for obj in object_types:
            table = data.get_object_table(obj)
            self.sink.assert_individuals(list(table.data[table.id]), obj)
            self.annotations.entity_individuals(obj, table.data[table.id])
            self.attributes.emit_columns(table.data, table.id, vocab.OBJECT_ATTRIBUTE)
