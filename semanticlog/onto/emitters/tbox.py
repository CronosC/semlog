"""The TBox every build starts from.

`Object`, `Event`, `Attribute` and `Time` and the properties between them are
fixed. Everything from the log is placed below them.
"""

from __future__ import annotations

from semanticlog.core import vocabulary as vocab
from semanticlog.onto.emitters.base import Emitter
from semanticlog.onto.sinks.literals import DATE_TIME


class TBoxEmitter(Emitter):
    """Emits the fixed classes and properties of the TBox."""

    def emit(self, attribute_history: bool | None = None) -> None:
        """Emit the TBox.

        Args:
            attribute_history: Whether to emit the attribute change part of
                the schema. Defaults to `settings.emits_attribute_history`.
                The builder passes False when the store holds no history.
        """
        if attribute_history is None:
            attribute_history = self.settings.emits_attribute_history

        has_inverses = self.settings.has_inverses
        axiomatic = self.settings.axiomatic_inverses
        for name in (vocab.OBJECT, vocab.EVENT, vocab.ATTRIBUTE, vocab.TIME):
            self.sink.add_class(name, source="the TBox")
        for name in (vocab.OBJECT_ATTRIBUTE, vocab.EVENT_ATTRIBUTE):
            self.sink.add_class(name, vocab.ATTRIBUTE, source="the TBox")

        self.sink.add_property(
            vocab.OBSERVED_AT,
            domain_class=vocab.EVENT,
            range_class=vocab.TIME,
            create_inverse=has_inverses,
            axiomatic_inverse=axiomatic,
            source="the TBox",
        )
        self.sink.add_property(
            vocab.EVENT_OBJECT_RELATION,
            domain_class=vocab.EVENT,
            range_class=vocab.OBJECT,
            create_inverse=has_inverses,
            axiomatic_inverse=axiomatic,
            source="the TBox",
        )
        self.sink.add_property(
            vocab.OBJECT_OBJECT_RELATION,
            domain_class=vocab.OBJECT,
            range_class=vocab.OBJECT,
            create_inverse=has_inverses,
            axiomatic_inverse=axiomatic,
            source="the TBox",
        )
        self.sink.add_property(
            vocab.HAS_ATTRIBUTE,
            domain_class=[vocab.OBJECT, vocab.EVENT],
            range_class=vocab.ATTRIBUTE,
            create_inverse=has_inverses,
            axiomatic_inverse=axiomatic,
            source="the TBox",
        )
        self.sink.add_property(vocab.HAS_DATA_VALUE, data_prop=True, source="the TBox")

        self.sink.add_property(
            vocab.TIMESTAMP,
            domain_class=vocab.TIME,
            range_class=DATE_TIME,
            superprop=vocab.HAS_DATA_VALUE,
            data_prop=True,
            source="the TBox",
        )

        self.declare_functional(vocab.OBSERVED_AT)
        self.sink.add_functional(vocab.TIMESTAMP, data_prop=True)

        if attribute_history:
            self._emit_attribute_history_schema()

        top = [vocab.OBJECT, vocab.EVENT, vocab.ATTRIBUTE, vocab.TIME]
        if attribute_history:
            top.append(vocab.ATTRIBUTE_CHANGE)
        self.sink.add_disjoint_classes(top, source="the TBox")
        self.sink.add_disjoint_classes(
            [vocab.EVENT_ATTRIBUTE, vocab.OBJECT_ATTRIBUTE], source="the TBox"
        )

        self.annotations.tbox()

    def _emit_attribute_history_schema(self) -> None:
        """Emit the change record class and the properties around it."""
        has_inverses = self.settings.has_inverses
        axiomatic = self.settings.axiomatic_inverses
        self.sink.add_class(vocab.ATTRIBUTE_CHANGE, source="the TBox")
        self.sink.add_property(
            vocab.HAS_ATTRIBUTE_CHANGE,
            domain_class=vocab.OBJECT,
            range_class=vocab.ATTRIBUTE_CHANGE,
            create_inverse=has_inverses,
            axiomatic_inverse=axiomatic,
            source="the TBox",
        )
        self.sink.add_property(
            vocab.ATTRIBUTE_CHANGE_VALUE,
            domain_class=vocab.ATTRIBUTE_CHANGE,
            range_class=vocab.ATTRIBUTE,
            create_inverse=has_inverses,
            axiomatic_inverse=axiomatic,
            source="the TBox",
        )
        self.sink.add_property(
            vocab.CHANGED_AT,
            domain_class=vocab.ATTRIBUTE_CHANGE,
            range_class=vocab.TIME,
            create_inverse=has_inverses,
            axiomatic_inverse=axiomatic,
            source="the TBox",
        )
        self.sink.add_property(
            vocab.CAUSED_BY,
            domain_class=vocab.ATTRIBUTE_CHANGE,
            range_class=vocab.EVENT,
            create_inverse=has_inverses,
            axiomatic_inverse=axiomatic,
            source="the TBox",
        )
        self.declare_inverse_functional(vocab.HAS_ATTRIBUTE_CHANGE)
        self.declare_functional(vocab.CHANGED_AT)
        self.declare_functional(vocab.CAUSED_BY)
