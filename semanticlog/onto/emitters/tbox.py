"""The TBox every build starts from.

`Object`, `Event`, `Attribute` and `Time` and the properties between them are
fixed. Everything a log contributes hangs below them, so two ontologies built
from different logs are comparable.
"""

from __future__ import annotations

from semanticlog.core import vocabulary as vocab
from semanticlog.onto.emitters.base import Emitter


class TBoxEmitter(Emitter):
    """Emits the fixed classes and properties of the TBox."""

    def emit(self, attribute_history: bool | None = None) -> None:
        """Emit the TBox.

        Args:
            attribute_history: Whether to include the attribute change part of
                the schema. Defaults to what the settings ask for. The builder
                passes False when the settings ask for it but the store cannot
                supply it, so the schema does not promise something no data
                will fill.
        """
        if attribute_history is None:
            attribute_history = self.settings.emits_attribute_history

        has_inverses = self.settings.has_inverses
        axiomatic = self.settings.axiomatic_inverses
        self.sink.add_classes(
            [vocab.OBJECT, vocab.EVENT, vocab.ATTRIBUTE, vocab.TIME],
            source="the TBox",
        )
        self.sink.add_classes(
            [vocab.OBJECT_ATTRIBUTE, vocab.EVENT_ATTRIBUTE],
            vocab.ATTRIBUTE,
            source="the TBox",
        )

        self.sink.add_property(
            vocab.OBSERVED_AT,
            domain_class=vocab.EVENT,
            range_class=vocab.TIME,
            create_inverse=has_inverses,
            axiomatic_inverse=axiomatic,
            inverse_name=vocab.OBSERVES if has_inverses else None,
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
            inverse_name=vocab.ATTRIBUTE_OF if has_inverses else None,
            source="the TBox",
        )
        self.sink.add_property(
            vocab.HAS_DATA_VALUE,
            domain_class=[vocab.ATTRIBUTE, vocab.TIME],
            data_prop=True,
            source="the TBox",
        )

        if attribute_history:
            self._emit_attribute_history_schema()

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
