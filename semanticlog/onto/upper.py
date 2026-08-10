from __future__ import annotations

from semanticlog.onto import vocabulary as vocab
from semanticlog.onto.builder import OCEDBaseBuilder
from semanticlog.onto.emit import Emitter
from semanticlog.onto.semlog import SemanticLog
from semanticlog.onto.settings import OwlProfile


class TBoxEmitter(Emitter):
    """Writes the upper ontology: the class hierarchy and the shared properties.

    Everything here is fixed schema, so the emitter reads only the settings and
    never touches an :class:`~semanticlog.data.oced.OCEDStore`.
    """

    def emit(self, attribute_history: bool | None = None) -> None:
        """Declare the core classes and properties.

        Args:
            attribute_history: Whether to include the ``AttributeChange`` schema.
                Defaults to the ``ATTRIBUTE_HISTORY`` extension setting. A full
                build passes ``False`` when the store cannot supply change data,
                so the schema is never written without the individuals to fill it.
        """
        if attribute_history is None:
            attribute_history = self.settings.instantiate_attribute_history

        has_inverses = self.settings.has_inverses
        is_dl = self.settings.profile == OwlProfile.OWL2DL
        self.sink.add_classes([vocab.OBJECT, vocab.EVENT, vocab.ATTRIBUTE, vocab.TIME])
        self.sink.add_classes(
            [vocab.OBJECT_ATTRIBUTE, vocab.EVENT_ATTRIBUTE], vocab.ATTRIBUTE
        )

        self.sink.add_property(
            vocab.OBSERVED_AT,
            domain_class=vocab.EVENT,
            range_class=vocab.TIME,
            create_inverse=has_inverses,
            inverse_name=vocab.OBSERVES if has_inverses else None,
        )
        self.sink.add_property(
            vocab.EVENT_OBJECT_RELATION,
            domain_class=vocab.EVENT,
            range_class=vocab.OBJECT,
            create_inverse=has_inverses,
        )
        self.sink.add_property(
            vocab.OBJECT_OBJECT_RELATION,
            domain_class=vocab.OBJECT,
            range_class=vocab.OBJECT,
            create_inverse=has_inverses,
        )
        self.sink.add_property(
            vocab.HAS_ATTRIBUTE,
            domain_class=f"{vocab.OBJECT} or {vocab.EVENT}" if is_dl else None,
            range_class=vocab.ATTRIBUTE,
            create_inverse=has_inverses,
            inverse_name=vocab.ATTRIBUTE_OF if has_inverses else None,
        )
        self.sink.add_property(
            vocab.HAS_DATA_VALUE,
            domain_class=f"{vocab.ATTRIBUTE} or {vocab.TIME}" if is_dl else None,
            data_prop=True,
        )

        if attribute_history:
            self._emit_attribute_history_schema()

        self.annotations.upper_ontology()

    def _emit_attribute_history_schema(self) -> None:
        has_inverses = self.settings.has_inverses
        # AttributeChange is a top-level class (not a subclass of Attribute).
        # Each individual represents one timestamped change to an object attribute.
        self.sink.add_class(vocab.ATTRIBUTE_CHANGE)
        self.sink.add_property(
            vocab.HAS_ATTRIBUTE_CHANGE,
            domain_class=vocab.OBJECT,
            range_class=vocab.ATTRIBUTE_CHANGE,
            create_inverse=has_inverses,
        )
        self.sink.add_property(
            vocab.ATTRIBUTE_CHANGE_VALUE,
            domain_class=vocab.ATTRIBUTE_CHANGE,
            range_class=vocab.ATTRIBUTE,
            create_inverse=has_inverses,
        )
        self.sink.add_property(
            vocab.CHANGED_AT,
            domain_class=vocab.ATTRIBUTE_CHANGE,
            range_class=vocab.TIME,
            create_inverse=has_inverses,
        )


class OCEDUpperOntoBuilder(OCEDBaseBuilder):
    """Builds the upper ontology (TBox schema) on its own.

    A full build writes the same schema through :class:`TBoxEmitter` directly;
    this class exists so the schema can be produced and inspected without any
    event data.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tbox = TBoxEmitter(self.sink, self.settings, self.annotations)

    def construct_upper_ontology(
        self, path: str | None = None, attribute_history: bool | None = None
    ) -> SemanticLog:
        """Build the upper ontology (TBox) schema and return it as a SemanticLog.

        Args:
            path: If provided, also save the OWL file to this path (e.g. for
                inspection or debugging). Optional.
            attribute_history: Whether to include the ``AttributeChange`` schema.
                Defaults to the ``ATTRIBUTE_HISTORY`` extension setting.

        Returns:
            SemanticLog containing the built upper ontology.
        """
        self.tbox.emit(attribute_history)

        if path is not None:
            self.save_onto(path)

        return SemanticLog(onto=self.onto, settings=self.settings, iri=self.iri)
