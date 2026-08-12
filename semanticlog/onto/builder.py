"""The build itself: emitters, in the order they have to run.

`OCEDOntoBuilder` owns a sink and a set of emitters and drives them over one
store. It decides the order and nothing else, so what each step produces stays
in the emitter that produces it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from semanticlog.data.oced import OCEDStore, RelationKind
from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.audit import audit_build
from semanticlog.onto.emitters import (
    AttributeEmitter,
    EventEmitter,
    HistoryEmitter,
    ObjectEmitter,
    OrderEmitter,
    ProvenanceEmitter,
    RelationEmitter,
    TBoxEmitter,
    TimeEmitter,
)
from semanticlog.onto.emitters.attributes import (
    attribute_owners,
    report_conflated_attributes,
)
from semanticlog.onto.result import BuildResult
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import AxiomSink, InMemorySink

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from owlapy.owl_ontology import Ontology


class OCEDOntoBuilder:
    """Runs the emitters over a store, in the order a build needs.

    The order is not arbitrary. The schema comes first so that everything else
    has something to hang below, classes are declared before the individuals
    that would collide with them, and the annotation document is applied last,
    once every name it could refer to exists.

    Attributes:
        sink: Where the axioms go.
        iri: Base IRI of the ontology, as the sink normalized it.
        annotations: Writer for derived labels and comments.
    """

    def __init__(
        self,
        iri: str,
        settings: OntoBuildSettings,
        sink: AxiomSink | None = None,
    ) -> None:
        """Prepare a builder and its emitters.

        Args:
            iri: Base IRI for the ontology.
            settings: Settings for this build.
            sink: Where to write. Defaults to an in-memory sink.
        """
        self.sink: AxiomSink = sink if sink is not None else InMemorySink(iri, settings)
        self.iri = self.sink.iri
        self._settings = settings
        self._annotation_doc: dict[str, dict[str, str]] = settings.annotation_data or {}
        self.annotations = AnnotationWriter(self.sink, settings)
        self._claimed = False
        self._make_emitters()

    @property
    def settings(self) -> OntoBuildSettings:
        """The settings this build runs with."""
        return self._settings

    @property
    def onto(self) -> Ontology:
        """The ontology under construction.

        Raises:
            AttributeError: If the sink keeps no ontology in memory.
        """
        onto = getattr(self.sink, "onto", None)
        if onto is None:
            raise AttributeError(
                f"{type(self.sink).__name__} does not keep an ontology in memory."
            )
        return onto

    def _make_emitters(self) -> None:
        """Create the emitters, sharing the sink, settings and annotations."""
        parts = (self.sink, self.settings, self.annotations)
        self.tbox = TBoxEmitter(*parts)
        self.attributes = AttributeEmitter(*parts)
        self.events = EventEmitter(*parts, self.attributes)
        self.objects = ObjectEmitter(*parts, self.attributes)
        self.times = TimeEmitter(*parts)
        self.order = OrderEmitter(*parts)
        self.relations = RelationEmitter(*parts)
        self.history = HistoryEmitter(*parts, self.attributes, self.times)
        self.provenance = ProvenanceEmitter(*parts)

    def _result(self) -> BuildResult:
        """Package the finished build, closing a streaming sink on the way."""
        onto = getattr(self.sink, "onto", None)
        if onto is not None:
            return BuildResult(onto=onto, settings=self.settings, iri=self.iri)
        self.sink.close()
        return BuildResult(
            onto=None,
            settings=self.settings,
            iri=self.iri,
            path=getattr(self.sink, "path", None),
        )

    def _claim_single_use(self) -> None:
        """Refuse a second build, which would merge two logs into one ontology."""
        if self._claimed:
            raise ValueError(
                "This builder has already produced an ontology. Axioms accumulate "
                "in one ontology object, so building again would merge the two "
                "logs. Construct a new builder for each build."
            )
        self._claimed = True

    def _apply_annotation_document(self) -> None:
        """Apply the supplied annotations, reporting entries that match nothing.

        A document written for another log, or for names as they appear before
        cleaning, would otherwise annotate nothing without saying so.
        """
        skipped = []
        for name, ann in self._annotation_doc.items():
            if not self.sink.is_declared(name):
                skipped.append(name)
                continue
            self.sink.annotate_entity(
                name, label=ann.get("label"), comment=ann.get("comment")
            )
        if skipped:
            logger.warning(
                f"annotation_document: {len(skipped)} entry(s) name nothing this "
                f"build declared, and were skipped: {', '.join(sorted(skipped))}"
            )

    def build_ontology(
        self,
        data: OCEDStore,
        source: str | None = None,
        run: Mapping[str, Any] | None = None,
    ) -> BuildResult:
        """Build the ontology for one store.

        The settings and the data are checked before anything is emitted, so
        a build that cannot work fails before it has written half an ontology.

        Args:
            data: Store to build from. It has to have been cleaned already.
            source: Path of the log, recorded as provenance.
            run: Anything else to record about this run, such as the input
                checksum and the command.

        Returns:
            The finished build.

        Raises:
            ValueError: If the settings are unusable, the store fails a
                pre-build check, or this builder has already been used.
        """
        self.settings.validate()
        audit_build(data, self.settings)
        self._claim_single_use()

        attribute_history = self.settings.emits_attribute_history
        if attribute_history and not data.ocel_mode:
            logger.warning(
                "ATTRIBUTE_HISTORY extension enabled but OCEDStore was not "
                "created with ocel_mode=True. Skipping attribute history."
            )
            attribute_history = False

        self.tbox.emit(attribute_history=attribute_history)
        self.provenance.emit(source, run)

        self.annotations.bind(data.backmap)
        owners = attribute_owners(data)
        report_conflated_attributes(owners)
        self.attributes.bind_owners(owners)

        logger.info("Instantiating ontology..")
        logger.info(
            f"Instantiating event data ({len(data.get_event_types())} activities).."
        )
        self.events.emit(data)
        logger.info(
            f"Instantiating event order via the strategy: {self.settings.order_representations}.."
        )
        self.order.emit(data)
        logger.info("Instantiating time data..")
        self.times.emit(data)
        logger.info(
            f"Instantiating object data ({len(data.get_object_types())} objects).."
        )
        self.objects.emit(data)
        logger.info(
            f"Instantiating event to object relation data ({len(data.get_relation_types(RelationKind.E2O))} relation types).."
        )
        self.relations.emit_event_to_object(data)
        logger.info(
            f"Instantiating object to object relation data ({len(data.get_relation_types(RelationKind.O2O))} relation types).."
        )
        self.relations.emit_object_to_object(data)

        if attribute_history:
            logger.info("Instantiating object attribute change history..")
            self.history.emit(data)

        self._apply_annotation_document()
        logger.info("done.")
        return self._result()


def construct_tbox(
    iri: str,
    settings: OntoBuildSettings,
    path: str | None = None,
    attribute_history: bool | None = None,
    sink: AxiomSink | None = None,
) -> BuildResult:
    """Build only the upper schema, with no data in it.

    Useful for inspecting or publishing the schema on its own.

    Args:
        iri: Base IRI for the ontology.
        settings: Settings for this build.
        path: Where to save it. Nothing is written when this is None.
        attribute_history: Whether to include the attribute change schema.
            Defaults to what the settings ask for.
        sink: Where to write. Defaults to an in-memory sink.

    Returns:
        The finished build.
    """
    builder = OCEDOntoBuilder(iri, settings, sink=sink)
    builder.tbox.emit(attribute_history)
    if path is not None:
        builder.sink.save(path)
    return builder._result()
