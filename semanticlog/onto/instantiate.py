from __future__ import annotations

import logging

import pandas as pd

from semanticlog.data.oced import OCEDStore, RelationKind
from semanticlog.onto.abox import (
    EventEmitter,
    ObjectEmitter,
    OrderEmitter,
    RelationEmitter,
    TimeEmitter,
)
from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.attributes import (
    AttributeEmitter,
    attribute_owners,
    report_conflated_attributes,
)
from semanticlog.onto.audit import audit_names
from semanticlog.onto.builder import OCEDBaseBuilder
from semanticlog.onto.history import HistoryEmitter
from semanticlog.onto.provenance import ProvenanceEmitter
from semanticlog.onto.semlog import SemanticLog
from semanticlog.onto.upper import TBoxEmitter


class OCEDOntoBuilder(OCEDBaseBuilder):
    """Full OCED-to-OWL pipeline builder: TBox + ABox.

    :meth:`build_ontology` validates the settings, then writes the upper ontology
    (TBox) followed by the ABox.  Nothing is emitted until validation passes, and
    an optional feature's schema is only written when its individuals follow.
    This is the main entry point for converting an
    :class:`~semanticlog.data.oced.OCEDStore` into a
    :class:`~semanticlog.onto.semlog.SemanticLog`.

    Each build step is an emitter this class holds rather than a method it
    inherits, so the steps can be run and tested independently of a builder.  The
    ``instantiate_*`` methods remain as named entry points onto them.

    Args:
        iri: Ontology IRI (e.g. ``DEFAULT_IRI``). Not a file path.
        settings: Build configuration controlling which ontology features are generated.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._make_emitters()

    def _make_emitters(self) -> None:
        """Build the emitter set against the current sink, settings and writer."""
        parts = (self.sink, self.settings, self.annotations)
        self.tbox = TBoxEmitter(*parts)
        self.attributes = AttributeEmitter(*parts)
        self.events = EventEmitter(*parts, self.attributes)
        self.objects = ObjectEmitter(*parts, self.attributes)
        self.times = TimeEmitter(*parts)
        self.order = OrderEmitter(*parts)
        self.relations = RelationEmitter(*parts)
        self.history = HistoryEmitter(*parts, self.attributes)
        self.provenance = ProvenanceEmitter(*parts)

    def build_ontology(self, data: OCEDStore, source: str | None = None) -> SemanticLog:
        """Validate the settings, build the TBox and ABox, and return a SemanticLog.

        Runs all ABox instantiation steps in order: events, event order, time,
        objects, e2o relations, o2o relations, and optionally attribute history.
        Applies the annotation document last if one is configured.

        Args:
            data: Cleaned :class:`~semanticlog.data.oced.OCEDStore`. Call
                :func:`~semanticlog.data.cleaning.clean_oced_data` before passing.
            source: Path or identifier of the input log, recorded in the
                ontology's provenance annotations when given.

        Returns:
            :class:`~semanticlog.onto.semlog.SemanticLog` containing the built ontology.

        Raises:
            ValueError: If the builder has already been used, the settings are
                inconsistent, an active strategy is missing its input, or two
                things in the data would be emitted under one IRI. Raised before
                any axiom is written.
        """
        self.settings.validate()
        audit_names(data, self.settings)
        # After the checks: a rejected build writes nothing, so the builder is
        # still clean and retrying it once the input is fixed is reasonable.
        self._claim_single_use()
        self._freeze_settings()

        attribute_history = self.settings.instantiate_attribute_history
        if attribute_history and not data.has_ocel_mode():
            logging.warning(
                "ATTRIBUTE_HISTORY extension enabled but OCEDStore was not "
                "created with ocel_mode=True. Skipping attribute history."
            )
            attribute_history = False

        self.tbox.emit(attribute_history=attribute_history)
        self.provenance.emit(source)

        self.annotations.bind(data.backmap)
        # Which types carry each attribute is a whole-store fact, so it is
        # gathered once here rather than rediscovered per table.
        owners = attribute_owners(data)
        report_conflated_attributes(owners)
        self.attributes.bind_owners(owners)

        logging.info("Instantiating ontology..")
        logging.info(
            f"Instantiating event data ({len(data.get_activities())} activities).."
        )
        self.events.emit(data)
        logging.info(
            f"Instantiating event order via the strategy: {self.settings.order_representation}.."
        )
        self.order.emit(data)
        logging.info("Instantiating time data..")
        self.times.emit(data)
        logging.info(
            f"Instantiating object data ({len(data.get_object_types())} objects).."
        )
        self.objects.emit(data)
        logging.info(
            f"Instantiating event to object relation data ({len(data.get_relation_types(RelationKind.E2O))} relation types).."
        )
        self.relations.emit_event_to_object(data)
        logging.info(
            f"Instantiating object to object relation data ({len(data.get_relation_types(RelationKind.O2O))} relation types).."
        )
        self.relations.emit_object_to_object(data)

        if attribute_history:
            logging.info("Instantiating object attribute change history..")
            self.history.emit(data)

        self.apply_annotation_document()
        logging.info("done.")
        return self._result()

    def _freeze_settings(self) -> None:
        """Take a private copy of the settings for the rest of the build.

        Everything that reads settings does so live, by design, so that a caller
        can adjust them between steps. Once a build has started that flexibility
        only means the finished ontology and the settings recorded next to it can
        disagree. Swapping in a snapshot here keeps both properties: adjustable
        until the build begins, fixed once it has.
        """
        self.settings.resolve()
        self.settings = self.settings.snapshot()
        self.sink.settings = self.settings
        self.annotations = AnnotationWriter(self, self.settings)
        self._make_emitters()

    def _result(self) -> SemanticLog:
        """Package the build output, which differs by sink.

        A streaming sink has already written the file and holds nothing, so the
        result carries the path instead of an ontology.
        """
        onto = getattr(self.sink, "onto", None)
        if onto is not None:
            return SemanticLog(onto=onto, settings=self.settings, iri=self.iri)
        self.sink.close()
        return SemanticLog(
            onto=None,
            settings=self.settings,
            iri=self.iri,
            path=getattr(self.sink, "path", None),
        )

    # ------------------------------------------------------------------
    # Named entry points onto the individual build steps
    # ------------------------------------------------------------------

    def instantiate_event_data(self, data: OCEDStore) -> None:
        """Assert all event individuals and their attribute values."""
        self.events.emit(data)

    def instantiate_object_data(self, data: OCEDStore) -> None:
        """Assert all object individuals and their attribute values."""
        self.objects.emit(data)

    def instantiate_time_data(self, data: OCEDStore) -> None:
        """Assert Time individuals and link each event to its timestamp."""
        self.times.emit(data)

    def instantiate_event_order(self, data: OCEDStore) -> None:
        """Assert event order under every active ``order_representation``."""
        self.order.emit(data)

    def instantiate_event_order_via_directly_follows(self, data: OCEDStore) -> None:
        """Assert the DF (and optionally EF) event-order properties."""
        self.order.emit_directly_follows(data)

    def instantiate_event_order_via_declare(self, data: OCEDStore) -> None:
        """Assert DECLARE constraint membership for each trace."""
        self.order.emit_declare(data)

    def instantiate_directly_follows(
        self, case_concept: str, events: pd.DataFrame, include_ef: bool
    ) -> None:
        """Assert DF (and optionally EF) relations for a single case concept."""
        self.order.emit_case_order(case_concept, events, include_ef)

    def instantiate_event_object_relation_data(self, data: OCEDStore) -> None:
        """Assert all e2o relation properties and their ABox triples."""
        self.relations.emit_event_to_object(data)

    def instantiate_object_to_object_relation_data(self, data: OCEDStore) -> None:
        """Assert all o2o relation properties and their ABox triples."""
        self.relations.emit_object_to_object(data)

    def instantiate_attributes(
        self, df: pd.DataFrame, key_col: str, attribute_parent: str
    ) -> None:
        """Instantiate attribute values for one event or object table."""
        self.attributes.emit(df, key_col, attribute_parent)

    def instantiate_object_attribute_history(self, data: OCEDStore) -> None:
        """Instantiate OCEL 2.0 object attribute change history."""
        self.history.emit(data)
