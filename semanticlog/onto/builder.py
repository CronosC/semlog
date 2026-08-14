"""The build itself: emitters, in the order they have to run.

`OCEDOntoBuilder` owns a sink and a set of emitters and drives them over one
store. It decides the order and nothing else, so what each step produces stays
in the emitter that produces it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from semanticlog.core.summary import count_phrase, name_lines
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind, RelationKind
from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.audit import audit_build
from semanticlog.onto.emitters import (
    AttributeEmitter,
    EntityEmitter,
    HistoryEmitter,
    OrderEmitter,
    ProvenanceEmitter,
    RelationEmitter,
    TBoxEmitter,
    TimeEmitter,
)
from semanticlog.onto.emitters.attributes import (
    attribute_binnings,
    attribute_owners,
    report_conflated_attributes,
)
from semanticlog.onto.result import BuildResult
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import AxiomSink, InMemorySink

logger = logging.getLogger(__name__)

_KIND_WORDS = {
    "class": ("class", "classes"),
    "object": ("object property", "object properties"),
    "data": ("data property", "data properties"),
}


class OCEDOntoBuilder:
    """Runs the emitters over a store, in the order a build needs.

    The order is not arbitrary. The schema comes first so that everything else
    has something to hang below, each emitter declares its own classes before
    asserting its own individuals, and the annotation document is applied
    last, once every name it could refer to exists. A class declared by a
    later emitter than the individual it collides with is caught by cleaning
    rather than by the sink, which is why the reserved set is derived from
    the vocabulary.

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

    def _make_emitters(self) -> None:
        """Create the emitters, sharing the sink, settings and annotations."""
        parts = (self.sink, self.settings, self.annotations)
        self.tbox = TBoxEmitter(*parts)
        self.attributes = AttributeEmitter(*parts)
        self.entities = EntityEmitter(*parts, self.attributes)
        self.times = TimeEmitter(*parts)
        self.order = OrderEmitter(*parts)
        self.relations = RelationEmitter(*parts)
        self.history = HistoryEmitter(*parts, self.attributes, self.times)
        self.provenance = ProvenanceEmitter(*parts)

    @contextmanager
    def _phase(self, message: str, absent: str = "", count: int = 1) -> Iterator[None]:
        """Log a build step, then the names and axioms it produced.

        Args:
            message: What the step is about to do.
            absent: What to say instead when the step has nothing to work
                with. Left empty by a step that always has something.
            count: How much the step has to work with. Zero picks `absent`.
        """
        logger.info(absent if absent and not count else message)
        mark = self.sink.declared_count
        axioms = self.sink.axiom_count
        yield
        self._report(
            self.sink.declared_since(mark),
            self.sink.axiom_count - axioms if axioms >= 0 else 0,
        )

    def _report(self, declared: dict[str, list[str]], axioms: int) -> None:
        """Log what one step produced, as counts and then names.

        The names are cut to a few lines each, unless the log is at debug
        level, where every one of them is printed.

        Args:
            declared: Names the step declared, grouped by kind.
            axioms: How many axioms it wrote.
        """
        counts = [
            (len(declared.get(kind, ())), singular, plural)
            for kind, (singular, plural) in _KIND_WORDS.items()
        ]
        counts.append((axioms, "axiom", "axioms"))
        phrase = count_phrase(counts)
        if not phrase:
            return

        logger.info(f"  {phrase}")
        complete = logger.isEnabledFor(logging.DEBUG)
        for kind, (_, plural) in _KIND_WORDS.items():
            for line in name_lines(plural, declared.get(kind, []), complete=complete):
                logger.info(line)

    def _report_dropped_unions(self) -> None:
        """Log the class expressions the profile refused, once for the build."""
        unions, classes = self.sink.dropped_unions
        if not unions:
            return
        logger.info(
            f"  omitted {count_phrase([(unions, 'union', 'unions')])} of "
            f"{count_phrase([(classes, 'class', 'classes')])} under OWL 2 EL, "
            "which has no disjunction"
        )

    def _result(self) -> BuildResult:
        """Package the finished build, closing the sink on the way.

        Closing is what lets a sink finish work it could only do once no
        further axioms were coming.
        """
        self.sink.close()
        onto, path = self.sink.result_target()
        return BuildResult(onto=onto, settings=self.settings, iri=self.iri, path=path)

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

        with self._phase("Instantiating the upper schema.."):
            self.tbox.emit(attribute_history=attribute_history)
            self.provenance.emit(source, run)

        self.annotations.bind(data.backmap)
        owners = attribute_owners(data)
        report_conflated_attributes(owners)
        self.attributes.bind_owners(owners)
        self.attributes.bind_binnings(attribute_binnings(data, self.settings))

        activities = len(data.get_event_types())
        object_types = len(data.get_object_types())
        e2o = len(data.get_relation_types(RelationKind.E2O))
        o2o = len(data.get_relation_types(RelationKind.O2O))
        order = self.settings.order_representations

        with self._phase(
            f"Instantiating event data ({activities} activities)..",
            "No event data in the log..",
            activities,
        ):
            self.entities.emit(data, EntityKind.EVENT)
        with self._phase(
            f"Instantiating event order ({', '.join(r.value for r in order)})..",
            "No event order to instantiate..",
            len(order),
        ):
            self.order.emit(data)
        with self._phase(
            "Instantiating time data..", "No time data in the log..", activities
        ):
            self.times.emit(data)
        with self._phase(
            f"Instantiating object data ({object_types} object types)..",
            "No object data in the log..",
            object_types,
        ):
            self.entities.emit(data, EntityKind.OBJECT)
        with self._phase(
            f"Instantiating event to object relation data ({e2o} relation types)..",
            "No event to object relations in the data..",
            e2o,
        ):
            self.relations.emit_event_to_object(data)
        with self._phase(
            f"Instantiating object to object relation data ({o2o} relation types)..",
            "No object to object relations in the data..",
            o2o,
        ):
            self.relations.emit_object_to_object(data)

        if attribute_history:
            changes = len(data.get_object_change_types())
            with self._phase(
                "Instantiating object attribute change history..",
                "No object attribute changes in the data..",
                changes,
            ):
                self.history.emit(data)

        self._apply_annotation_document()
        self._report_dropped_unions()
        logger.info("done.")
        return self._result()
