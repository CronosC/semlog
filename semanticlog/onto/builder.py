"""The build itself: emitters, in the order they have to run.

`OCEDOntoBuilder` owns a sink and a set of emitters and drives them over one
store. It decides the order and nothing else, so what each step produces stays
in the emitter that produces it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from semanticlog.core.summary import count_phrase, name_lines
from semanticlog.data.oced import attribute_owners
from semanticlog.data.readable import ReadableStore
from semanticlog.data.tables import EntityKind, RelationKind
from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.audit import audit_build
from semanticlog.onto.emitted import EmittedIndividuals
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
    report_conflated_attributes,
)
from semanticlog.onto.inputs import BuildInputs
from semanticlog.onto.result import BuildResult
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import AxiomSink, InMemorySink

logger = logging.getLogger(__name__)

_KIND_WORDS = {
    "class": ("class", "classes"),
    "object": ("object property", "object properties"),
    "data": ("data property", "data properties"),
}


@dataclass(frozen=True)
class Emitters:
    """Every step of one build, sharing one sink and one record of what it wrote."""

    tbox: TBoxEmitter
    attributes: AttributeEmitter
    entities: EntityEmitter
    times: TimeEmitter
    order: OrderEmitter
    relations: RelationEmitter
    history: HistoryEmitter
    provenance: ProvenanceEmitter


class OCEDOntoBuilder:
    """Runs the emitters over a store, in the order a build needs.

    The schema comes first so that everything else has something to hang
    below, each emitter declares its own classes before asserting its own
    individuals, and the annotation document is applied last, once every name
    it could refer to exists. A class declared by a
    later emitter than the individual it collides with is caught by cleaning
    rather than by the sink, which is why the reserved set is derived from
    the vocabulary.

    Attributes:
        sink: Where the axioms go.
        iri: Base IRI of the ontology, as the sink normalized it.
        emitted: What the build has asserted, shared by every emitter.
    """

    def __init__(
        self,
        iri: str,
        settings: OntoBuildSettings,
        sink: AxiomSink | None = None,
    ) -> None:
        """Prepare a builder.

        The emitters are made once the store is in hand, since what they need
        beyond the settings comes from the log.

        Args:
            iri: Base IRI for the ontology.
            settings: Settings for this build.
            sink: Where to write. Defaults to an in-memory sink.
        """
        self.sink: AxiomSink = (
            sink if sink is not None else InMemorySink(iri, settings.profile)
        )
        self.iri = self.sink.iri
        self._settings = settings
        self.emitted = EmittedIndividuals()
        self._claimed = False

    @property
    def settings(self) -> OntoBuildSettings:
        """The settings this build runs with."""
        return self._settings

    def _make_emitters(self, data: ReadableStore) -> Emitters:
        """Create the emitters for one store, complete from the start.

        What an emitter needs is read from the log here rather than bound
        into it afterwards, so nothing has to defend against being reached
        before it was ready.
        """
        owners = attribute_owners(data)
        report_conflated_attributes(owners)
        annotations = AnnotationWriter(self.sink, self.settings, data.backmap)
        parts = (self.sink, self.settings, annotations, self.emitted)
        attributes = AttributeEmitter(
            *parts, owners, attribute_binnings(data, self.settings)
        )
        times = TimeEmitter(*parts)
        return Emitters(
            tbox=TBoxEmitter(*parts),
            attributes=attributes,
            entities=EntityEmitter(*parts, attributes),
            times=times,
            order=OrderEmitter(*parts),
            relations=RelationEmitter(*parts),
            history=HistoryEmitter(*parts, attributes, times),
            provenance=ProvenanceEmitter(*parts),
        )

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

    def _report_what_was_kept(self) -> None:
        """Log what the build held on to, which is what its memory went on.

        Neither sink writes an axiom twice, and what that costs is one entry
        per declared name and one per distinct value, moment and bin. A
        streamed build is flat in events and grows in these, so this is where
        to look when one uses more memory than the log seems to warrant.
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return
        kept = {"names": self.sink.declared_count, **self.emitted.counts()}
        logger.debug("kept: " + ", ".join(f"{k}={v:,}" for k, v in kept.items()))

    def _result(self) -> BuildResult:
        """Package the finished build, closing the sink on the way.

        A sink finishes on close whatever work it could only do once no
        further axioms were coming.
        """
        self.sink.close()
        onto, path = self.sink.result_target()
        return BuildResult(
            onto=onto,
            settings=self.settings,
            iri=self.iri,
            path=path,
            stats={**self.sink.declared_by_kind(), "axioms": self.sink.axiom_count},
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

    def _apply_annotations(self, annotations: dict[str, dict[str, str]]) -> None:
        """Apply the supplied annotations, reporting entries that match nothing.

        A document written for another log, or for names as they appear before
        cleaning, would otherwise annotate nothing without saying so.
        """
        skipped = []
        for name, ann in annotations.items():
            if not self.sink.is_declared(name):
                skipped.append(name)
                continue
            self.sink.annotate_entity(
                name, label=ann.get("label"), comment=ann.get("comment")
            )
        if skipped:
            logger.warning(
                f"annotations: {len(skipped)} entry(s) name nothing this "
                f"build declared, and were skipped: {', '.join(sorted(skipped))}"
            )

    def build_ontology(
        self,
        data: ReadableStore,
        inputs: BuildInputs | None = None,
        source: str | None = None,
        run: Mapping[str, Any] | None = None,
    ) -> BuildResult:
        """Build the ontology for one store.

        The settings, the inputs and the data are checked before anything is
        emitted, so a build that cannot work fails before it has written half
        an ontology. A build that fails anyway abandons its sink, which is
        what removes a file it had started.

        Args:
            data: Store to build from. It has to have been cleaned already.
            inputs: Case concepts, DECLARE constraints and annotations for this
                log. Defaults to none of the three, which is enough for a build
                that emits no event order.
            source: Path of the log, recorded as provenance.
            run: Anything else to record about this run, such as the input
                checksum and the command.

        Returns:
            The finished build.

        Raises:
            ConfigError: If the settings or the inputs are unusable.
            InputError: If the store fails a pre-build check.
            ValueError: If this builder has already been used.
        """
        try:
            return self._build(
                data, BuildInputs() if inputs is None else inputs, source, run
            )
        except BaseException:
            self.sink.abort()
            raise

    def _build(
        self,
        data: ReadableStore,
        inputs: BuildInputs,
        source: str | None,
        run: Mapping[str, Any] | None,
    ) -> BuildResult:
        """Run the build itself, leaving a failure to `build_ontology`."""
        self.settings.validate()
        inputs.validate(self.settings)
        audit_build(data, self.settings, inputs)
        self._claim_single_use()

        attribute_history = self.settings.emits_attribute_history
        if attribute_history and not data.ocel_mode:
            logger.warning(
                "ATTRIBUTE_HISTORY extension enabled but OCEDStore was not "
                "created with ocel_mode=True. Skipping attribute history."
            )
            attribute_history = False

        emitters = self._make_emitters(data)
        with self._phase("Instantiating the upper schema.."):
            emitters.provenance.emit(source, run)
            emitters.tbox.emit(attribute_history=attribute_history)

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
            emitters.entities.emit(data, EntityKind.EVENT)
        with self._phase(
            f"Instantiating event order ({', '.join(r.value for r in order)})..",
            "No event order to instantiate..",
            len(order),
        ):
            emitters.order.emit(data, inputs)
        with self._phase(
            "Instantiating time data..", "No time data in the log..", activities
        ):
            emitters.times.emit(data)
        with self._phase(
            f"Instantiating object data ({object_types} object types)..",
            "No object data in the log..",
            object_types,
        ):
            emitters.entities.emit(data, EntityKind.OBJECT)
        with self._phase(
            f"Instantiating event to object relation data ({e2o} relation types)..",
            "No event to object relations in the data..",
            e2o,
        ):
            emitters.relations.emit_event_to_object(data)
        with self._phase(
            f"Instantiating object to object relation data ({o2o} relation types)..",
            "No object to object relations in the data..",
            o2o,
        ):
            emitters.relations.emit_object_to_object(data)

        if attribute_history:
            changes = len(data.get_object_change_types())
            with self._phase(
                "Instantiating object attribute change history..",
                "No object attribute changes in the data..",
                changes,
            ):
                emitters.history.emit(data)

        self._apply_annotations(inputs.annotations)
        self._report_dropped_unions()
        self._report_what_was_kept()
        logger.info("done.")
        return self._result()
