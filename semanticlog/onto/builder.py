"""The build: emitters run in order over one store.

`OCEDOntoBuilder` checks the settings, the inputs and the store, then runs the
emitters over the store in `BUILD_STEPS` order through one sink.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from semanticlog.core.enums import SinkEntity
from semanticlog.core.summary import counted, report_by_kind, report_names
from semanticlog.data.oced import attribute_owners_by_kind
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
from semanticlog.onto.emitters.attributes import attribute_binnings
from semanticlog.onto.inputs import BuildInputs
from semanticlog.onto.result import BuildResult
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import AxiomSink, InMemorySink

logger = logging.getLogger(__name__)

_KIND_WORDS: dict[str, tuple[str, str]] = {
    SinkEntity.CLASS.value: ("class", "classes"),
    SinkEntity.OBJECT_PROPERTY.value: ("object property", "object properties"),
    SinkEntity.DATA_PROPERTY.value: ("data property", "data properties"),
    SinkEntity.ANNOTATION_PROPERTY.value: (
        "annotation property",
        "annotation properties",
    ),
}


@dataclass(frozen=True)
class Emitters:
    """The emitters of one build. They share one sink and one `EmittedIndividuals`.

    `AttributeEmitter` is not listed. The entity and history emitters call it
    for their attribute columns.
    """

    tbox: TBoxEmitter
    entities: EntityEmitter
    times: TimeEmitter
    order: OrderEmitter
    relations: RelationEmitter
    history: HistoryEmitter
    provenance: ProvenanceEmitter


@dataclass(frozen=True)
class BuildContext:
    """The arguments of every build step.

    Attributes:
        data: Store being built from.
        inputs: Case concepts, DECLARE constraints and annotations for it.
        settings: What to build.
        source: Path of the log, recorded as provenance.
        record: Anything else to record about this run.
        attribute_history: Whether this build emits the attribute change
            history. True when the setting is on and the store holds one.
    """

    data: ReadableStore
    inputs: BuildInputs
    settings: OntoBuildSettings
    source: str | None
    record: Mapping[str, Any] | None
    attribute_history: bool


@dataclass(frozen=True)
class BuildStep:
    """One step of a build, and what it logs before it runs.

    Attributes:
        message: Log line written before the step runs. May name `{count}`
            and `{detail}`.
        absent: Log line written instead when `count` is zero. Empty for a
            step that always has data.
        count: Number of items the step works on. Zero selects `absent`.
        detail: Further text for the message, for a step whose message names
            `{detail}`.
        when: Whether the step runs. None means it always runs.
        run: The emitter calls this step makes.
    """

    message: str
    run: Callable[[Emitters, BuildContext], None]
    absent: str = ""
    count: Callable[[BuildContext], int] | None = None
    detail: Callable[[BuildContext], str] | None = None
    when: Callable[[BuildContext], bool] | None = None


def _upper_schema(emitters: Emitters, context: BuildContext) -> None:
    """Emit the provenance record, then the upper schema."""
    emitters.provenance.emit(context.source, context.record)
    emitters.tbox.emit(attribute_history=context.attribute_history)


BUILD_STEPS: tuple[BuildStep, ...] = (
    BuildStep(
        message="Instantiating the upper schema..",
        run=_upper_schema,
    ),
    BuildStep(
        message="Instantiating event data ({count} activities)..",
        absent="No event data in the log..",
        count=lambda c: len(c.data.types(EntityKind.EVENT)),
        run=lambda e, c: e.entities.emit(c.data, EntityKind.EVENT),
    ),
    BuildStep(
        message="Instantiating event order ({detail})..",
        absent="No event order to instantiate..",
        count=lambda c: len(c.settings.order_representations),
        detail=lambda c: ", ".join(r.value for r in c.settings.order_representations),
        run=lambda e, c: e.order.emit(c.data, c.inputs),
    ),
    BuildStep(
        message="Instantiating time data..",
        absent="No time data in the log..",
        count=lambda c: len(c.data.types(EntityKind.EVENT)),
        run=lambda e, c: e.times.emit(c.data),
    ),
    BuildStep(
        message="Instantiating object data ({count} object types)..",
        absent="No object data in the log..",
        count=lambda c: len(c.data.types(EntityKind.OBJECT)),
        run=lambda e, c: e.entities.emit(c.data, EntityKind.OBJECT),
    ),
    BuildStep(
        message="Instantiating event to object relation data "
        "({count} relation types)..",
        absent="No event to object relations in the data..",
        count=lambda c: len(c.data.get_relation_types(RelationKind.E2O)),
        run=lambda e, c: e.relations.emit(c.data, RelationKind.E2O),
    ),
    BuildStep(
        message="Instantiating object to object relation data "
        "({count} relation types)..",
        absent="No object to object relations in the data..",
        count=lambda c: len(c.data.get_relation_types(RelationKind.O2O)),
        run=lambda e, c: e.relations.emit(c.data, RelationKind.O2O),
    ),
    BuildStep(
        message="Instantiating object attribute change history..",
        absent="No object attribute changes in the data..",
        count=lambda c: len(c.data.get_object_change_types()),
        when=lambda c: c.attribute_history,
        run=lambda e, c: e.history.emit(c.data),
    ),
)


class OCEDOntoBuilder:
    """Runs the emitters over a store in build order.

    The upper schema is emitted first. Each emitter declares its classes
    before it asserts individuals. The annotation document is applied last,
    after every name has been declared.

    Args:
        iri: Base IRI for the ontology.
        settings: Settings for this build.
        sink: Where to write. Defaults to an in-memory sink.

    Raises:
        ConfigError: If `iri` is empty or holds whitespace or `>`, and no
            sink is given.

    Attributes:
        sink: Where the axioms go.
        iri: Base IRI of the ontology, as the sink normalized it.
        emitted: The individuals the build has asserted, shared by every
            emitter.
    """

    def __init__(
        self,
        iri: str,
        settings: OntoBuildSettings,
        sink: AxiomSink | None = None,
    ) -> None:
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

    def _make_emitters(
        self, data: ReadableStore, document: Mapping[str, Mapping[str, str]]
    ) -> Emitters:
        """Create the emitters for one store and its annotation document."""
        owners = attribute_owners_by_kind(data)
        annotations = AnnotationWriter(self.sink, self.settings, data.backmap, document)
        parts = (self.sink, self.settings, annotations, self.emitted)
        attributes = AttributeEmitter(
            *parts, owners, attribute_binnings(data, self.settings)
        )
        times = TimeEmitter(*parts)
        return Emitters(
            tbox=TBoxEmitter(*parts),
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
            message: Log line written before the step runs.
            absent: Log line written instead when `count` is zero.
            count: Number of items the step works on.
        """
        logger.info(absent if absent and not count else message)
        mark = self.sink.declared_count
        axioms = self.sink.axiom_count
        yield
        self._report(self.sink.declared_since(mark), self.sink.axiom_count - axioms)

    def _report(self, declared: dict[str, list[str]], axioms: int) -> None:
        """Log what one step produced, as counts and then names.

        Args:
            declared: Names the step declared, grouped by kind.
            axioms: How many axioms it wrote.
        """
        report_by_kind(
            logger, _KIND_WORDS, declared, extra=[(axioms, "axiom", "axioms")]
        )

    def _report_dropped_unions(self) -> None:
        """Log how many union class expressions the profile dropped or replaced."""
        dropped = self.sink.dropped_unions
        if dropped:
            logger.info(
                "  omitted %s with a union of classes, which OWL 2 EL does not allow",
                counted(dropped, "axiom", "axioms"),
            )
        replaced = self.sink.replaced_unions
        if replaced:
            logger.info(
                "  replaced the union of classes in %s by their common superclass "
                "under OWL 2 EL",
                counted(replaced, "axiom", "axioms"),
            )

    def _report_what_was_kept(self) -> None:
        """Log the counts of declared names and of emitted values, bins and timestamps.

        These are what a streamed build keeps in memory.
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return
        kept = {"names": self.sink.declared_count, **self.emitted.counts()}
        logger.debug("kept: %s", ", ".join(f"{k}={v:,}" for k, v in kept.items()))

    def _result(self) -> BuildResult:
        """Close the sink and return the result."""
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
        """Raise if this builder has built once."""
        if self._claimed:
            raise ValueError(
                "This builder has produced an ontology. Construct a new "
                "builder for each build."
            )
        self._claimed = True

    def _apply_annotations(self, annotations: dict[str, dict[str, str]]) -> None:
        """Apply the annotations and warn about entries that name no declared entity."""
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
                "Applied %s of %s. %s name nothing this build declared.",
                len(annotations) - len(skipped),
                counted(len(annotations), "annotation entry", "annotation entries"),
                len(skipped),
            )
            report_names(logger, "skipped", sorted(skipped), level=logging.WARNING)

    def build_ontology(
        self,
        data: ReadableStore,
        inputs: BuildInputs | None = None,
        source: str | None = None,
        run: Mapping[str, Any] | None = None,
    ) -> BuildResult:
        """Build the ontology for one store.

        The settings, the inputs and the data are checked before anything is
        emitted. A build that fails aborts its sink, which removes any file it
        started.

        Args:
            data: Store to build from. Must be cleaned.
            inputs: Case concepts, DECLARE constraints and annotations for this
                log. Defaults to empty inputs, which suffice for a build with
                no event order.
            source: Path of the log, recorded as provenance.
            run: Other values to record about this run, such as the input
                checksum and the command. Each key becomes an annotation
                property and must be a safe local name.

        Returns:
            The finished build.

        Raises:
            ConfigError: If the settings or the inputs are unusable.
            InputError: If the store fails a pre-build check.
            ValueError: If this builder has been used, or a key of
                `run` cannot be an IRI local name.
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
        """Run the build without aborting the sink on failure."""
        self.settings.validate()
        inputs.validate(self.settings)
        audit_build(data, self.settings, inputs)
        self._claim_single_use()

        attribute_history = self.settings.emits_attribute_history
        if attribute_history and not data.ocel_mode:
            logger.warning(
                "ocel_extensions includes attribute_history, but the store was "
                "not created with ocel_mode=True. The attribute history is "
                "skipped."
            )
            attribute_history = False

        context = BuildContext(
            data=data,
            inputs=inputs,
            settings=self.settings,
            source=source,
            record=run,
            attribute_history=attribute_history,
        )
        emitters = self._make_emitters(data, inputs.annotations)
        for step in BUILD_STEPS:
            if step.when is not None and not step.when(context):
                continue
            count = 1 if step.count is None else step.count(context)
            detail = "" if step.detail is None else step.detail(context)
            with self._phase(
                step.message.format(count=count, detail=detail),
                step.absent,
                count,
            ):
                step.run(emitters, context)

        self._apply_annotations(inputs.annotations)
        self._report_dropped_unions()
        self._report_what_was_kept()
        logger.info("done.")
        return self._result()
