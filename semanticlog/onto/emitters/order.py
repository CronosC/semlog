"""Event order, as properties between the events of a trace.

Every representation is per case concept: `df_order` orders the events of an
order, `df_item` those of an item. One event can be in several chains. The
settings select which representations run.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import IndividualKind
from semanticlog.core.errors import ConfigError
from semanticlog.data.cleaning import object_type_renames
from semanticlog.data.declare import DeclareResult, template_name
from semanticlog.data.flatten import iter_logs
from semanticlog.data.readable import ReadableStore
from semanticlog.data.tables import EntityKind
from semanticlog.onto.annotations import AnnotationWriter, split_constraint
from semanticlog.onto.emitted import EmittedIndividuals
from semanticlog.onto.emitters.base import Emitter
from semanticlog.onto.inputs import BuildInputs
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import AxiomSink

logger = logging.getLogger(__name__)

_EVENT_PAIR = (IndividualKind.EVENT, IndividualKind.EVENT)


class OrderEmitter(Emitter):
    """Emits directly-follows, eventually-follows and DECLARE membership."""

    def __init__(
        self,
        sink: AxiomSink,
        settings: OntoBuildSettings,
        annotations: AnnotationWriter,
        emitted: EmittedIndividuals,
    ) -> None:
        super().__init__(sink, settings, annotations, emitted)
        self._ef_pairs_so_far = 0

    def emit(self, data: ReadableStore, inputs: BuildInputs) -> None:
        """Emit whichever order representations the settings selected."""
        if self.settings.emits_traces:
            self._emit_trace_order(data, inputs.case_concepts)
        if self.settings.emits_declare and inputs.declare is not None:
            self._emit_declare(data, inputs.declare)

    def _emit_trace_order(
        self, data: ReadableStore, case_concepts: Sequence[str]
    ) -> None:
        """Emit the order super-properties, then the order of each case concept.

        The log is flattened once per case concept, and each projection is
        emitted before the next is built.
        """
        include_df = self.settings.emits_df
        include_ef = self.settings.emits_ef
        has_inverses = self.settings.has_inverses
        axiomatic = self.settings.axiomatic_inverses

        order_source = "the event order super-properties"
        if include_df:
            self.sink.add_property(
                vocab.DF,
                vocab.EVENT,
                vocab.EVENT,
                create_inverse=has_inverses,
                axiomatic_inverse=axiomatic,
                source=order_source,
            )
        if include_ef:
            self.sink.add_property(
                vocab.EF,
                vocab.EVENT,
                vocab.EVENT,
                create_inverse=has_inverses,
                axiomatic_inverse=axiomatic,
                source=order_source,
            )
        self.annotations.order_superproperties(include_df, include_ef)

        for case_concept, events in iter_logs(data, case_concepts):
            self._emit_case_order(case_concept, events, include_df, include_ef)

    def _emit_case_order(
        self,
        case_concept: str,
        events: pd.DataFrame,
        include_df: bool,
        include_ef: bool,
    ) -> None:
        """Emit the order of one flattened projection.

        The first and last event of each trace are also typed as its start
        and end events.

        Args:
            case_concept: Object type this projection was taken on.
            events: The flat log of that projection.
            include_df: Also emit directly-follows.
            include_ef: Also emit eventually-follows.
        """
        # `iter_logs` returns the frame in this order. The sort is kept
        # because it reallocates the frame more compactly than a copy.
        events = events.sort_values(
            ["case:concept:name", "time:timestamp"], kind="mergesort"
        )

        events["event_id"] = events.index
        shared = events.index[events.index.duplicated()].nunique()
        if shared and self.settings.emits_ef_axiomatic:
            logger.warning(
                "%s event(s) are in more than one '%s' trace. ef_%s derives "
                "eventually-follows across those traces. Use ef_explicit for "
                "pairs within one trace.",
                shared,
                case_concept,
                case_concept,
            )
        grouped = events.groupby("case:concept:name")
        events["next_event_id"] = grouped["event_id"].shift(-1)

        start_ids = grouped["event_id"].nth(0)
        end_ids = grouped["event_id"].nth(-1)

        has_inverses = self.settings.has_inverses
        axiomatic = self.settings.axiomatic_inverses
        start_cls = vocab.start_class(case_concept)
        end_cls = vocab.end_class(case_concept)
        case_source = f"case concept {case_concept!r}"
        self.sink.add_class(start_cls, vocab.EVENT, source=case_source)
        self.sink.add_class(end_cls, vocab.EVENT, source=case_source)
        self.annotations.trace_boundary_classes(case_concept, start_cls, end_cls)
        self.sink.assert_individuals(start_cls, start_ids, IndividualKind.EVENT)
        self.sink.assert_individuals(end_cls, end_ids, IndividualKind.EVENT)

        df_rel = vocab.df_property(case_concept)
        if include_df:
            df_pairs = events.loc[
                events["next_event_id"].notna(), ["event_id", "next_event_id"]
            ]
            self.sink.add_property(
                df_rel,
                vocab.EVENT,
                vocab.EVENT,
                vocab.DF,
                create_inverse=has_inverses,
                axiomatic_inverse=axiomatic,
                source=case_source,
            )
            self.annotations.df_property(case_concept, df_rel)
            self.assert_relations(
                df_rel, df_pairs, "event_id", "next_event_id", _EVENT_PAIR
            )

        if not include_ef:
            return

        ef_rel = vocab.ef_property(case_concept)
        self.sink.add_property(
            ef_rel,
            vocab.EVENT,
            vocab.EVENT,
            vocab.EF,
            create_inverse=has_inverses,
            axiomatic_inverse=axiomatic,
            source=case_source,
        )
        self.annotations.ef_property(case_concept, ef_rel)

        if self.settings.emits_ef_axiomatic:
            self._axiomatize_eventually_follows(df_rel, ef_rel, events)

        if self.settings.emits_ef_explicit:
            self._materialize_eventually_follows(ef_rel, events)

    def _axiomatize_eventually_follows(
        self, df_rel: str, ef_rel: str, events: pd.DataFrame
    ) -> None:
        """Declare eventually-follows transitive with directly-follows below it.

        With implicit inverses and no inverse axiom, the inverse properties
        get the same two axioms.
        """
        self.sink.add_sub_property(df_rel, ef_rel)
        self.sink.add_transitive(ef_rel)
        if self.settings.has_inverses and not self.settings.axiomatic_inverses:
            self.sink.add_sub_property(
                self.sink.inverse_name(df_rel), self.sink.inverse_name(ef_rel)
            )
            self.sink.add_transitive(self.sink.inverse_name(ef_rel))

    def _materialize_eventually_follows(
        self, ef_rel: str, events: pd.DataFrame
    ) -> None:
        """Assert every pair of an event and a later event of its trace.

        Works in every profile. The number of pairs is quadratic in trace
        length.

        Args:
            ef_rel: The eventually-follows property of the case concept.
            events: The flat log, with an `event_id` column.
        """
        groups = list(events.groupby("case:concept:name", sort=False))
        self._check_ef_budget(ef_rel, groups)

        for _, grp in groups:
            ids = grp["event_id"].values
            n = len(ids)
            if n < 2:
                continue
            row_idx, col_idx = np.triu_indices(n, k=1)
            ef_pairs = pd.DataFrame(
                {"event_id_from": ids[row_idx], "event_id_to": ids[col_idx]}
            )
            self.assert_relations(
                ef_rel, ef_pairs, "event_id_from", "event_id_to", _EVENT_PAIR
            )

    def _check_ef_budget(self, ef_rel: str, groups: list) -> None:
        """Raise when the pairs across all case concepts exceed `max_ef_pairs`."""
        projected = sum(len(grp) * (len(grp) - 1) // 2 for _, grp in groups)
        self._ef_pairs_so_far += projected
        total = self._ef_pairs_so_far
        budget = self.settings.max_ef_pairs
        if budget and total > budget:
            longest = max((len(grp) for _, grp in groups), default=0)
            raise ConfigError(
                f"ef_explicit would assert {total:,} pairs (reached at "
                f"'{ef_rel}'), over max_ef_pairs={budget:,}. The longest trace "
                f"has {longest:,} events. Use ef under OWL2DL, choose a case "
                "concept with shorter traces, or raise max_ef_pairs."
            )
        logger.info(
            "  %s: materializing %s eventually-follows pairs (%s so far)",
            ef_rel,
            f"{projected:,}",
            f"{total:,}",
        )

    def _emit_declare(self, data: ReadableStore, declare: DeclareResult) -> None:
        """Type each case object by the DECLARE constraints its trace satisfies.

        Every mined constraint becomes a class below the constraint hierarchy
        and below the object types of the case concepts it holds in. The case
        object of a trace that satisfies it becomes a member. The class is
        annotated with its template and, when the name splits into activities
        of the log, with the activity classes it constrains.

        Args:
            data: Store being built from. Its activity names split the
                constraint names for labels.
            declare: The mined constraints.

        Raises:
            NameCollisionError: If a constraint has the name of a class the
                build declared from the log. The sink raises it.
        """
        constraints = declare.constraint_names()
        object_types = _case_concept_types(data, declare.case_concepts)

        self.sink.add_class(
            vocab.DECLARE_CONSTRAINT,
            vocab.OBJECT,
            source="the DECLARE constraint hierarchy",
        )
        for constraint in constraints:
            source = f"DECLARE constraint {constraint!r}"
            self.sink.add_class(constraint, vocab.DECLARE_CONSTRAINT, source=source)
            # A constraint named after its own object type is a collision the
            # entity emitter reports. A class below itself would fail first.
            types = [
                object_types[c]
                for c in declare.case_concepts_of(constraint)
                if c in object_types and object_types[c] != constraint
            ]
            if types:
                self.sink.add_class(constraint, types, source=source)
        activities = data.types(EntityKind.EVENT)
        for prop in (vocab.TEMPLATE, vocab.ACTIVATION, vocab.TARGET):
            self.sink.declare_annotation_property(prop, "the DECLARE schema")
        for constraint in constraints:
            self._annotate_constraint(constraint, activities)
        self.annotations.declare_constraints(
            vocab.DECLARE_CONSTRAINT, constraints, activities
        )

        for constraint in constraints:
            self.sink.assert_individuals(
                constraint, declare.members(constraint), IndividualKind.OBJECT
            )

    def _annotate_constraint(self, constraint: str, activities: Sequence[str]) -> None:
        """Annotate one constraint class with its template and its activities.

        The activities are written only when every part of the name is an
        activity of the log. The first is the activation, the second the
        target.
        """
        template, parts = split_constraint(constraint, activities)
        self.sink.annotate_entity_with(
            constraint, vocab.TEMPLATE, template_name(template)
        )
        if not parts or any(part not in activities for part in parts):
            return
        self.sink.annotate_entity_with(constraint, vocab.ACTIVATION, parts[0], iri=True)
        if len(parts) > 1:
            self.sink.annotate_entity_with(constraint, vocab.TARGET, parts[1], iri=True)


def _case_concept_types(
    data: ReadableStore, case_concepts: Sequence[str]
) -> dict[str, str]:
    """Return each case concept with the object type the store holds it under.

    A concept that names no object type, before or after cleaning, is left out.
    """
    object_types = set(data.types(EntityKind.OBJECT))
    renames = object_type_renames(data)
    found = {}
    for case_concept in case_concepts:
        name = (
            case_concept if case_concept in object_types else renames.get(case_concept)
        )
        if name is not None:
            found[case_concept] = name
    return found
