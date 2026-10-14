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
from semanticlog.core.errors import ConfigError
from semanticlog.data.declare import DeclareResult
from semanticlog.data.flatten import iter_logs
from semanticlog.data.readable import ReadableStore
from semanticlog.data.tables import EntityKind
from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.emitted import EmittedIndividuals
from semanticlog.onto.emitters.base import Emitter
from semanticlog.onto.inputs import BuildInputs
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import AxiomSink

logger = logging.getLogger(__name__)


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
        self.sink.assert_individuals(start_cls, start_ids)
        self.sink.assert_individuals(end_cls, end_ids)

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
            self.assert_relations(df_rel, df_pairs, "event_id", "next_event_id")

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
            self._materialize_eventually_follows(
                ef_rel, events, skip_self=self.settings.emits_ef_axiomatic
            )

    def _axiomatize_eventually_follows(
        self, df_rel: str, ef_rel: str, events: pd.DataFrame
    ) -> None:
        """Declare eventually-follows transitive with directly-follows below it.

        The reflexive pairs are asserted. OWL 2 DL allows no self-restriction
        on a transitive property. With implicit inverses and no inverse
        axiom, the inverse properties get the same two axioms.
        """
        self.sink.add_sub_property(df_rel, ef_rel)
        self.sink.add_transitive(ef_rel)
        if self.settings.has_inverses and not self.settings.axiomatic_inverses:
            self.sink.add_sub_property(
                self.sink.inverse_name(df_rel), self.sink.inverse_name(ef_rel)
            )
            self.sink.add_transitive(self.sink.inverse_name(ef_rel))

        self_pairs = pd.DataFrame({"event_id": events["event_id"].unique()})
        self_pairs["event_id_self"] = self_pairs["event_id"]
        self.assert_relations(ef_rel, self_pairs, "event_id", "event_id_self")

    def _materialize_eventually_follows(
        self, ef_rel: str, events: pd.DataFrame, skip_self: bool = False
    ) -> None:
        """Assert every eventually-follows pair, including each event with itself.

        Works in every profile. The number of pairs is quadratic in trace
        length.

        Args:
            ef_rel: The eventually-follows property of the case concept.
            events: The flat log, with an `event_id` column.
            skip_self: Leave out the reflexive pairs, when `EF` asserted them.
        """
        groups = list(events.groupby("case:concept:name", sort=False))
        self._check_ef_budget(ef_rel, groups, skip_self)

        for _, grp in groups:
            ids = grp["event_id"].values
            n = len(ids)
            if n == 0:
                continue
            row_idx, col_idx = np.triu_indices(n, k=1 if skip_self else 0)
            ef_pairs = pd.DataFrame(
                {"event_id_from": ids[row_idx], "event_id_to": ids[col_idx]}
            )
            self.assert_relations(ef_rel, ef_pairs, "event_id_from", "event_id_to")

    def _check_ef_budget(self, ef_rel: str, groups: list, skip_self: bool) -> None:
        """Raise when the pairs across all case concepts exceed `max_ef_pairs`."""
        offset = -1 if skip_self else 1
        projected = sum(len(grp) * (len(grp) + offset) // 2 for _, grp in groups)
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
        """Type each trace by the DECLARE constraints it satisfies.

        Every mined constraint becomes a class. The case object of a trace
        that satisfies it becomes a member.

        Args:
            data: Store being built from. Its activity names split the
                constraint names for labels.
            declare: The mined constraints.

        Raises:
            NameCollisionError: If a constraint has the name of a class the
                build declared from the log. The sink raises it.
        """
        constraints = declare.constraint_names()

        self.sink.add_class(
            vocab.DECLARE_CONSTRAINT, source="the DECLARE constraint hierarchy"
        )
        for constraint in constraints:
            self.sink.add_class(
                constraint,
                vocab.DECLARE_CONSTRAINT,
                source=f"DECLARE constraint {constraint!r}",
            )
        self.annotations.declare_constraints(
            vocab.DECLARE_CONSTRAINT, constraints, data.types(EntityKind.EVENT)
        )

        for constraint in constraints:
            self.sink.assert_individuals(constraint, declare.members(constraint))
