"""Event order, as properties between the events of a trace.

Order only exists relative to a case notion, so every representation here is
per case concept: `df_order` orders the events of an order, `df_item` those of
an item, and the same event can sit in several such chains. Which
representations run is decided by the settings, and they can be combined.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from semanticlog.core import vocabulary as vocab
from semanticlog.core.errors import ConfigError
from semanticlog.data.flatten import iter_logs
from semanticlog.data.readable import ReadableStore
from semanticlog.onto.emitters.base import Emitter
from semanticlog.onto.inputs import BuildInputs, constraint_names

logger = logging.getLogger(__name__)


def declare_membership(constraints: pd.DataFrame, constraint: str) -> pd.Series:
    """Return which traces satisfy one constraint, treating no answer as no."""
    return constraints[constraint].fillna(False).astype(bool)


class OrderEmitter(Emitter):
    """Emits directly-follows, eventually-follows and DECLARE membership."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ef_pairs_so_far = 0

    def emit(self, data: ReadableStore, inputs: BuildInputs) -> None:
        """Emit whichever order representations the settings selected."""
        if self.settings.emits_traces:
            self.emit_trace_order(data, inputs.case_concepts)
        if self.settings.emits_declare:
            self.emit_declare(inputs.declare_constraints)

    def emit_trace_order(
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
            self.emit_case_order(case_concept, events, include_df, include_ef)

    def emit_case_order(
        self,
        case_concept: str,
        events: pd.DataFrame,
        include_df: bool,
        include_ef: bool,
    ) -> None:
        """Emit the order of one flattened projection.

        Besides the pairs themselves, the first and last event of each trace
        are typed as its start and end events, so a trace boundary can be
        reached without asking for the absence of a predecessor.

        Args:
            case_concept: Object type this projection was taken on.
            events: The flat log of that projection.
            include_df: Also emit directly-follows.
            include_ef: Also emit eventually-follows.
        """
        # `iter_logs` already returns this sorted, so the sort is redundant for
        # order. It is not redundant for memory: sorting reallocates the frame
        # more compactly than copying it, and dropping it costs 7 MB per
        # 300k events.
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
            self._materialize_eventually_follows(ef_rel, events)

    def _axiomatize_eventually_follows(
        self, df_rel: str, ef_rel: str, events: pd.DataFrame
    ) -> None:
        """Leave the pairs to a reasoner, via transitivity.

        Directly-follows is placed below eventually-follows, which is declared
        transitive, so a reasoner derives every pair. The reflexive pairs are
        asserted instead of being written as a self-restriction, which OWL 2
        DL does not allow on a transitive property.
        """
        self.sink.add_sub_property(df_rel, ef_rel)
        self.sink.add_transitive(ef_rel)

        self_pairs = pd.DataFrame({"event_id": events["event_id"].unique()})
        self_pairs["event_id_self"] = self_pairs["event_id"]
        self.assert_relations(ef_rel, self_pairs, "event_id", "event_id_self")

    def _materialize_eventually_follows(
        self, ef_rel: str, events: pd.DataFrame
    ) -> None:
        """Assert every eventually-follows pair, including each event with itself.

        This needs no reasoner and works in every profile, at the cost of a
        number of pairs that grows with the square of the trace length.
        """
        groups = list(events.groupby("case:concept:name", sort=False))
        self._check_ef_budget(ef_rel, groups)

        for _, grp in groups:
            ids = grp["event_id"].values
            n = len(ids)
            if n == 0:
                continue
            row_idx, col_idx = np.triu_indices(n)
            ef_pairs = pd.DataFrame(
                {"event_id_from": ids[row_idx], "event_id_to": ids[col_idx]}
            )
            self.assert_relations(ef_rel, ef_pairs, "event_id_from", "event_id_to")

    def _check_ef_budget(self, ef_rel: str, groups: list) -> None:
        """Refuse to materialize more pairs than the build allows.

        The count runs across every case concept, since the budget covers what
        the whole build writes.
        """
        projected = sum(len(grp) * (len(grp) + 1) // 2 for _, grp in groups)
        self._ef_pairs_so_far += projected
        total = self._ef_pairs_so_far
        budget = self.settings.max_ef_pairs
        if budget and total > budget:
            longest = max((len(grp) for _, grp in groups), default=0)
            raise ConfigError(
                f"EF_EXPLICIT would materialize {total:,} pairs in total "
                f"(reached at '{ef_rel}'), over the max_ef_pairs limit of "
                f"{budget:,}. Eventually-follows is "
                f"quadratic in trace length and the longest trace here has {longest:,} "
                "events. Use EventOrderRepresentation.EF to let a reasoner derive "
                "them instead, choose a case concept with shorter traces, or raise "
                "settings.max_ef_pairs."
            )
        logger.info(
            f"  {ef_rel}: materializing {projected:,} eventually-follows pairs "
            f"({total:,} so far)"
        )

    def emit_declare(self, df_declare: pd.DataFrame | None) -> None:
        """Type each trace by the DECLARE constraints it satisfies.

        Every mined constraint becomes a class, and a trace that satisfies it
        becomes a member. The trace is the case object itself, so the
        assertion is about the object, not its events.

        Args:
            df_declare: One row per trace, as `BuildInputs` carries it.

        Raises:
            ConfigError: If no constraints are available.
        """
        if df_declare is None:
            raise ConfigError(
                "No DECLARE constraints available. Mine them with "
                "'semlog mine-declare' and pass the table as "
                "BuildInputs.declare_constraints."
            )

        constraints = constraint_names(df_declare)

        self.sink.add_class(
            vocab.DECLARE_CONSTRAINT, source="the DECLARE constraint hierarchy"
        )
        for constraint in constraints:
            self.sink.add_class(
                constraint,
                vocab.DECLARE_CONSTRAINT,
                source=f"DECLARE constraint {constraint!r}",
            )
        self.annotations.declare_constraints(vocab.DECLARE_CONSTRAINT, constraints)

        for constraint in constraints:
            trace_ids = df_declare.loc[
                declare_membership(df_declare, constraint), "trace_id"
            ]
            self.sink.assert_individuals(constraint, trace_ids)
