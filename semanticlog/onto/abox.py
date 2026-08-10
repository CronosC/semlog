"""The ABox build steps: events, event order, time, objects and relations.

Each emitter reads an :class:`~semanticlog.data.oced.OCEDStore` and writes to an
:class:`~semanticlog.onto.sink.AxiomSink`. They share no state, so the order they
run in is the pipeline's decision rather than something encoded in a class
hierarchy.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from semanticlog.data.oced import OCEDStore, RelationKind
from semanticlog.onto import vocabulary as vocab
from semanticlog.onto.annotations import AnnotationWriter, timestamp_labels
from semanticlog.onto.attributes import AttributeEmitter
from semanticlog.onto.emit import Emitter
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sink import AxiomSink


class _EntityEmitter(Emitter):
    """Shared setup for the two emitters that also write attribute values."""

    def __init__(
        self,
        sink: AxiomSink,
        settings: OntoBuildSettings,
        annotations: AnnotationWriter,
        attributes: AttributeEmitter | None = None,
    ) -> None:
        super().__init__(sink, settings, annotations)
        self.attributes = attributes or AttributeEmitter(sink, settings, annotations)


class EventEmitter(_EntityEmitter):
    """Event individuals, their activity classes and their attribute values."""

    def emit(self, data: OCEDStore) -> None:
        """Assert all event individuals and their attribute values.

        Creates an OWL subclass of ``Event`` for each activity and asserts one
        individual per event row.

        Args:
            data: Populated OCEDStore.
        """
        self.annotations.bind(data.backmap)
        activities = data.get_activities()

        self.sink.add_classes(activities, vocab.EVENT)
        for activity in activities:
            self.annotations.activity_class(activity)
        for activity in activities:
            table = data.get_events(activity)
            self.sink.assert_individuals(list(table.data[table.id]), activity)
            self.annotations.entity_individuals(activity, table.data[table.id])
            self.attributes.emit(
                table.data.drop(columns=[table.time]),
                table.id,
                vocab.EVENT_ATTRIBUTE,
            )


class ObjectEmitter(_EntityEmitter):
    """Object individuals, their type classes and their attribute values."""

    def emit(self, data: OCEDStore) -> None:
        """Assert all object individuals and their attribute values.

        Creates an OWL subclass of ``Object`` for each object type and asserts
        one individual per object row.

        Args:
            data: Populated OCEDStore.
        """
        self.annotations.bind(data.backmap)
        object_types = data.get_object_types()
        self.sink.add_classes(object_types, vocab.OBJECT)
        for obj_type in object_types:
            self.annotations.object_class(obj_type)
        for obj in object_types:
            table = data.get_object(obj)
            self.sink.assert_individuals(list(table.data[table.id]), obj)
            self.annotations.entity_individuals(obj, table.data[table.id])
            self.attributes.emit(table.data, table.id, vocab.OBJECT_ATTRIBUTE)


class TimeEmitter(Emitter):
    """Time individuals and the ``observed_at`` link from each event."""

    def emit(self, data: OCEDStore) -> None:
        """Assert Time individuals and link each event to its timestamp.

        Args:
            data: Populated OCEDStore.
        """
        self.annotations.bind(data.backmap)

        # Only the identifier and timestamp are needed. Concatenating the whole
        # tables instead would union every attribute column across activities and
        # allocate a wide frame of nulls to read two columns out of.
        df_list = [
            data.get_events(activity)
            .data[[data.get_events(activity).id, data.get_events(activity).time]]
            .rename(
                columns={
                    data.get_events(activity).id: "event",
                    data.get_events(activity).time: "time",
                }
            )
            for activity in data.get_activities()
        ]
        df_time = pd.concat(df_list)
        df_time["time_str"] = vocab.timestamp_ids(df_time["time"])
        df_time["time_label"] = timestamp_labels(df_time["time"])
        self.sink.assert_individuals(df_time["time_str"].to_list(), vocab.TIME)
        unique_times = df_time[["time_str", "time_label", "time"]].drop_duplicates(
            "time_str"
        )
        self.annotations.time_individuals(
            unique_times["time_str"], unique_times["time_label"]
        )
        self.sink.assert_relations(
            unique_times,
            prop=vocab.HAS_DATA_VALUE,
            from_key="time_str",
            to_key="time",
            data_prop=True,
        )
        self.sink.assert_relations(
            df_time,
            prop=vocab.OBSERVED_AT,
            from_key="event",
            to_key="time_str",
            inverse_name=vocab.OBSERVES,
        )


class RelationEmitter(Emitter):
    """Event-to-object and object-to-object relation properties and their triples."""

    def emit_event_to_object(self, data: OCEDStore) -> None:
        """Assert all e2o relation properties and their ABox triples."""
        self._emit(data, RelationKind.E2O, vocab.EVENT_OBJECT_RELATION)

    def emit_object_to_object(self, data: OCEDStore) -> None:
        """Assert all o2o relation properties and their ABox triples."""
        self._emit(data, RelationKind.O2O, vocab.OBJECT_OBJECT_RELATION)

    def _emit(
        self,
        data: OCEDStore,
        relation_filter: RelationKind,
        superprop: str,
    ) -> None:
        self.annotations.bind(data.backmap)
        for relation in data.get_relation_types(relation_filter):
            table = data.get_relation(relation)
            from_types = table.from_type or []
            # An o2o relation names its source types; an e2o relation does not,
            # because the source of one is always an event. Both get a domain:
            # leaving e2o without one made the two families inconsistent for no
            # reason other than that the source was implicit.
            domain = from_types or [vocab.EVENT]
            self.sink.add_property(
                relation,
                superprop=superprop,
                domain_class=" or ".join(domain),
                range_class=" or ".join(table.to_type),
                create_inverse=self.settings.has_inverses,
            )
            self.annotations.relation(relation, from_types, table.to_type)
            self.sink.assert_relations(
                table.data,
                relation,
                from_key=table.from_id,
                to_key=table.to_id,
            )


class OrderEmitter(Emitter):
    """Directly-follows, eventually-follows and DECLARE constraint membership."""

    def emit(self, data: OCEDStore) -> None:
        """Dispatch on ``settings.order_representation``; strategies can combine."""
        self.annotations.bind(data.backmap)
        if self.settings.instantiate_traces:
            self.emit_directly_follows(data)
        if self.settings.instantiate_declare:
            self.emit_declare(data)

    def emit_directly_follows(self, data: OCEDStore) -> None:
        """Build DF (and optionally EF) event-order properties for all case concepts."""
        include_ef = self.settings.instantiate_ef
        has_inverses = self.settings.has_inverses

        self.sink.add_property(
            vocab.DF, vocab.EVENT, vocab.EVENT, create_inverse=has_inverses
        )
        if include_ef:
            self.sink.add_property(
                vocab.EF, vocab.EVENT, vocab.EVENT, create_inverse=has_inverses
            )
        self.annotations.order_superproperties(include_ef)

        # One flattening pass for all case concepts rather than one each.
        logs = data.get_logs(self.settings.case_concepts)
        for case_concept, events in logs.items():
            self.emit_case_order(case_concept, events, include_ef)

    def emit_case_order(
        self, case_concept: str, events: pd.DataFrame, include_ef: bool
    ) -> None:
        """Assert DF (and optionally EF) relations for a single case concept.

        Args:
            case_concept: Object type used as the case concept.
            events: Flat event log for ``case_concept``, as returned by
                :meth:`~semanticlog.data.oced.OCEDStore.get_log`.
            include_ef: Whether to include eventually-follows relations.
        """
        events = events.sort_values(
            ["case:concept:name", "time:timestamp"], kind="mergesort"
        )

        events["event_id"] = events.index
        grouped = events.groupby("case:concept:name")
        events["next_event_id"] = grouped["event_id"].shift(-1)

        start_events = grouped.first()
        end_events = grouped.last()

        has_inverses = self.settings.has_inverses
        start_cls = vocab.start_class(case_concept)
        end_cls = vocab.end_class(case_concept)
        self.sink.add_class(start_cls, vocab.EVENT)
        self.sink.add_class(end_cls, vocab.EVENT)
        self.annotations.trace_boundary_classes(case_concept, start_cls, end_cls)
        self.sink.assert_individuals(list(start_events["event_id"]), start_cls)
        self.sink.assert_individuals(list(end_events["event_id"]), end_cls)

        df_pairs = events.loc[
            events["next_event_id"].notna(), ["event_id", "next_event_id"]
        ]

        df_rel = vocab.df_property(case_concept)
        self.sink.add_property(
            df_rel,
            vocab.EVENT,
            vocab.EVENT,
            vocab.DF,
            create_inverse=has_inverses,
        )
        self.annotations.df_property(case_concept, df_rel)
        self.sink.assert_relations(df_pairs, df_rel, "event_id", "next_event_id")

        if not include_ef:
            return

        ef_rel = vocab.ef_property(case_concept)
        self.sink.add_property(
            ef_rel,
            vocab.EVENT,
            vocab.EVENT,
            vocab.EF,
            create_inverse=has_inverses,
        )
        self.annotations.ef_property(case_concept, ef_rel)

        if self.settings.instantiate_ef_axiomatic:
            # validate() guarantees OWL2DL here — EF is rejected for OWL2EL.
            self._axiomatise_eventually_follows(df_rel, ef_rel, events)

        if self.settings.instantiate_ef_explicit:
            self._materialise_eventually_follows(ef_rel, events)

    def _axiomatise_eventually_follows(
        self, df_rel: str, ef_rel: str, events: pd.DataFrame
    ) -> None:
        """Leave eventually-follows for a reasoner to derive from directly-follows.

        ``ef`` is the reflexive transitive closure of ``df``.  Transitivity and
        ``df ⊑ ef`` are TBox axioms, but the reflexive part cannot be: OWL 2 DL
        forbids a self-restriction on a non-simple property, and a transitive
        property is non-simple.  Asserting ``∃event_object_relation.{cc} ⊑ ∃ef.Self``
        alongside transitive ``ef`` puts the ontology outside the profile, and a
        reasoner rejects the file on load rather than at build time.  The
        self-loops are therefore asserted per event in the ABox instead.
        """
        self.sink.add_sub_property(df_rel, ef_rel)
        self.sink.make_transitive(ef_rel)

        self_pairs = pd.DataFrame({"event_id": events["event_id"].unique()})
        self_pairs["event_id_self"] = self_pairs["event_id"]
        self.sink.assert_relations(self_pairs, ef_rel, "event_id", "event_id_self")

    def _materialise_eventually_follows(
        self, ef_rel: str, events: pd.DataFrame
    ) -> None:
        """Assert every eventually-follows pair directly, reflexive pairs included.

        The upper triangle of each trace's time-sorted event list gives all pairs
        (position p <= q within a case iff eventually-follows holds).  The diagonal
        is part of that triangle, so the self-loops come with it and need no
        separate pass.

        Emitted trace by trace rather than as one concatenated frame, so a
        streaming sink never has more than a single trace's pairs in memory.
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
            self.sink.assert_relations(ef_pairs, ef_rel, "event_id_from", "event_id_to")

    def _check_ef_budget(self, ef_rel: str, groups: list) -> None:
        """Refuse a materialisation that is going to be too large to finish.

        The cost is driven by trace length, not event count, so it cannot be
        predicted from the size of the input file. Counting first turns an
        out-of-memory death hours in into an immediate, actionable error.

        Raises:
            ValueError: If the projected pair count exceeds ``max_ef_pairs``.
        """
        projected = sum(len(grp) * (len(grp) + 1) // 2 for _, grp in groups)
        budget = self.settings.max_ef_pairs
        logging.info(
            f"  {ef_rel}: materialising {projected:,} eventually-follows pairs"
        )
        if budget and projected > budget:
            longest = max((len(grp) for _, grp in groups), default=0)
            raise ValueError(
                f"EF_explicit for '{ef_rel}' would materialise {projected:,} pairs, "
                f"over the max_ef_pairs limit of {budget:,}. Eventually-follows is "
                f"quadratic in trace length and the longest trace here has {longest:,} "
                "events. Use EventOrderRepresentation.EF to let a reasoner derive "
                "them instead, choose a case concept with shorter traces, or raise "
                "settings.max_ef_pairs."
            )

    def emit_declare(self, data: OCEDStore) -> None:
        """Assert DECLARE constraint membership for each trace.

        Creates a ``DeclareConstraint`` class hierarchy and types each trace
        individual as a member of the constraints it satisfies. Works for any
        number of case concepts — separation is implicit because trace individuals
        are distinct across case concepts.

        Args:
            data: Populated OCEDStore (used only to satisfy the interface;
                constraint data comes from ``settings.declare_result``).

        Raises:
            ValueError: If no DECLARE constraints are available.
        """
        declare_result = self.settings.declare_result
        if declare_result is None:
            raise ValueError(
                "No DECLARE constraints available. Set settings.declare_file to a "
                "results CSV or assign settings.declare_result directly."
            )

        df_declare = declare_result.constraints
        constraints = [
            c for c in df_declare.columns if c not in ["case_concept", "trace_id"]
        ]

        self.sink.add_class(vocab.DECLARE_CONSTRAINT)
        self.sink.add_classes(constraints, vocab.DECLARE_CONSTRAINT)

        for constraint in constraints:
            trace_ids = df_declare.loc[df_declare[constraint], "trace_id"]
            self.sink.assert_individuals(trace_ids, constraint)
