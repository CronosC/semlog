from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from owlapy.class_expression import OWLObjectHasSelf, OWLObjectSomeValuesFrom
from owlapy.owl_axiom import (
    OWLSubClassOfAxiom,
    OWLSubPropertyAxiom,
    OWLTransitiveObjectPropertyAxiom,
)
from owlapy.owl_property import OWLObjectProperty

from semanticlog.data.oced import OCEDStore, RelationKind
from semanticlog.onto.annotations import timestamp_labels
from semanticlog.onto.history import AttributeHistoryMixin
from semanticlog.onto.semlog import SemanticLog
from semanticlog.onto.settings import OntoBuildSettings, OwlProfile
from semanticlog.onto.transform import parse_manchester
from semanticlog.onto.upper import OCEDUpperOntoBuilder


class OCEDOntoBuilder(AttributeHistoryMixin, OCEDUpperOntoBuilder):
    """Full OCED-to-OWL pipeline builder: TBox + ABox.

    Constructs the upper ontology (TBox) during ``__init__``, then populates the
    ABox via :meth:`build_ontology`.  This is the main entry point for converting
    an :class:`~semanticlog.data.oced.OCEDStore` into a
    :class:`~semanticlog.onto.semlog.SemanticLog`.

    Args:
        iri: Ontology IRI (e.g. ``DEFAULT_IRI``). Not a file path.
        settings: Build configuration controlling which ontology features are generated.
    """

    def __init__(self, iri: str, settings: OntoBuildSettings) -> None:
        super().__init__(iri, settings)
        self.construct_upper_ontology()

    def build_ontology(self, data: OCEDStore) -> SemanticLog:
        """Build the full ontology (TBox already done in __init__) and return a SemanticLog.

        Runs all ABox instantiation steps in order: events, event order, time,
        objects, e2o relations, o2o relations, and optionally attribute history.
        Applies the annotation document last if one is configured.

        Args:
            data: Cleaned :class:`~semanticlog.data.oced.OCEDStore`. Call
                :func:`~semanticlog.data.cleaning.clean_oced_data` before passing.

        Returns:
            :class:`~semanticlog.onto.semlog.SemanticLog` containing the built ontology.
        """
        self.settings.validate()
        self.annotations.bind(data.backmap)
        logging.info("Instantiating ontology..")
        logging.info(
            f"Instantiating event data ({len(data.get_activities())} activities).."
        )
        self.instantiate_event_data(data)
        logging.info(
            f"Instantiating event order via the strategy: {self.settings.order_representation}.."
        )
        self.instantiate_event_order(data)
        logging.info("Instantiating time data..")
        self.instantiate_time_data(data)
        logging.info(
            f"Instantiating object data ({len(data.get_object_types())} objects).."
        )
        self.instantiate_object_data(data)
        logging.info(
            f"Instantiating event to object relation data ({len(data.get_relation_types(RelationKind.E2O))} relation types).."
        )
        self.instantiate_event_object_relation_data(data)
        logging.info(
            f"Instantiating object to object relation data ({len(data.get_relation_types(RelationKind.O2O))} relation types).."
        )
        self.instantiate_object_to_object_relation_data(data)

        if self.settings.instantiate_attribute_history:
            if data.has_ocel_mode():
                logging.info("Instantiating object attribute change history..")
                self.instantiate_object_attribute_history(data)
            else:
                logging.warning(
                    "ATTRIBUTE_HISTORY extension enabled but OCEDStore was not "
                    "created with ocel_mode=True. Skipping attribute history."
                )

        self.apply_annotation_document()
        logging.info("done.")
        return SemanticLog(onto=self.onto, settings=self.settings, iri=self.iri)

    def instantiate_event_data(self, data: OCEDStore) -> None:
        """Assert all event individuals and their attribute values.

        Creates an OWL subclass of ``Event`` for each activity, asserts one
        individual per event row, and delegates attribute instantiation to
        :meth:`instantiate_attributes`.

        Args:
            data: Populated OCEDStore.
        """
        self.annotations.bind(data.backmap)
        activities = data.get_activities()

        self.add_classes(activities, "Event")
        for activity in activities:
            self.annotations.activity_class(activity)
        for activity in activities:
            table = data.get_events(activity)
            self.assert_individuals(list(table.data[table.id]), activity)
            self.annotations.entity_individuals(activity, table.data[table.id])
            self.instantiate_attributes(
                table.data.drop(columns=[table.time]),
                table.id,
                "EventAttribute",
            )

    def instantiate_event_order(self, data: OCEDStore) -> None:
        """Dispatch event order instantiation based on ``settings.order_representation``.

        Calls :meth:`instantiate_event_order_via_directly_follows` when DF, EF,
        or EF_explicit is active, and :meth:`instantiate_event_order_via_declare`
        when DECLARE is active.  Both can be active simultaneously.

        Args:
            data: Populated OCEDStore.
        """
        self.annotations.bind(data.backmap)
        if self.settings.instantiate_traces:
            self.instantiate_event_order_via_directly_follows(data)

        if self.settings.instantiate_declare:
            self.instantiate_event_order_via_declare(data)

    def instantiate_event_order_via_directly_follows(self, data: OCEDStore) -> None:
        """Build DF (and optionally EF) event-order properties for all case concepts.

        Adds the generic ``df`` / ``ef`` super-properties and then delegates
        per-case work to :meth:`instantiate_directly_follows`.

        Args:
            data: Populated OCEDStore.
        """
        include_ef = self.settings.instantiate_ef
        has_inverses = self.settings.has_inverses

        self.add_property("df", "Event", "Event", create_inverse=has_inverses)
        if include_ef:
            self.add_property("ef", "Event", "Event", create_inverse=has_inverses)
        self.annotations.order_superproperties(include_ef)

        for case_concept in self.settings.case_concepts:
            self.instantiate_directly_follows(case_concept, data, include_ef)

    def instantiate_directly_follows(
        self, case_concept: str, data: OCEDStore, include_ef: bool
    ) -> None:
        """Assert DF (and optionally EF) relations for a single case concept.

        Adds case-specific sub-properties of ``df`` / ``ef``, classifies
        start/end events, and asserts all DF pairs.  When ``include_ef`` is True
        and the profile is OWL2DL, adds reflexive/transitive TBox axioms for EF;
        when ``EF_explicit`` is active, materialises all EF pairs directly.

        Args:
            case_concept: Object type used as the case concept.
            data: Populated OCEDStore.
            include_ef: Whether to include eventually-follows relations.
        """
        events = data.get_log(case_concept)
        events = events.sort_values(
            ["case:concept:name", "time:timestamp"], kind="mergesort"
        )

        events["event_id"] = events.index
        grouped = events.groupby("case:concept:name")
        events["next_event_id"] = grouped["event_id"].shift(-1)

        start_events = grouped.first()
        end_events = grouped.last()

        has_inverses = self.settings.has_inverses
        start_cls = "StartEvent_" + case_concept
        end_cls = "EndEvent_" + case_concept
        self.add_class(start_cls, "Event")
        self.add_class(end_cls, "Event")
        self.annotations.trace_boundary_classes(case_concept, start_cls, end_cls)
        self.assert_individuals(list(start_events["event_id"]), start_cls)
        self.assert_individuals(list(end_events["event_id"]), end_cls)

        df_pairs = events.loc[
            events["next_event_id"].notna(), ["event_id", "next_event_id"]
        ]

        df_rel = "df_" + case_concept
        self.add_property(
            df_rel,
            "Event",
            "Event",
            "df",
            create_inverse=has_inverses,
        )
        self.annotations.df_property(case_concept, df_rel)
        self.assert_relations(df_pairs, df_rel, "event_id", "next_event_id")

        if include_ef:
            ef_rel = "ef_" + case_concept
            self.add_property(
                ef_rel,
                "Event",
                "Event",
                "ef",
                create_inverse=has_inverses,
            )
            self.annotations.ef_property(case_concept, ef_rel)

            if self.settings.profile == OwlProfile.OWL2DL:
                # ef as reflexive, transitive closure of df only if profile is DL
                related_to_cc = OWLObjectSomeValuesFrom(
                    OWLObjectProperty(self.iri + "event_object_relation"),
                    parse_manchester(self.iri, case_concept),
                )
                locally_reflexive = OWLObjectHasSelf(
                    OWLObjectProperty(self.iri + ef_rel)
                )
                # events related to a case-concept object are ef themselves.
                self.onto.add_axiom(
                    OWLSubClassOfAxiom(related_to_cc, locally_reflexive)
                )
                # make df subclass of ef
                self.onto.add_axiom(
                    OWLSubPropertyAxiom(  # type: ignore[abstract]
                        OWLObjectProperty(self.iri + df_rel),
                        OWLObjectProperty(self.iri + ef_rel),
                    )
                )
                # make ef transitive
                self.onto.add_axiom(
                    OWLTransitiveObjectPropertyAxiom(
                        OWLObjectProperty(self.iri + ef_rel)
                    )
                )

            if self.settings.instantiate_ef_explicit:
                # Upper triangle of the sorted event matrix gives all EF pairs
                # (position p <= q within a case iff eventually-follows holds).
                ef_from_lists: list[np.ndarray] = []
                ef_to_lists: list[np.ndarray] = []
                for _, grp in events.groupby("case:concept:name", sort=False):
                    ids = grp["event_id"].values
                    n = len(ids)
                    if n == 0:
                        continue
                    row_idx, col_idx = np.triu_indices(n)
                    ef_from_lists.append(ids[row_idx])
                    ef_to_lists.append(ids[col_idx])
                if ef_from_lists:
                    ef_pairs = pd.DataFrame(
                        {
                            "event_id_from": np.concatenate(ef_from_lists),
                            "event_id_to": np.concatenate(ef_to_lists),
                        }
                    )
                    self.assert_relations(
                        ef_pairs, ef_rel, "event_id_from", "event_id_to"
                    )

    def instantiate_event_order_via_declare(self, data: OCEDStore) -> None:
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

        self.add_class("DeclareConstraint")
        self.add_classes(constraints, "DeclareConstraint")

        for constraint in constraints:
            trace_ids = df_declare.loc[df_declare[constraint], "trace_id"]
            self.assert_individuals(trace_ids, constraint)

    def instantiate_time_data(self, data: OCEDStore) -> None:
        """Assert Time individuals and link each event to its timestamp.

        Collects timestamps across all activities, creates one ``Time`` individual
        per distinct timestamp, and asserts ``observed_at`` / ``has_data_value``
        relations.

        Args:
            data: Populated OCEDStore.
        """
        self.annotations.bind(data.backmap)
        df_list = []

        for activity in data.get_activities():
            table = data.get_events(activity)
            df_list.append(
                table.data.rename(columns={table.id: "event", table.time: "time"})
            )
        df_time = pd.concat(df_list)
        if pd.api.types.is_datetime64_any_dtype(df_time["time"]):
            df_time["time_str"] = df_time["time"].dt.strftime("%Y%m%dT%H%M%S_%f")
        else:
            df_time["time_str"] = df_time["time"].apply(self.generate_timestamp_string)
        df_time["time_label"] = timestamp_labels(df_time["time"])
        str_list = df_time["time_str"].to_list()
        self.assert_individuals(str_list, "Time")
        unique_times = df_time[["time_str", "time_label", "time"]].drop_duplicates(
            "time_str"
        )
        self.annotations.time_individuals(
            unique_times["time_str"], unique_times["time_label"]
        )
        self.assert_relations(
            unique_times,
            prop="has_data_value",
            from_key="time_str",
            to_key="time",
            data_prop=True,
        )
        self.assert_relations(
            df_time,
            prop="observed_at",
            from_key="event",
            to_key="time_str",
            inverse_name="observes",
        )

    def instantiate_object_data(self, data: OCEDStore) -> None:
        """Assert all object individuals and their attribute values.

        Creates an OWL subclass of ``Object`` for each object type, asserts
        one individual per object row, and delegates attribute instantiation to
        :meth:`instantiate_attributes`.

        Args:
            data: Populated OCEDStore.
        """
        self.annotations.bind(data.backmap)
        object_types = data.get_object_types()
        self.add_classes(object_types, "Object")
        for obj_type in object_types:
            self.annotations.object_class(obj_type)
        for obj in object_types:
            table = data.get_object(obj)
            self.assert_individuals(list(table.data[table.id]), obj)
            self.annotations.entity_individuals(obj, table.data[table.id])
            self.instantiate_attributes(table.data, table.id, "ObjectAttribute")

    def _instantiate_relation_data(
        self,
        data: OCEDStore,
        relation_filter: RelationKind,
        superprop: str,
        include_domain: bool = False,
    ) -> None:
        self.annotations.bind(data.backmap)
        for relation in data.get_relation_types(relation_filter):
            table = data.get_relation(relation)
            from_types = table.from_type or []
            self.add_property(
                relation,
                superprop=superprop,
                domain_class=" or ".join(from_types) if include_domain else None,
                range_class=" or ".join(table.to_type),
                create_inverse=self.settings.has_inverses,
            )
            self.annotations.relation(relation, from_types, table.to_type)
            self.assert_relations(
                table.data,
                relation,
                from_key=table.from_id,
                to_key=table.to_id,
            )

    def instantiate_event_object_relation_data(self, data: OCEDStore) -> None:
        """Assert all e2o relation properties and their ABox triples.

        Args:
            data: Populated OCEDStore.
        """
        self._instantiate_relation_data(data, RelationKind.E2O, "event_object_relation")

    def instantiate_object_to_object_relation_data(self, data: OCEDStore) -> None:
        """Assert all o2o relation properties and their ABox triples.

        Args:
            data: Populated OCEDStore.
        """
        self._instantiate_relation_data(
            data, RelationKind.O2O, "object_object_relation", include_domain=True
        )
