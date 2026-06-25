from __future__ import annotations

import logging
from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
from owlapy.class_expression import OWLObjectHasSelf, OWLObjectSomeValuesFrom
from owlapy.owl_axiom import (
    OWLSubClassOfAxiom,
    OWLSubPropertyAxiom,
    OWLTransitiveObjectPropertyAxiom,
)
from owlapy.owl_property import OWLObjectProperty

from semanticlog.data.binning import bin_series, equal_bin_series
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.builder import OCEDBaseBuilder
from semanticlog.onto.semlog import SemanticLog
from semanticlog.onto.settings import (
    DataRepresentation,
    EventOrderRepresentation,
    OcelExtension,
    OntoBuildSettings,
    OwlProfile,
    validate_settings,
)
from semanticlog.onto.transform import parse_manchester


class OCEDOntoBuilder(OCEDBaseBuilder):
    """Full OCED-to-OWL pipeline builder: TBox + ABox.

    Constructs the upper ontology (TBox) internally during ``__init__``, then
    populates the ABox via :meth:`build_ontology`.  This is the main entry point
    for converting an :class:`~semanticlog.data.oced.OCEDStore` into a
    :class:`~semanticlog.onto.semlog.SemanticLog`.

    Args:
        iri: Ontology IRI (e.g. ``DEFAULT_IRI``). Not a file path.
        settings: Build configuration controlling which ontology features are generated.
    """

    def __init__(self, iri: str, settings: OntoBuildSettings) -> None:
        from semanticlog.onto.upper import OCEDUpperOntoBuilder

        upper = OCEDUpperOntoBuilder(iri, settings)
        upper.construct_upper_ontology()
        super().__init__(iri, settings, _onto=upper.onto)
        self._named_inverse_map = dict(upper._named_inverse_map)

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
        validate_settings(self.settings)
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
            f"Instantiating event to object relation data ({len(data.get_relation_types(relation_type='e2o'))} relation types).."
        )
        self.instantiate_event_object_relation_data(data)
        logging.info(
            f"Instantiating object to object relation data ({len(data.get_relation_types(relation_type='o2o'))} relation types).."
        )
        self.instantiate_object_to_object_relation_data(data)

        if OcelExtension.ATTRIBUTE_HISTORY in self.settings.ocel_extensions:
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
        activities = data.get_activities()

        self.add_classes(activities, "Event")
        if self.settings.derive_annotations:
            for a in activities:
                orig = data.backmap.get(a, a)
                self.annotate_entity(a, label=orig, comment=f"Events of type '{orig}'.")
        for activity in activities:
            activity_data = data.get_events(activity)
            activity_df = cast(pd.DataFrame, activity_data["data"])
            id_col = cast(str, activity_data["id"])
            self.assert_individuals(list(activity_df[id_col]), activity)
            if self.settings.derive_annotations:
                orig_activity = data.backmap.get(activity, activity)
                self.annotate_individuals_bulk(
                    [
                        (
                            str(eid),
                            f"{orig_activity} ({data.backmap.get(str(eid), str(eid))})",
                        )
                        for eid in activity_df[id_col]
                    ]
                )
            self.instantiate_attributes(
                activity_df.drop(columns=[cast(str, activity_data["time"])]),
                id_col,
                "EventAttribute",
                backmap=data.backmap,
            )

    def instantiate_event_order(self, data: OCEDStore) -> None:
        """Dispatch event order instantiation based on ``settings.order_representation``.

        Calls :meth:`instantiate_event_order_via_directly_follows` when DF, EF,
        or EF_explicit is active, and :meth:`instantiate_event_order_via_declare`
        when DECLARE is active.  Both can be active simultaneously.

        Args:
            data: Populated OCEDStore.
        """
        if (
            EventOrderRepresentation.DF in self.settings.order_representation
            or EventOrderRepresentation.EF in self.settings.order_representation
            or EventOrderRepresentation.EF_explicit
            in self.settings.order_representation
        ):
            self.instantiate_event_order_via_directly_follows(data)

        if EventOrderRepresentation.DECLARE in self.settings.order_representation:
            self.instantiate_event_order_via_declare(data)

    def instantiate_event_order_via_directly_follows(self, data: OCEDStore) -> None:
        """Build DF (and optionally EF) event-order properties for all case concepts.

        Adds the generic ``df`` / ``ef`` super-properties and then delegates
        per-case work to :meth:`instantiate_directly_follows`.

        Args:
            data: Populated OCEDStore.
        """
        include_ef = (
            EventOrderRepresentation.EF in self.settings.order_representation
            or EventOrderRepresentation.EF_explicit
            in self.settings.order_representation
        )
        has_inverses = bool(self.settings.inverse_representation)

        self.add_property("df", "Event", "Event", create_inverse=has_inverses)
        if self.settings.derive_annotations:
            self.annotate_entity(
                "df",
                label="directly followed by",
                comment="Holds between two events when one immediately precedes the other within any case. Supertype of all case-specific directly-follows properties.",
            )
            if has_inverses:
                self.annotate_entity(
                    "inv_df",
                    label="directly preceded by",
                    comment="Relates an event to the event that directly precedes it within any case. Inverse of 'directly followed by'.",
                )

        if include_ef:
            self.add_property("ef", "Event", "Event", create_inverse=has_inverses)
            if self.settings.derive_annotations:
                self.annotate_entity(
                    "ef",
                    label="eventually followed by",
                    comment="Holds between two events whenever one comes at or before the other within any case, including itself. Supertype of all case-specific eventually-follows properties.",
                )
                if has_inverses:
                    self.annotate_entity(
                        "inv_ef",
                        label="eventually preceded by",
                        comment="Relates an event to any event that comes at or before it within any case, including itself. Inverse of 'eventually followed by'.",
                    )

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

        has_inverses = bool(self.settings.inverse_representation)
        start_cls = "StartEvent_" + case_concept
        end_cls = "EndEvent_" + case_concept
        self.add_class(start_cls, "Event")
        self.add_class(end_cls, "Event")
        if self.settings.derive_annotations:
            orig_cc = data.backmap.get(case_concept, case_concept)
            self.annotate_entity(
                start_cls,
                label=f"Start Event (via {orig_cc})",
                comment=f"Events with no directly-follows predecessor in the '{orig_cc}' case trace.",
            )
            self.annotate_entity(
                end_cls,
                label=f"End Event (via {orig_cc})",
                comment=f"Events with no directly-follows successor in the '{orig_cc}' case trace.",
            )
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
        if self.settings.derive_annotations:
            _orig_cc = data.backmap.get(case_concept, case_concept)
            self.annotate_entity(
                df_rel,
                label=f"directly followed (via {_orig_cc}) by",
                comment=f"Holds between two events when one immediately precedes the other in the same '{_orig_cc}' case trace.",
            )
            if has_inverses:
                self.annotate_entity(
                    "inv_" + df_rel,
                    label=f"directly preceded (via {_orig_cc}) by",
                    comment=f"Relates an event to its direct predecessor in the same '{_orig_cc}' case trace. Inverse of 'directly followed (via {_orig_cc}) by'.",
                )
        self.assert_relations(df_pairs, df_rel, "event_id", "next_event_id")

        if include_ef:
            make_explicit = (
                EventOrderRepresentation.EF_explicit
                in self.settings.order_representation
            )

            ef_rel = "ef_" + case_concept
            self.add_property(
                ef_rel,
                "Event",
                "Event",
                "ef",
                create_inverse=has_inverses,
            )
            if self.settings.derive_annotations:
                _orig_cc = data.backmap.get(case_concept, case_concept)
                self.annotate_entity(
                    ef_rel,
                    label=f"eventually followed (via {_orig_cc}) by",
                    comment=f"Holds between two events whenever one comes at or before the other in the same '{_orig_cc}' case trace, including itself.",
                )
                if has_inverses:
                    self.annotate_entity(
                        "inv_" + ef_rel,
                        label=f"eventually preceded (via {_orig_cc}) by",
                        comment=f"Relates an event to any event that comes at or before it in the same '{_orig_cc}' case trace, including itself. Inverse of 'eventually followed (via {_orig_cc}) by'.",
                    )

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

            if make_explicit:
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
            ValueError: If ``settings.declare_result`` is ``None``.
        """
        if self.settings.declare_result is None:
            raise ValueError(
                "settings.declare_result must be set before instantiating DECLARE constraints. "
                "Call settings.load_declare_file() or assign settings.declare_result directly."
            )

        df_declare = self.settings.declare_result.constraints
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
        df_list = []

        for activity in data.get_activities():
            activity_data = data.get_events(activity)
            activity_df = cast(pd.DataFrame, activity_data["data"])
            activity_df = activity_df.rename(
                columns={
                    cast(str, activity_data["id"]): "event",
                    cast(str, activity_data["time"]): "time",
                }
            )
            df_list.append(activity_df)
        df_time = pd.concat(df_list)
        if pd.api.types.is_datetime64_any_dtype(df_time["time"]):
            df_time["time_str"] = df_time["time"].dt.strftime("%Y%m%dT%H%M%S_%f")
            df_time["time_label"] = df_time["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            df_time["time_str"] = df_time["time"].apply(self.generate_timestamp_string)
            df_time["time_label"] = df_time["time"].astype(str)
        str_list = df_time["time_str"].to_list()
        self.assert_individuals(str_list, "Time")
        unique_times = df_time[["time_str", "time_label", "time"]].drop_duplicates(
            "time_str"
        )
        if self.settings.derive_annotations:
            self.annotate_individuals_bulk(
                list(
                    unique_times[["time_str", "time_label"]].itertuples(
                        index=False, name=None
                    )
                )
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
        object_types = data.get_object_types()
        self.add_classes(object_types, "Object")
        if self.settings.derive_annotations:
            for ot in object_types:
                orig = data.backmap.get(ot, ot)
                self.annotate_entity(
                    ot, label=orig, comment=f"Objects of type '{orig}'."
                )
        for obj in object_types:
            object_data = data.get_object(obj)
            object_df = cast(pd.DataFrame, object_data["data"])
            id_col = cast(str, object_data["id"])
            self.assert_individuals(list(object_df[id_col]), obj)
            if self.settings.derive_annotations:
                orig_obj = data.backmap.get(obj, obj)
                self.annotate_individuals_bulk(
                    [
                        (
                            str(oid),
                            f"{orig_obj} ({data.backmap.get(str(oid), str(oid))})",
                        )
                        for oid in object_df[id_col]
                    ]
                )
            self.instantiate_attributes(
                object_df, id_col, "ObjectAttribute", backmap=data.backmap
            )

    def instantiate_attributes(
        self,
        df: pd.DataFrame,
        key_col: str,
        attribute_parent: str,
        backmap: dict[str, str] | None = None,
    ) -> None:
        """Instantiate attribute values for a single event or object type.

        For each non-key column in ``df``: creates an attribute class under
        ``attribute_parent``, adds a ``has_attribute`` sub-property, and asserts
        attribute-value individuals using the configured ``data_representation``
        strategy (values, equal-frequency bins, distribution-fitted bins).

        Args:
            df: Event or object DataFrame with the key column and attribute columns.
            key_col: Column name of the individual's IRI local name.
            attribute_parent: Parent class name (``"EventAttribute"`` or ``"ObjectAttribute"``).
            backmap: Optional cleaned→original name mapping for annotation labels.
        """
        dr = self.settings.data_representation
        for attr in (c for c in df.columns if c != key_col):
            mask = df[attr].notna()
            if not mask.any():
                continue
            df_attr = df.loc[mask, [key_col, attr]]
            s = df_attr[attr]

            if pd.api.types.is_numeric_dtype(s) and not dr:
                continue

            prop = self.generate_specialized_has_attribute_name(attr)
            self.add_class(attr, attribute_parent)
            self.add_property(
                prop,
                superprop="has_attribute",
                create_inverse=bool(self.settings.inverse_representation),
            )
            if self.settings.derive_annotations and backmap is not None:
                orig = backmap.get(attr, attr)
                kind = "event" if attribute_parent == "EventAttribute" else "object"
                self.annotate_entity(
                    attr, label=orig, comment=f"Attribute '{orig}' of a {kind}."
                )
                self.annotate_entity(
                    prop,
                    label=f"has {orig}",
                    comment=f"Links a {kind} to its '{orig}' attribute value.",
                )
                if bool(self.settings.inverse_representation):
                    inv = self.generate_inverse_property_name(prop)
                    self.annotate_entity(
                        inv,
                        label=f"is {orig} of",
                        comment=f"Links a '{orig}' attribute value to its owning {kind}.",
                    )

            if pd.api.types.is_datetime64_any_dtype(s):
                self._instantiate_datetime_attr(df_attr, key_col, attr, prop, backmap)
            elif pd.api.types.is_numeric_dtype(s):
                self._instantiate_numeric_attr(
                    df_attr, key_col, attr, prop, dr, backmap
                )
            else:
                self._instantiate_string_attr(df_attr, key_col, attr, prop, backmap)

    def _instantiate_datetime_attr(
        self,
        df_attr: pd.DataFrame,
        key_col: str,
        attr: str,
        prop: str,
        backmap: dict[str, str] | None = None,
    ) -> None:
        self._instantiate_attr(
            df_attr,
            key_col,
            attr,
            prop,
            value_to_id=lambda v, a=attr: f"{a}_{self.generate_timestamp_string(v)}",
            also_assert_data_value=True,
            backmap=backmap,
        )

    def _instantiate_numeric_attr(
        self,
        df_attr: pd.DataFrame,
        key_col: str,
        attr: str,
        prop: str,
        dr: list[DataRepresentation],
        backmap: dict[str, str] | None = None,
    ) -> None:
        s = df_attr[attr]
        if DataRepresentation.VALUES in dr:
            self._instantiate_attr(
                df_attr,
                key_col,
                attr,
                prop,
                value_to_id=lambda v, a=attr: f"{a}_{v}",
                also_assert_data_value=True,
                backmap=backmap,
            )
        quant = next(
            (
                x
                for x in dr
                if x
                in (
                    DataRepresentation.EQUAL_BINS,
                    DataRepresentation.DIST_BINS,
                )
            ),
            None,
        )
        if quant is None:
            return
        if quant == DataRepresentation.EQUAL_BINS:
            binned = equal_bin_series(
                s,
                bins=self.settings.bins,
                labels=self.settings.bin_labels,
            )
        else:
            binned = bin_series(
                s,
                bins=self.settings.bins,
                labels=self.settings.bin_labels,
            )
        df_merge = df_attr.copy()
        df_merge["_bin"] = binned
        out_of_bin = df_attr.loc[df_merge["_bin"].isna()]
        df_q = df_merge[df_merge["_bin"].notna()]
        for bin_label, bin_group in df_q.groupby("_bin", observed=True):
            bin_class = f"{attr}_{str(bin_label)}"
            self.add_class(bin_class, attr)
            if DataRepresentation.VALUES in dr:
                value_inds = [f"{attr}_{v}" for v in bin_group[attr].unique()]
                self.assert_individuals(value_inds, bin_class)
            else:
                self.assert_individuals(bin_group[key_col].tolist(), bin_class)
        if not out_of_bin.empty:
            logging.warning(
                f"BINNING: {len(out_of_bin)} value(s) for attribute "
                f"'{attr}' fall outside all bin boundaries; using raw "
                "numeric representation as fallback."
            )
            self._instantiate_attr(
                out_of_bin,
                key_col,
                attr,
                prop,
                value_to_id=lambda v, a=attr: f"{a}_{v}",
                also_assert_data_value=True,
                backmap=backmap,
            )

    def _instantiate_string_attr(
        self,
        df_attr: pd.DataFrame,
        key_col: str,
        attr: str,
        prop: str,
        backmap: dict[str, str] | None = None,
    ) -> None:
        self._instantiate_attr(
            df_attr,
            key_col,
            attr,
            prop,
            value_to_id=lambda v, a=attr: f"{a}_{str(v)}",
            also_assert_data_value=False,
            backmap=backmap,
        )

    def _instantiate_attr(
        self,
        df: pd.DataFrame,
        key_col: str,
        attr: str,
        prop: str,
        value_to_id: Callable[..., str],
        also_assert_data_value: bool,
        backmap: dict[str, str] | None = None,
    ) -> None:
        tmp = df.copy()

        # map each distinct value to an attribute-individual IRI/name
        vals = tmp[attr].unique()
        mapping = {v: value_to_id(v) for v in vals}
        tmp["attr_ind"] = tmp[attr].map(mapping)

        # create individuals for the attribute values (unique only — one per distinct value)
        self.assert_individuals(tmp["attr_ind"].unique().tolist(), attr)

        if self.settings.derive_annotations:
            bm = backmap or {}
            self.annotate_individuals_bulk(
                [
                    (
                        iid,
                        bm.get(
                            iid,
                            v.strftime("%Y-%m-%d %H:%M:%S")
                            if isinstance(v, pd.Timestamp)
                            else str(v),
                        ),
                    )
                    for v, iid in mapping.items()
                ]
            )

        # connect objects/events (key_col) to attribute-individuals via the property
        self.assert_relations(tmp, prop=prop, from_key=key_col, to_key="attr_ind")

        # for numeric/datetime, also attach the literal value
        if also_assert_data_value:
            self.assert_relations(
                tmp,
                prop="has_data_value",
                from_key="attr_ind",
                to_key=attr,
                data_prop=True,
            )

    def _instantiate_relation_data(
        self,
        data: OCEDStore,
        relation_filter: str,
        superprop: str,
        include_domain: bool = False,
    ) -> None:
        for relation in data.get_relation_types(relation_type=relation_filter):
            relation_data = data.get_relation(relation)
            self.add_property(
                relation,
                superprop=superprop,
                domain_class=(
                    " or ".join(relation_data["from_type"]) if include_domain else None
                ),
                range_class=" or ".join(relation_data["to_type"]),
                create_inverse=bool(self.settings.inverse_representation),
            )
            if self.settings.derive_annotations:
                orig = data.backmap.get(relation, relation)
                raw_from = relation_data.get("from_type", [])
                from_types = (
                    " or ".join(data.backmap.get(t, t) for t in raw_from)
                    if raw_from
                    else "event"
                )
                to_types = " or ".join(
                    data.backmap.get(t, t) for t in relation_data["to_type"]
                )
                self.annotate_entity(
                    relation,
                    label=orig,
                    comment=f"Relates '{from_types}' to '{to_types}' via the '{orig}' relationship.",
                )
                if bool(self.settings.inverse_representation):
                    inv = self.generate_inverse_property_name(relation)
                    self.annotate_entity(
                        inv,
                        label=f"inverse of {orig}",
                        comment=f"Inverse of '{orig}': relates '{to_types}' back to '{from_types}'.",
                    )
            self.assert_relations(
                relation_data["data"],
                relation,
                from_key=relation_data["from_id"],
                to_key=relation_data["to_id"],
            )

    def instantiate_event_object_relation_data(self, data: OCEDStore) -> None:
        """Assert all e2o relation properties and their ABox triples.

        Args:
            data: Populated OCEDStore.
        """
        self._instantiate_relation_data(data, "e2o", "event_object_relation")

    def instantiate_object_to_object_relation_data(self, data: OCEDStore) -> None:
        """Assert all o2o relation properties and their ABox triples.

        Args:
            data: Populated OCEDStore.
        """
        self._instantiate_relation_data(
            data, "o2o", "object_object_relation", include_domain=True
        )

    def instantiate_object_attribute_history(self, data: OCEDStore) -> None:
        """Instantiate object attribute change history as AttributeChange individuals.

        For each change record ``(ocel:oid, ocel:field, value, ocel:timestamp)``,
        creates one ``AttributeChange`` individual and asserts:

        - ``has_attribute_change``: object → AttributeChange
        - ``attribute_change_value``: AttributeChange → attribute-value individual
        - ``changed_at``: AttributeChange → Time individual

        Attribute-value individuals are linked via a per-attribute subproperty of
        ``attribute_change_value`` (e.g., ``change_value_price``).  Datetime attributes
        and numeric attributes with ``DataRepresentation.VALUES`` also get a
        ``has_data_value`` literal.  Binning modes (``EQUAL_BINS``,
        ``DIST_BINS``) are intentionally ignored for changes — only the exact
        numeric value is recorded.

        Args:
            data: OCEDStore created with ``ocel_mode=True`` that contains object
                change history loaded via ``add_object_changes``.
        """
        has_inverses = bool(self.settings.inverse_representation)
        for obj_type in data.get_object_change_types():
            changes_df = data.get_object_changes(obj_type)

            for attr_name, grp in changes_df.groupby("ocel:field"):
                attr_name = str(attr_name)
                if attr_name not in grp.columns:
                    logging.warning(
                        f"ATTRIBUTE_HISTORY: ocel:field value '{attr_name}' has no "
                        f"matching column in object_changes for type '{obj_type}' — "
                        "skipping. This indicates unexpected object_changes structure."
                    )
                    continue

                self.add_class(attr_name, "ObjectAttribute")
                attr_change_cls = attr_name + "Change"
                self.add_class(attr_change_cls, "AttributeChange")
                change_prop = self.generate_attr_change_value_prop_name(attr_name)
                self.add_property(
                    change_prop,
                    superprop="attribute_change_value",
                    range_class=attr_name,
                    create_inverse=has_inverses,
                )
                if self.settings.derive_annotations:
                    orig_attr = data.backmap.get(attr_name, attr_name)
                    self.annotate_entity(
                        attr_name,
                        label=orig_attr,
                        comment=f"Attribute '{orig_attr}' of an object, tracking changes over time.",
                    )
                    self.annotate_entity(
                        attr_change_cls,
                        label=f"{orig_attr} Change",
                        comment=f"A timestamped record of the attribute '{orig_attr}' taking a new value.",
                    )
                    self.annotate_entity(
                        change_prop,
                        label=f"changes {orig_attr} to",
                        comment=f"Links an attribute change record to the new value of '{orig_attr}'.",
                    )
                    if has_inverses:
                        self.annotate_entity(
                            self.generate_inverse_property_name(change_prop),
                            label=f"is new {orig_attr} in change",
                            comment=f"Links a '{orig_attr}' value to the attribute change record in which it became the new value.",
                        )

                tmp = grp[["ocel:oid", "ocel:timestamp", attr_name]].copy()
                tmp = tmp[tmp[attr_name].notna()].copy()
                if tmp.empty:
                    continue

                s = tmp[attr_name]
                if pd.api.types.is_numeric_dtype(s):
                    if (
                        DataRepresentation.VALUES
                        not in self.settings.data_representation
                    ):
                        continue

                tmp["ts_str"] = tmp["ocel:timestamp"].apply(
                    self.generate_timestamp_string
                )
                tmp["attr_change_id"] = (
                    "AttrChange_"
                    + tmp["ocel:oid"].astype(str)
                    + "_"
                    + attr_name
                    + "_"
                    + tmp["ts_str"]
                )
                tmp["time_id"] = tmp["ts_str"]

                self.assert_individuals(tmp["attr_change_id"].tolist(), attr_change_cls)
                if self.settings.derive_annotations:
                    ts_labels = (
                        tmp["ocel:timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
                        if pd.api.types.is_datetime64_any_dtype(tmp["ocel:timestamp"])
                        else tmp["ocel:timestamp"].astype(str)
                    )
                    obj_labels = obj_type + " (" + tmp["ocel:oid"].astype(str) + ")"
                    self.annotate_individuals_bulk(
                        list(
                            zip(
                                tmp["attr_change_id"],
                                f"{attr_name} Change for "
                                + obj_labels
                                + " at "
                                + ts_labels,
                            )
                        )
                    )
                self.assert_relations(
                    tmp,
                    "has_attribute_change",
                    from_key="ocel:oid",
                    to_key="attr_change_id",
                )
                s = tmp[attr_name]
                if pd.api.types.is_datetime64_any_dtype(s):
                    self._instantiate_attr(
                        tmp,
                        "attr_change_id",
                        attr_name,
                        change_prop,
                        lambda v, a=attr_name: (
                            f"{a}_{self.generate_timestamp_string(v)}"
                        ),
                        also_assert_data_value=True,
                    )
                elif pd.api.types.is_numeric_dtype(s):
                    self._instantiate_attr(
                        tmp,
                        "attr_change_id",
                        attr_name,
                        change_prop,
                        lambda v, a=attr_name: f"{a}_{v}",
                        also_assert_data_value=True,
                    )
                else:
                    self._instantiate_attr(
                        tmp,
                        "attr_change_id",
                        attr_name,
                        change_prop,
                        lambda v, a=attr_name: f"{a}_{str(v)}",
                        also_assert_data_value=False,
                    )
                self.assert_individuals(tmp["time_id"].unique().tolist(), "Time")
                unique_change_times = tmp[
                    ["time_id", "ocel:timestamp"]
                ].drop_duplicates("time_id")
                if self.settings.derive_annotations:
                    ts_labels = (
                        unique_change_times["ocel:timestamp"].dt.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        if pd.api.types.is_datetime64_any_dtype(
                            unique_change_times["ocel:timestamp"]
                        )
                        else unique_change_times["ocel:timestamp"].astype(str)
                    )
                    self.annotate_individuals_bulk(
                        list(zip(unique_change_times["time_id"], ts_labels))
                    )
                self.assert_relations(
                    unique_change_times,
                    "has_data_value",
                    from_key="time_id",
                    to_key="ocel:timestamp",
                    data_prop=True,
                )
                self.assert_relations(
                    tmp, "changed_at", from_key="attr_change_id", to_key="time_id"
                )
