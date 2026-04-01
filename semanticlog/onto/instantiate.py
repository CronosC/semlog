import logging
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

from semanticlog.data.misc import quantize_series, simple_quantize
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.builder import OCEDBaseBuilder
from semanticlog.onto.settings import (
    DataRepresentation,
    EventOrderRepresentation,
    OcelExtension,
    OntoBuildSettings,
    OwlProfile,
)
from semanticlog.onto.transform import parse_manchester


class OCEDOntoBuilder(OCEDBaseBuilder):

    def __init__(self, iri: str, settings: OntoBuildSettings):
        super().__init__(iri, settings)

    def build_ontology(self, data: OCEDStore) -> None:
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

        if OcelExtension.RELATION_TIMESTAMPS in self.settings.ocel_extensions:
            logging.info("Instantiating e2o relation timestamps..")
            self.instantiate_e2o_relation_timestamps(data)

        logging.info("done.")

    def instantiate_event_data(self, data: OCEDStore) -> None:
        activities = data.get_activities()

        self.add_classes(activities, "Event")
        for activity in activities:
            activity_data = data.get_events(activity)
            activity_df = cast(pd.DataFrame, activity_data["data"])
            self.assert_individuals(
                list(activity_df[cast(str, activity_data["id"])]), activity
            )
            self.instantiate_attributes(
                activity_df.drop(columns=[cast(str, activity_data["time"])]),
                cast(str, activity_data["id"]),
            )

    def instantiate_event_order(self, data: OCEDStore) -> None:
        self.add_class("Trace")
        self.add_property(
            "part_of", "Event", "Trace", create_inverse=self.settings.inverse_properties
        )
        for case_concept in self.settings.case_concepts:
            self.add_class("Trace_" + case_concept, "Trace")
            events = data.get_log(case_concept)
            events["event_id"] = events.index
            self.assert_individuals(
                events["case:concept:name"].unique(), "Trace_" + case_concept
            )
            self.assert_relations(
                events, "part_of", from_key="event_id", to_key="case:concept:name"
            )

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
        include_ef = (
            EventOrderRepresentation.EF in self.settings.order_representation
            or EventOrderRepresentation.EF_explicit
            in self.settings.order_representation
        )

        self.add_property(
            "df", "Event", "Event", create_inverse=self.settings.inverse_properties
        )
        self.add_classes(["StartEvent", "EndEvent"], "Event")

        if include_ef:
            self.add_property(
                "ef", "Event", "Event", create_inverse=self.settings.inverse_properties
            )

        for case_concept in self.settings.case_concepts:
            self.instantiate_directly_follows(case_concept, data, include_ef)

    def instantiate_directly_follows(self, case_concept, data, include_ef):
        events = data.get_log(case_concept)
        events = events.sort_values(
            ["case:concept:name", "time:timestamp"], kind="mergesort"
        )

        events["event_id"] = events.index
        events["_event_order"] = range(len(events))
        grouped = events.groupby("case:concept:name")
        events["next_event_id"] = grouped["event_id"].shift(-1)

        start_events = grouped.first()
        end_events = grouped.last()
        self.assert_individuals(list(start_events["event_id"]), "StartEvent")
        self.assert_individuals(list(end_events["event_id"]), "EndEvent")

        df_pairs = events.loc[
            events["next_event_id"].notna(), ["event_id", "next_event_id"]
        ]

        df_rel = "df_" + case_concept
        self.add_property(
            df_rel,
            "Event",
            "Event",
            "df",
            create_inverse=self.settings.inverse_properties,
        )
        self.assert_relations(df_pairs, df_rel, "event_id", "next_event_id")

        if include_ef:
            make_explicit = (
                EventOrderRepresentation.EF_explicit
                in self.settings.order_representation
            )
            if not make_explicit and self.settings.profile == OwlProfile.OWL2EL:
                raise ValueError(
                    "Can not include ef relations that are not explicit in the OWL2EL profile!"
                )

            ef_rel = "ef_" + case_concept
            self.add_property(
                ef_rel,
                "Event",
                "Event",
                "ef",
                create_inverse=self.settings.inverse_properties,
            )

            if self.settings.profile == OwlProfile.OWL2DL:
                # ef as reflexive, transitive closure of df only if profile is DL
                part_of_trace = OWLObjectSomeValuesFrom(
                    OWLObjectProperty(self.iri + "part_of"),
                    parse_manchester(self.iri, f"Trace_{case_concept}"),
                )
                locally_reflexive = OWLObjectHasSelf(
                    OWLObjectProperty(self.iri + ef_rel)
                )
                # make all events in such a trace ef themselves.
                self.onto.add_axiom(
                    OWLSubClassOfAxiom(part_of_trace, locally_reflexive)
                )
                # make df subclass of ef
                self.onto.add_axiom(
                    OWLSubPropertyAxiom(
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
                # Generate EF pairs per case without a Cartesian self-merge.
                # After sort_values(["case:concept:name", "time:timestamp"]),
                # position p <= q within a case iff the EF condition holds,
                # so the upper triangle (np.triu_indices) gives the correct pairs.
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
        if self.settings.declare_file is None:
            raise ValueError(
                "Need a declare file when instantiating declare constraints! But was given None."
            )

        df_declare = pd.read_csv(self.settings.declare_file)
        if df_declare["case_concept"].nunique() != 1:
            raise NotImplementedError(
                "Multiple declare case concept representation is not implemented yet!"
            )

        for case_concept in df_declare["case_concept"].unique():

            constraints = [
                c for c in df_declare.columns if c not in ["case_concept", "trace_id"]
            ]

            self.add_class("Trace_" + case_concept, "Trace")
            self.add_class("DeclareConstraint")
            self.add_classes(constraints, "DeclareConstraint")

            for constraint in constraints:
                trace_ids = df_declare.loc[df_declare[constraint], "trace_id"]
                self.assert_individuals(trace_ids, constraint)

    def instantiate_time_data(self, data: OCEDStore) -> None:
        df_list = []

        for activity in data.get_activities():
            activity_data = data.get_event(activity)
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
        else:
            df_time["time_str"] = df_time["time"].apply(self.generate_timestamp_string)
        str_list = df_time["time_str"].to_list()
        self.assert_individuals(str_list, "Time")
        self.assert_relations(
            df_time, prop="observed_at", from_key="event", to_key="time_str"
        )

    def instantiate_object_data(self, data: OCEDStore) -> None:
        object_types = data.get_object_types()
        self.add_classes(object_types, "Object")
        for obj in object_types:
            object_data = data.get_object(obj)
            object_df = cast(pd.DataFrame, object_data["data"])
            self.assert_individuals(list(object_df[cast(str, object_data["id"])]), obj)
            self.instantiate_attributes(object_df, cast(str, object_data["id"]))

    def instantiate_attributes(self, df: pd.DataFrame, key_col: str) -> None:
        for attr in (c for c in df.columns if c != key_col):
            mask = df[attr].notna()
            if not mask.any():
                continue
            df_attr = df.loc[mask, [key_col, attr]]
            s = df_attr[attr]

            if pd.api.types.is_numeric_dtype(s):
                if self.settings.data_representation == DataRepresentation.NONE:
                    continue
            elif not pd.api.types.is_datetime64_any_dtype(s):
                pass  # string path — always processed

            prop = self.generate_spezialized_has_attribute_name(attr)
            self.add_class(attr, "Attribute")
            self.add_property(prop, superprop="has_attribute")

            if pd.api.types.is_datetime64_any_dtype(s):
                self._instantiate_attr(
                    df_attr,
                    key_col,
                    attr,
                    prop,
                    value_to_id=lambda v: f"{attr}_{self.generate_timestamp_string(v)}",
                    also_assert_data_value=True,
                )
            elif pd.api.types.is_numeric_dtype(s):
                dr = self.settings.data_representation
                if dr == DataRepresentation.VALUES:
                    self._instantiate_attr(
                        df_attr,
                        key_col,
                        attr,
                        prop,
                        value_to_id=lambda v: f"{attr}_{v}",
                        also_assert_data_value=True,
                    )
                else:
                    if dr == DataRepresentation.QUANTIZE_SIMPLE:
                        binned = simple_quantize(
                            s,
                            bins=self.settings.quantize_bins,
                            labels=self.settings.quantize_labels,
                        )
                    else:  # QUANTIZE_DIST
                        binned = quantize_series(
                            s,
                            bins=self.settings.quantize_bins,
                            labels=self.settings.quantize_labels,
                        )
                    df_q = df_attr.copy()
                    df_q[attr] = binned
                    df_q = df_q[df_q[attr].notna()]
                    if df_q.empty:
                        continue
                    self._instantiate_attr(
                        df_q,
                        key_col,
                        attr,
                        prop,
                        value_to_id=lambda v: f"{attr}_{str(v)}",
                        also_assert_data_value=False,
                    )
            else:
                # treat object/string/categorical/bool as string labels
                self._instantiate_attr(
                    df_attr,
                    key_col,
                    attr,
                    prop,
                    value_to_id=lambda v: f"{attr}_{str(v)}",
                    also_assert_data_value=False,
                )

    def _instantiate_attr(
        self,
        df: pd.DataFrame,
        key_col: str,
        attr: str,
        prop: str,
        value_to_id,
        also_assert_data_value: bool,
    ) -> None:
        tmp = df.copy()

        # map each distinct value to an attribute-individual IRI/name
        vals = tmp[attr].unique()
        mapping = {v: value_to_id(v) for v in vals}
        tmp["attr_ind"] = tmp[attr].map(mapping)

        # create individuals for the attribute values (unique only — one per distinct value)
        self.assert_individuals(tmp["attr_ind"].unique().tolist(), attr)

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

    def instantiate_event_object_relation_data(self, data: OCEDStore) -> None:
        event_object_relations = data.get_relation_types(relation_type="e2o")

        for relation in event_object_relations:
            relation_data = data.get_relation(relation)
            relation_df = relation_data["data"]

            self.add_property(
                relation,
                superprop="event_object_relation",
                range_class=" or ".join(relation_data["to_type"]),
                create_inverse=self.settings.inverse_properties,
            )
            self.assert_relations(
                relation_df,
                relation,
                from_key=relation_data["from_id"],
                to_key=relation_data["to_id"],
            )

    def instantiate_object_to_object_relation_data(self, data: OCEDStore) -> None:
        object_object_relations = data.get_relation_types(relation_type="o2o")

        for relation in object_object_relations:
            relation_data = data.get_relation(relation)
            relation_df = relation_data["data"]

            self.add_property(
                relation,
                superprop="object_object_relation",
                domain_class=" or ".join(relation_data["from_type"]),
                range_class=" or ".join(relation_data["to_type"]),
                create_inverse=self.settings.inverse_properties,
            )
            self.assert_relations(
                relation_df,
                relation,
                from_key=relation_data["from_id"],
                to_key=relation_data["to_id"],
            )

    def instantiate_object_attribute_history(self, data: OCEDStore) -> None:
        """Instantiate object attribute change history as AttributeChange individuals.

        For each change record ``(ocel:oid, ocel:field, value, ocel:timestamp)``,
        creates one ``AttributeChange`` individual and asserts:

        - ``has_attribute_change``: object → AttributeChange
        - ``attribute_change_value``: AttributeChange → attribute-value individual
        - ``changed_at``: AttributeChange → Time individual

        Args:
            data: OCEDStore created with ``ocel_mode=True`` that contains object
                change history loaded via ``add_object_changes``.
        """
        for obj_type in data.get_object_change_types():
            changes_df = data.get_object_changes(obj_type)

            for attr_name, grp in changes_df.groupby("ocel:field"):
                attr_name = str(attr_name)
                if attr_name not in grp.columns:
                    continue

                self.add_class(attr_name, "Attribute")
                prop = self.generate_spezialized_has_attribute_name(attr_name)
                self.add_property(prop, superprop="has_attribute")

                tmp = grp[["ocel:oid", "ocel:timestamp", attr_name]].copy()
                tmp = tmp[tmp[attr_name].notna()].copy()
                if tmp.empty:
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
                tmp["attr_val_id"] = attr_name + "_" + tmp[attr_name].astype(str)
                tmp["time_id"] = tmp["ts_str"]

                self.assert_individuals(
                    tmp["attr_change_id"].tolist(), "AttributeChange"
                )
                self.assert_relations(
                    tmp,
                    "has_attribute_change",
                    from_key="ocel:oid",
                    to_key="attr_change_id",
                )
                self.assert_individuals(tmp["attr_val_id"].unique().tolist(), attr_name)
                self.assert_relations(
                    tmp,
                    "attribute_change_value",
                    from_key="attr_change_id",
                    to_key="attr_val_id",
                )
                self.assert_individuals(tmp["time_id"].unique().tolist(), "Time")
                self.assert_relations(
                    tmp, "changed_at", from_key="attr_change_id", to_key="time_id"
                )

    def instantiate_e2o_relation_timestamps(self, data: OCEDStore) -> None:
        """Instantiate timed e2o relation observations as RelationObservation individuals.

        For each event-to-object relation row that carries an ``ocel:timestamp``
        column, creates one ``RelationObservation`` individual and asserts:

        - ``has_relation_observation``: event → RelationObservation
        - ``relation_involves_object``: RelationObservation → object
        - ``relation_observed_at``: RelationObservation → Time individual

        Relations without an ``ocel:timestamp`` column are silently skipped.
        The existing direct ``event_object_relation`` property assertions are
        preserved for backward-compatible queries.

        Args:
            data: OCEDStore whose e2o relation DataFrames optionally contain an
                ``ocel:timestamp`` column (loaded via ``PM4PYLoader`` with the
                ``RELATION_TIMESTAMPS`` extension active).
        """
        for relation in data.get_relation_types(relation_type="e2o"):
            relation_data = data.get_relation(relation)
            rel_df = relation_data["data"]

            if "ocel:timestamp" not in rel_df.columns:
                continue

            from_id_col = relation_data["from_id"]
            to_id_col = relation_data["to_id"]

            tmp = rel_df[[from_id_col, to_id_col, "ocel:timestamp"]].copy()
            tmp = tmp[tmp["ocel:timestamp"].notna()].copy()
            if tmp.empty:
                continue

            tmp["ts_str"] = tmp["ocel:timestamp"].apply(self.generate_timestamp_string)
            tmp["rel_obs_id"] = (
                "RelObs_"
                + tmp[from_id_col].astype(str)
                + "_"
                + tmp[to_id_col].astype(str)
                + "_"
                + tmp["ts_str"]
            )
            tmp["time_id"] = tmp["ts_str"]

            self.assert_individuals(tmp["rel_obs_id"].tolist(), "RelationObservation")
            self.assert_relations(
                tmp,
                "has_relation_observation",
                from_key=from_id_col,
                to_key="rel_obs_id",
            )
            self.assert_relations(
                tmp, "relation_involves_object", from_key="rel_obs_id", to_key=to_id_col
            )
            self.assert_individuals(tmp["time_id"].unique().tolist(), "Time")
            self.assert_relations(
                tmp, "relation_observed_at", from_key="rel_obs_id", to_key="time_id"
            )
