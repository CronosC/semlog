from abc import ABC, abstractmethod
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.builder import OCEDBaseBuilder
from semanticlog.onto.settings import OntoBuildSettings, EventOrderRepresentation, OwlProfile
from semanticlog.onto.transform import parse_manchester
from owlapy.class_expression import OWLObjectHasSelf, OWLObjectSomeValuesFrom
from owlapy.owl_property import OWLObjectProperty
from owlapy.owl_axiom import OWLSubClassOfAxiom, OWLSubPropertyAxiom, OWLTransitiveObjectPropertyAxiom
import pandas as pd
import logging
import warnings


class OCEDOntoBuilder(OCEDBaseBuilder):

    def __init__(self, iri: str, settings: OntoBuildSettings):
        super().__init__(iri, settings)

    def build_ontology(self, data: OCEDStore) -> None:
        logging.info(f"Instantiating ontology..")
        logging.info(
            f"Instantiating event data ({len(data.get_activities())} activities).."
        )
        self.instantiate_event_data(data)
        logging.info(
            f"Instantiating event order via the strategy: {self.settings.order_representation}.."
        )
        self.instantiate_event_order(data)
        logging.info(f"Instantiating time data..")
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
        logging.info(f"done.")

    def instantiate_event_data(self, data: OCEDStore) -> None:
        activities = data.get_activities()

        self.add_classes(activities, "Event")
        for activity in activities:
            activity_data = data.get_events(activity)
            activity_df = activity_data["data"]
            self.assert_individuals(list(activity_df[activity_data["id"]]), activity)
            self.instantiate_attributes(
                activity_df.drop(columns=[activity_data["time"]]),
                activity_data["id"],
            )

    def instantiate_event_order(self, data: OCEDStore) -> None:
        self.add_class("Trace")
        self.add_property("part_of", "Event", "Trace", create_inverse=self.settings.inverse_properties)
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
            or EventOrderRepresentation.EF_explicit in self.settings.order_representation
        ):
            self.instantiate_event_order_via_directly_follows(data)

        if EventOrderRepresentation.DECLARE in self.settings.order_representation:
            self.instantiate_event_order_via_declare(data)

    def instantiate_event_order_via_directly_follows(self, data: OCEDStore) -> None:
        include_ef = (
            EventOrderRepresentation.EF in self.settings.order_representation
            or EventOrderRepresentation.EF_explicit in self.settings.order_representation
        )

        self.add_property("df", "Event", "Event", create_inverse=self.settings.inverse_properties)
        self.add_classes(["StartEvent", "EndEvent"], "Event")

        if include_ef:
            self.add_property("ef", "Event", "Event", create_inverse=self.settings.inverse_properties)

        for case_concept in self.settings.case_concepts:
            self.instantiate_directly_follows(case_concept, data, include_ef)

    def instantiate_directly_follows(self, case_concept, data, include_ef):
        events = data.get_log(case_concept)
        events = events.sort_values(
            ["case:concept:name", "time:timestamp"], kind="mergesort"
        )

        events["event_id"] = events.index
        events["next_event_id"] = events.groupby("case:concept:name")["event_id"].shift(
            -1
        )

        start_events = events.groupby("case:concept:name").first()
        end_events = events.groupby("case:concept:name").last()
        self.assert_individuals(list(start_events["event_id"]), "StartEvent")
        self.assert_individuals(list(end_events["event_id"]), "EndEvent")

        df_pairs = events.loc[
            events["next_event_id"].notna(), ["event_id", "next_event_id"]
        ]

        df_rel = "df_" + case_concept
        self.add_property(df_rel, "Event", "Event", "df", create_inverse=self.settings.inverse_properties)
        self.assert_relations(df_pairs, df_rel, "event_id", "next_event_id")

        if include_ef:
            make_explicit = EventOrderRepresentation.EF_explicit in self.settings.order_representation
            if not make_explicit and self.settings.profile == OwlProfile.OWL2EL:
                raise ValueError("Can not include ef relations that are not explicit in the OWL2EL profile!")
            
            ef_rel = "ef_" + case_concept
            self.add_property(ef_rel, "Event", "Event", "ef", create_inverse=self.settings.inverse_properties)
            
            if self.settings.profile == OwlProfile.OWL2DL:
                # ef as reflexive, transitive closure of df only if profile is DL
                part_of_trace = OWLObjectSomeValuesFrom(OWLObjectProperty(self.iri + "part_of"), parse_manchester(f"Trace_{case_concept}"))
                locally_reflexive = OWLObjectHasSelf(OWLObjectProperty(self.iri + ef_rel))
                # make all events in such a trace ef themselves.
                self.add_axiom(OWLSubClassOfAxiom(part_of_trace, locally_reflexive))
                # make df subclass of ef
                self.add_axiom(OWLSubPropertyAxiom(OWLObjectProperty(self.iri + df_rel), OWLObjectProperty(self.iri + ef_rel)))
                # make ef transitive
                self.add_axiom(OWLTransitiveObjectPropertyAxiom(OWLObjectProperty(self.iri + ef_rel)))
                
            
            if make_explicit:
                ef_pairs = events.merge(
                    events, on="case:concept:name", suffixes=("_from", "_to")
                )
                ef_pairs = ef_pairs[
                    (ef_pairs["time:timestamp_from"] < ef_pairs["time:timestamp_to"])
                    | (
                        (ef_pairs["time:timestamp_from"] == ef_pairs["time:timestamp_to"])
                        & (ef_pairs["event_id_from"] <= ef_pairs["event_id_to"])
                    )
                ]

                self.assert_relations(ef_pairs, ef_rel, "event_id_from", "event_id_to")

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
            self.add_class("DeclareConstraint", "Trace")
            self.add_classes(constraints, "DeclareConstraint")

            for constraint in constraints:
                trace_ids = df_declare.loc[df_declare[constraint], "trace_id"]
                self.assert_individuals(trace_ids, constraint)

    def instantiate_time_data(self, data: OCEDStore) -> None:
        df_list = []

        for activity in data.get_activities():
            activity_data = data.get_event(activity)
            activity_df = activity_data["data"]
            activity_df = activity_df.rename(
                columns={
                    activity_data["id"]: "event",
                    activity_data["time"]: "time",
                }
            )
            df_list.append(activity_df)
        df_time = pd.concat(df_list)
        df_time["time_str"] = df_time["time"].apply(
            lambda x: self.generate_timestamp_string(x)
        )
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
            object_df = object_data["data"]
            self.assert_individuals(list(object_df[object_data["id"]]), obj)
            self.instantiate_attributes(object_df, object_data["id"])

    def instantiate_attributes(self, df: pd.DataFrame, key_col: str) -> None:
        for attr in (c for c in df.columns if c != key_col):
            if df[attr].notna().any():
                df_attr = df[df[attr].notna()].copy()
                prop = self.generate_spezialized_has_attribute_name(attr)
                self.add_class(attr, "Attribute")
                self.add_property(prop, superprop="has_attribute")

                s = df_attr[attr]
                if pd.api.types.is_datetime64_any_dtype(s):
                    self._instantiate_attr(
                        df_attr[[key_col, attr]],
                        key_col,
                        attr,
                        prop,
                        value_to_id=lambda v: f"{attr}_{self.generate_timestamp_string(v)}",
                        also_assert_data_value=True,
                    )
                elif pd.api.types.is_numeric_dtype(s):
                    self._instantiate_attr(
                        df_attr[[key_col, attr]],
                        key_col,
                        attr,
                        prop,
                        value_to_id=lambda v: f"{attr}_{v}",
                        also_assert_data_value=True,
                    )
                else:
                    # treat object/string/categorical/bool as string labels
                    self._instantiate_attr(
                        df_attr[[key_col, attr]],
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
        tmp = df[[key_col, attr]].copy().dropna(subset=[attr])

        # map each distinct value to an attribute-individual IRI/name
        vals = tmp[attr].unique()
        mapping = {v: value_to_id(v) for v in vals}
        tmp["attr_ind"] = tmp[attr].map(mapping)

        # create individuals for the attribute values
        self.assert_individuals(tmp["attr_ind"].tolist(), attr)

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
