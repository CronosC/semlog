from abc import ABC, abstractmethod
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.builder import OCEDBaseBuilder
from semanticlog.onto.settings import OntoBuildSettings, EventOrderRepresentation
import pandas as pd
import logging


class OCEDOntoBuilder(OCEDBaseBuilder):

    def __init__(self, iri: str, settings: OntoBuildSettings):
        super().__init__(iri)
        self.settings = settings

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
        if self.settings.order_representation == EventOrderRepresentation.IMPLICIT:
            # nothing more needs to be done
            pass
        elif self.settings.order_representation == EventOrderRepresentation.DF:
            raise NotImplementedError("Implementation of directly follows order representation not yet done!")
        elif self.settings.order_representation == EventOrderRepresentation.DECLARE:
            if self.settings.declare_file == None:
                raise ValueError("Need a declare file when instantiating declare constraints! But was given None.")
            df_declare = pd.read_csv(self.settings.declare_file)
            if df_declare["case_concept"].nunique() == 1:
                case_concept = df_declare["case_concept"].unique()[0]
                constraints = [col for col in df_declare.columns if col not in ["case_concept", "trace_id"]]
                self.add_class("DeclareConstraint")
                self.add_classes(constraints, "DeclareConstraint")
                for constraint in constraints:
                    self.assert_individuals(df_declare[df_declare[constraint] == True]["trace_id"], constraint)
            else:
                raise NotImplementedError("Multiple case concept representation is not implemented yet!")
        else:
            raise ValueError("Unknown event ordering strategy")

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

    def instantiate_attributes(self, attribute_df: pd.DataFrame, key_col: str) -> None:
        for attribute in attribute_df.columns:
            if attribute != key_col:
                self.add_class(attribute, "Attribute")
                has_attribute_prop = self.generate_spezialized_has_attribute_name(
                    attribute
                )
                self.add_property(has_attribute_prop, superprop="has_attribute")

                # converts all string values to numeric if that is possible
                if (
                    attribute_df.dtypes[attribute] == str
                    or attribute_df.dtypes[attribute] == object
                ):
                    try:
                        attribute_df[attribute] = pd.to_numeric(
                            attribute_df[attribute], errors="raise"
                        )
                    except:
                        pass
                    try:
                        pd.to_datetime(series, errors="raise")
                    except:
                        pass

                # for string attributes we instantiate each value as a class
                if (
                    attribute_df.dtypes[attribute]
                    == str
                    # or (attribute_df.dtypes[attribute] == object)
                ):
                    logging.info(f"    instantiating string attribute..")
                    attribute_values = attribute_df[attribute].unique()
                    self.add_classes(attribute_values, attribute)
                    n = 1
                    for attribute_value in attribute_values:
                        individual_name = f"{attribute}_{str(attribute_value)}"
                        self.assert_individual(individual_name, attribute_value)

                        attribute_df[attribute] = attribute_df[attribute].replace(
                            attribute_value, individual_name
                        )
                        n = n + 1
                    self.assert_relations(
                        attribute_df,
                        prop=has_attribute_prop,
                        from_key=key_col,
                        to_key=attribute,
                    )
                # for other attribute types we do not instantiate intermeadiate classes
                else:
                    logging.info(f"    instantiating {attribute_df.dtypes[attribute]} attribute..")
                    attribute_df["attribute_index"] = [
                        f"{attribute}_{i}" for i in range(1, len(attribute_df) + 1)
                    ]
                    self.assert_individuals(
                        list(attribute_df["attribute_index"]), attribute
                    )
                    self.assert_relations(
                        attribute_df,
                        prop=has_attribute_prop,
                        from_key=key_col,
                        to_key="attribute_index",
                    )
                    self.assert_relations(
                        attribute_df,
                        prop="has_value",
                        from_key="attribute_index",
                        to_key=attribute,
                        data_prop=True,
                    )

    def instantiate_event_object_relation_data(self, data: OCEDStore) -> None:
        event_object_relations = data.get_relation_types(relation_type="e2o")

        for relation in event_object_relations:
            relation_data = data.get_relation(relation)
            relation_df = relation_data["data"]

            self.add_property(
                relation,
                superprop="event_to_object_relation",
                range_class=" or ".join(relation_data["to_type"]),
                create_inverse=True,
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
                superprop="object_to_object_relation",
                domain_class=" or ".join(relation_data["from_type"]),
                range_class=" or ".join(relation_data["to_type"]),
                create_inverse=True,
            )
            self.assert_relations(
                relation_df,
                relation,
                from_key=relation_data["from_id"],
                to_key=relation_data["to_id"],
            )
