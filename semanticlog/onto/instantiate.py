from abc import ABC, abstractmethod
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.builder import OCEDBaseBuilder
import pandas as pd


class OCEDOntoBuilder(OCEDBaseBuilder):

    def __init__(self, iri):
        super().__init__(iri)

    def build_ontology(self, data: OCEDStore) -> None:
        self.instantiate_event_data(data)
        self.instantiate_time_data(data)

        self.instantiate_object_data(data)

    def instantiate_event_data(self, data: OCEDStore) -> None:
        activities = data.get_activities()

        self.add_classes(activities, "Event")
        for activity in activities:
            activity_data = data.get_events(activity)
            activity_df = activity_data["data"]
            self.assert_individuals(
                list(activity_df[activity_data["id_col"]]), activity
            )
        self.instantiate_attributes(
            activity_df.drop(columns=[activity_data["time_col"]]),
            activity_data["id_col"],
        )

    def instantiate_time_data(self, data: OCEDStore) -> None:
        df_list = []

        for activity in data.get_activities():
            activity_data = data.get_event(activity)
            activity_df = activity_data["data"]
            activity_df = activity_df.rename(
                columns={
                    activity_data["id_col"]: "event",
                    activity_data["time_col"]: "time",
                }
            )
            df_list.append(activity_df)

        df_time = pd.concat(df_list)
        df_time['time_str'] = df_time['time'].apply(lambda x: self.generate_timestamp_string(x))
        self.assert_individuals(list(df_time["time_str"]), "Time")
        self.assert_relations(
            df_time, prop="observed_at", from_key="event", to_key="time_str"
        )
    def instantiate_object_data(self, data: OCEDStore) -> None:
        object_types = data.get_object_types()

        self.add_classes(object_types, "Object")
        for obj in object_types:
            object_data = data.get_object(obj)
            object_df = object_data["data"]
            self.assert_individuals(list(object_df[object_data["id_col"]]), obj)
        self.instantiate_attributes(object_df, object_data["id_col"])

    def instantiate_attributes(self, attribute_df: pd.DataFrame, key_col: str) -> None:
        for attribute in attribute_df.columns:
            if attribute != key_col:
                self.add_class(attribute, "Attribute")
                has_attribute_prop = self.generate_spezialized_has_attribute_name(
                    attribute
                )
                self.add_property(has_attribute_prop, superprop="has_attribute")

                # for string attributes we instantiate each value as a class
                if (
                    attribute_df.dtypes[attribute] == str
                    or attribute_df.dtypes[attribute] == object
                ):
                    attribute_values = attribute_df[attribute].unique()
                    self.add_classes(attribute_values, attribute)
                    n = 1
                    for attribute_value in attribute_values:
                        individual_name = attribute + "_" + str(n)
                        attribute_value_df = attribute_df[attribute][
                            attribute_df[attribute] == attribute_value
                        ]
                        self.assert_individual(individual_name, attribute_value)
                        attribute_df[attribute] = attribute_df[attribute].replace(attribute_value, individual_name)
                        n = n + 1
                    self.assert_relations(
                        attribute_df,
                        prop=has_attribute_prop,
                        from_key=key_col,
                        to_key=attribute,
                    )
                # for other attribute types we do not instantiate intermeadiate classes
                else:
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
                range_class=relation_data["to_type"],
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
                domain_class=relation_data["from_type"],
                range_class=relation_data["to_type"],
                create_inverse=True,
            )
            self.assert_relations(
                relation_df,
                relation,
                from_key=relation_data["from_id"],
                to_key=relation_data["to_id"],
            )
