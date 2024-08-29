import cython
import owlready2 as owl
import pandas as pd
import types
import datetime
import semanticlog.onto.datatypes as dt
from urllib.parse import quote

import logging


class OCEDInstantiator:

    def __init__(self):
        pass

    def instantiate(self, onto, data, logname):
        """takes an OCED upper-level ontology and data to instantiate it with, returns an instantiated ontology"""
        logging.info("instantiating upper level ontology with data for %s..", logname)
        with onto:
            self.log_instance = onto["Log"](logname)
        logging.info("   instantiating events..")
        onto = self.instantiate_events(onto, data["events"])
        logging.info("   instantiating objects..")
        onto = self.instantiate_objects(onto, data["objects"])
        logging.info("   instantiating event to object relations..")
        onto = self.instantiate_event_to_object_relations(
            onto, data["relations"]["event_to_object"]
        )
        logging.info("   instantiating object to object relations..")
        onto = self.instantiate_object_to_object_relations(
            onto, data["relations"]["object_to_object"]
        )
        if "misc" in data.keys():
            if "case_concepts" in data["misc"].keys():
                logging.info("   instantiating event order via case concepts..")
                onto = self.instantiate_case_concepts(
                    onto, data["misc"]["case_concepts"]
                )
            if "declare" in data["misc"].keys():
                logging.info("   instantiating declare constraints..")
                onto = self.instantiate_declare_constraints(onto, data["misc"]["declare"])

        self.onto = onto

        logging.info("done." + "-" * 25)
        return onto

    def instantiate_events(self, onto, events):
        """takes an OCED ontology and event data, instantiates the data in the ontology"""
        df = events["data"]
        activity_column = events["event_type_column"]
        timestamp_column = events["timestamp_column"]
        attributes = events["attribute_columns"]

        activities = df[activity_column].unique()
        onto = self.setup_classes(onto, activities, onto.Event)
        onto = self.setup_classes(onto, attributes, onto.EventAttribute)

        df.apply(
            lambda x: self.instantiate_event(
                x, onto, activity_column, timestamp_column, attributes
            ),
            axis=1,
        )

        return onto

    def instantiate_event(
        self,
        event_row,
        onto,
        activity_column,
        timestamp_column,
        attributes,
    ):
        """instantiates a single event into the OCED ontology"""
        event_instance = onto[event_row[activity_column]](event_row.name)
        event_instance.recordedAt = onto.Time(
            self.clean_timestamp_string(event_row[timestamp_column])
        )
        event_instance = self.instantiate_attributes(
            onto, event_instance, event_row, attributes
        )

        return event_instance

    def clean_timestamp_string(self, timestamp):
        """URIs can not contain certain symbols, in order to use the timestamp as an id we need to clean it"""
        return quote(timestamp, safe="/:#")

    def instantiate_objects(self, onto, objects):
        """takes an OCED ontology and object data, instantiates the data in the ontology"""
        onto = self.setup_classes(onto, objects.keys(), onto.Object)

        for object_name in objects:
            logging.info("      instantiating object %s..", object_name)
            df = objects[object_name]["data"]
            attributes = objects[object_name]["attribute_columns"]

            onto = self.setup_classes(onto, attributes, onto.ObjectAttribute)
            onto = self.setup_classes(
                onto,
                ["has_" + str(x) + "_attribute" for x in attributes],
                onto.hasAttribute,
            )

            df.apply(
                lambda x: self.instantiate_object(x, onto, object_name, attributes),
                axis=1,
            )

        return onto

    def instantiate_object(self, object_row, onto, object_name, attributes):
        """instantiates a single object into the OCED ontology based on one row from the csv"""
        object_instance = onto[object_name](object_row.name)

        object_instance = self.instantiate_attributes(
            onto, object_instance, object_row, attributes
        )
        return object_instance

    def instantiate_attributes(self, onto, instance, row_data, attributes):
        """instantiates a dict of attributes for some instance of class event or object in an ontology"""
        attribute_assignment_dict = {"hasAttribute": []}

        for attribute in attributes:
            string_value = str(row_data[attribute])
            attribute_class = onto[attribute]
            attribute_instance = attribute_class(string_value)

            attribute_assignment_dict["hasAttribute"].append(attribute_instance)

            specific_relation = "has_" + str(attribute) + "_attribute"
            attribute_assignment_dict[specific_relation] = [attribute_instance]

        for attribute_assignment_relation in attribute_assignment_dict:
            setattr(
                instance,
                attribute_assignment_relation,
                attribute_assignment_dict[attribute_assignment_relation],
            )

        return instance

    def instantiate_event_to_object_relations(self, onto, event_to_object_relations):
        """takes an OCED ontology and event to object relation data, instantiates the data in the ontology"""  #

        onto = self.setup_classes(
            onto, event_to_object_relations.keys(), onto.EventObjectRelation
        )

        for relation_name in event_to_object_relations:
            logging.info(
                "      instantiating event to object relation %s..", relation_name
            )
            df = event_to_object_relations[relation_name]["data"]
            from_column = event_to_object_relations[relation_name]["from_foreign_key"]
            to_column = event_to_object_relations[relation_name]["to_foreign_key"]

            df.apply(
                lambda x: self.instantiate_relation(
                    x, onto, relation_name, from_column, to_column
                ),
                axis=1,
            )

        return onto

    def instantiate_object_to_object_relations(
        self, onto, object_to_object_relations, reify=True
    ):
        """takes an OCED ontology and object to object relation data, instantiates the data in the ontology"""
        if reify:
            onto = self.setup_classes(
                onto, object_to_object_relations.keys(), onto.ObjectObjectRelation
            )

            for relation_name in object_to_object_relations:
                logging.info(
                    "      instantiating object to object relation %s..", relation_name
                )
                df = object_to_object_relations[relation_name]["data"]
                from_column = object_to_object_relations[relation_name][
                    "from_foreign_key"
                ]
                to_column = object_to_object_relations[relation_name]["to_foreign_key"]

                df.apply(
                    lambda x: self.instantiate_reified_object_to_object_relation(
                        x, onto, relation_name, from_column, to_column
                    ),
                    axis=1,
                )
        else:
            onto = self.setup_classes(
                onto, object_to_object_relations.keys(), onto.ObjectObjectRelation
            )

            for relation_name in object_to_object_relations:
                logging.info(
                    "      instantiating object to object relation %s..", relation_name
                )
                df = object_to_object_relations[relation_name]["data"]
                from_column = object_to_object_relations[relation_name][
                    "from_foreign_key"
                ]
                to_column = object_to_object_relations[relation_name]["to_foreign_key"]

                df.apply(
                    lambda x: self.instantiate_relation(
                        x, onto, relation_name, from_column, to_column
                    ),
                    axis=1,
                )

        return onto

    def instantiate_relation(
        self, relation_row, onto, relation_name, from_column, to_column
    ):
        """instantiates a single relation from row data, as well as the key columns"""
        from_instance = onto[relation_row[from_column]]
        to_instance = onto[relation_row[to_column]]
        getattr(from_instance, relation_name).append(to_instance)

    def instantiate_reified_object_to_object_relation(
        self, relation_row, onto, relation_name, from_column, to_column
    ):
        """instantiates a single reified object to object relation from row data, as well as the key columns"""
        from_str = relation_row[from_column]
        to_str = relation_row[to_column]
        from_instance = onto[from_str]
        to_instance = onto[to_str]
        relation_instance = onto[relation_name](
            from_str + "_" + relation_name + "_" + to_str
        )

        from_instance.inRelation.append(relation_instance)
        relation_instance.with_ = to_instance

    def instantiate_case_concepts(self, onto, case_concepts):
        """instantiates the event order via directlyFollows relations over events derived from case concepts"""
        onto = self.setup_classes(onto, case_concepts.keys(), onto.df)

        for case_co_name in case_concepts:
            logging.info(
                "      instantiating df relations for case concept %s..", case_co_name
            )
            df_case_co = case_concepts[case_co_name]["data"]
            event_column = case_concepts[case_co_name]["event_key"]
            time_column = case_concepts[case_co_name]["timestamp_column"]
            primary_key = case_concepts[case_co_name]["primary_key"]

            df_case_co = df_case_co.sort_values(by=[primary_key, time_column])
            df_case_co = (
                df_case_co.groupby(primary_key)[event_column].apply(list).reset_index()
            )

            df_case_co.apply(
                lambda x: self.instantiate_case_concept(
                    x, onto, case_co_name, event_column, primary_key
                ),
                axis=1,
            )

        return onto

    def instantiate_case_concept(
        self, case_concept_row, onto, case_co_name, event_column, case_co_column
    ):
        with onto:
            event_sequence = case_concept_row[event_column]

            onto.Start(event_sequence[0])
            onto.End(event_sequence[-1])

            trace_instance = onto.Trace(self.get_trace_id_from_case_concept_id(case_concept_row[case_co_column]))
            trace_instance.partOf.append(self.log_instance)
            for i in range(0, len(event_sequence)):
                event_instance = onto[event_sequence[i]]
                event_instance.partOf.append(trace_instance)

            for i in range(0, len(event_sequence) - 1):
                event_instance_current = onto[event_sequence[i]]
                event_instance_next = onto[event_sequence[i + 1]]
                setattr(event_instance_current, case_co_name, event_instance_next)

    def get_trace_id_from_case_concept_id(self, case_co_id):
        return "trace_" + case_co_id

    def instantiate_declare_constraints(self, onto, declare):
        for declare_view in declare:
            df_declare = declare[declare_view]["data"]
            case_column = declare[declare_view]["df_view"]

            constraints = list(df_declare.columns)
            constraints.remove(case_column)
            constraints.remove("Unnamed: 0")
            onto = self.setup_classes(onto, constraints, onto.DeclareConstraint)
            
            df_declare.apply(
                lambda x: self.instantiate_declare_constraint(x, onto, case_column),
                axis=1,
            )

        return onto

    def instantiate_declare_constraint(self, constraint_row, onto, case_column):
        trace_id = self.get_trace_id_from_case_concept_id(constraint_row[case_column])
        constraints = list(constraint_row.keys())
        constraints.remove(case_column)
        constraints.remove("Unnamed: 0")
        with onto:
            valid_constraints = filter(lambda c: constraint_row[c] == 1, constraints)
            list(map(lambda c: (onto[c](trace_id)), valid_constraints))

    def setup_classes(self, onto, classes, superclass, disjoint=True):
        """gets an OCED ontology, list of new concept names and their superclass and creates the new classes in the ontology as
        children of the superclass"""
        with onto:
            for class_ in classes:
                types.new_class(class_, (superclass,))
            if disjoint:
                owl.AllDisjoint(classes)

        return onto
