import cython
import owlready2 as owl
import pandas as pd
import types
import datetime
import semanticlog.onto.datatypes as dt
from urllib.parse import quote
from rdflib import Graph, URIRef, RDF

import logging


class OCEDInstantiator:

    def __init__(self):
        pass

    def instantiate(self, onto, data, logname, reason=False):
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
                
        if reason:
            logging.info("   running Pellet..")
            with onto:
                owl.sync_reasoner_pellet()

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
        onto = self.setup_classes(onto, attributes, onto.EventAttribute, disjoint=False)

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

            onto = self.setup_classes(onto, attributes, onto.ObjectAttribute, disjoint=False)
            onto = self.setup_classes(
                onto,
                ["has_" + str(x) + "_attribute" for x in attributes],
                onto.hasAttribute,
                disjoint=False
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
            string_value = quote(str(row_data[attribute]), safe="/:#")
            attribute_class = onto[attribute]
            attribute_instance = attribute_class(attribute + "_" + string_value)
            attribute_instance.hasValue = string_value

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
        relations_to_setup = list(event_to_object_relations.keys())
        onto = self.setup_classes(
            onto, relations_to_setup, onto.EventObjectRelation, disjoint=False, create_inverse_relation=True
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
        relations_to_setup = list(object_to_object_relations.keys())
        onto = self.setup_classes(
            onto, relations_to_setup, onto.ObjectObjectRelation, disjoint=False
        )
        if reify:
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

    def instantiate_case_concepts(self, onto, case_concepts, add_variants=True):
        """instantiates the event order via directlyFollows relations over events derived from case concepts"""
        onto = self.setup_classes(onto, case_concepts.keys(), onto.df, disjoint=False)
        if add_variants:
            onto = self.setup_classes(onto, ["has_variant_attribute"], onto.hasAttribute, disjoint=False, create_inverse_relation=True)
            onto = self.setup_classes(onto, ["variant"], onto.ObjectAttribute, disjoint=False)
        
        for case_co_name in case_concepts:
            logging.info(
                "      instantiating df relations for case concept %s..", case_co_name
            )
            df_case_co = case_concepts[case_co_name]["data"]
            event_type_column = case_concepts[case_co_name]["event_type_column"]
            event_column = case_concepts[case_co_name]["event_key"]
            time_column = case_concepts[case_co_name]["timestamp_column"]
            primary_key = case_concepts[case_co_name]["primary_key"]
            


            df_case_co = df_case_co.sort_values(by=[primary_key, time_column])
            df_event_sequence = (
                df_case_co.groupby(primary_key)[event_column].apply(list).reset_index()
            )

            df_event_sequence.apply(
                lambda x: self.instantiate_case_concept(
                    x, onto, case_co_name, event_column, primary_key
                ),
                axis=1,
            )

            if add_variants:
                df_variants = (
                    df_case_co.groupby(primary_key)[event_type_column].apply(list).reset_index()
                )

                df_variants = df_variants.rename(columns={event_type_column: "variant"})
                df_variants.apply(
                    lambda x: self.instantiate_variant_information(
                        x, onto, case_co_name, primary_key
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

            trace_instance = onto.Trace(case_concept_row[case_co_column])
            trace_instance.partOf.append(self.log_instance)
            for i in range(0, len(event_sequence)):
                event_instance = onto[event_sequence[i]]
                event_instance.partOf.append(trace_instance)

            for i in range(0, len(event_sequence) - 1):
                event_instance_current = onto[event_sequence[i]]
                event_instance_next = onto[event_sequence[i + 1]]
                setattr(event_instance_current, case_co_name, event_instance_next)
                

    def instantiate_variant_information(
        self, case_concept_row, onto, case_co_name, case_co_column
    ):
        with onto:
            variant = case_concept_row["variant"]
            trace_instance = onto.Trace(case_concept_row[case_co_column])

            self.instantiate_attributes(onto, trace_instance, case_concept_row, ["variant"])

    def instantiate_declare_constraints(self, onto, declare):
        for declare_view in declare:
            df_declare = declare[declare_view]["data"]
            case_column = declare[declare_view]["df_view"]

            constraints = [
                col for col in df_declare.columns 
                if col not in {case_column, "Unnamed: 0"} and (df_declare[col] == 0).any()
            ]
            onto = self.setup_classes(onto, constraints, onto.DeclareConstraint, disjoint=False)

            # pre-filter once to create a dictionary of trace_ids by constraint
            valid_constraints = {
                onto[constraint]: df_declare.loc[df_declare[constraint] == 1, case_column].tolist()
                for constraint in constraints
            }
            
            n=1
            l=len(constraints)
            with onto:
                for constraint, trace_ids in valid_constraints.items():
                    logging.info(f"      {n}/{l}..\033[1A")
                    n=n+1
                    for trace_id in trace_ids:
                        constraint(trace_id)

        return onto

    def setup_classes(self, onto, classes, superclass, disjoint=True, create_inverse_relation=False):
        """gets an OCED ontology, list of new concept names and their superclass and creates the new classes in the ontology as
        children of the superclass"""
        with onto:
            class_instances = []
            for class_ in classes:
                new_class = types.new_class(class_, (superclass,))
                class_instances.append(new_class)
                if create_inverse_relation:
                    inverse_relation = types.new_class(self.get_name_of_inverse(class_), (owl.ObjectProperty,))
                    new_class.inverse_property = inverse_relation
            if disjoint:
                owl.AllDisjoint(class_instances)

        return onto
    
    def get_name_of_inverse(self, relation_name):
        return "inverse_" + relation_name
