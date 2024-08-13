import cython
import owlready2 as owl
import pandas as pd
import types
import datetime
import semanticlog.onto.datatypes as dt
from urllib.parse import quote

import logging

logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)


class OCEDInstantiator:

    def __init__(self):
        pass

    def instantiate(self, onto, data, logname):
        """takes an OCED upper-level ontology and data to instantiate it with, returns an instantiated ontology"""
        logging.info("instantiating upper level ontology with data for %s..", logname)
        with onto:
            log_instance = onto["Log"](logname)
        logging.info("   instantiating events..")
        onto = self.instantiate_events(onto, data["events"], log_instance)
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
        self.onto = onto
        
        logging.info("done." + "-"*25)
        return onto

    def instantiate_events(self, onto, events, log_instance):
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
                x, onto, activity_column, timestamp_column, attributes, log_instance
            ),
            axis=1,
        )

        return onto

    def instantiate_event(
        self, event_row, onto, activity_column, timestamp_column, attributes, log_instance
    ):
        """instantiates a single event into the OCED ontology"""
        event_instance = onto[event_row[activity_column]](event_row.name)
        event_instance.recordedAt = onto.Time(
            self.clean_timestamp_string(event_row[timestamp_column])
        )
        event_instance.partOf.append(log_instance)
        event_instance = self.instantiate_attributes(onto, event_instance, event_row, attributes)

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
            onto = self.setup_classes(onto, ["has_" + str(x) + "_attribute" for x in attributes], onto.hasAttribute)

            df.apply(
                lambda x: self.instantiate_object(x, onto, object_name, attributes),
                axis=1,
            )

        return onto

    def instantiate_object(self, object_row, onto, object_name, attributes):
        """instantiates a single object into the OCED ontology based on one row from the csv"""
        object_instance = onto[object_name](object_row.name)
        
        object_instance = self.instantiate_attributes(onto, object_instance, object_row, attributes)
        return object_instance
    
    def instantiate_attributes(self, onto, instance, row_data, attributes):
        """ instantiates a dict of attributes for some instance of class event or object in an ontology"""
        attribute_assignment_dict = {"hasAttribute": []}
        
        for attribute in attributes:
            string_value = str(row_data[attribute])
            attribute_class = onto[attribute]
            attribute_instance = attribute_class(string_value)

            attribute_assignment_dict["hasAttribute"].append(attribute_instance)
            
            specific_relation = "has_" + str(attribute) + "_attribute" 
            attribute_assignment_dict[specific_relation] = [attribute_instance]
            
        for attribute_assignment_relation in attribute_assignment_dict:
            setattr(instance, attribute_assignment_relation, attribute_assignment_dict[attribute_assignment_relation])
        
        return instance


    def instantiate_event_to_object_relations(self, onto, event_to_object_relations):
        """takes an OCED ontology and event to object relation data, instantiates the data in the ontology"""#
        
        onto = self.setup_classes(onto, event_to_object_relations.keys(), onto.EventObjectRelation)

        for relation_name in event_to_object_relations:
            logging.info("      instantiating event to object relation %s..", relation_name)
            df = event_to_object_relations[relation_name]["data"]
            from_column = event_to_object_relations[relation_name]["from_foreign_key"]            
            to_column = event_to_object_relations[relation_name]["to_foreign_key"]

            df.apply(
                lambda x: self.instantiate_relation(x, onto, relation_name, from_column, to_column),
                axis=1,
            )

        
        return onto

    def instantiate_object_to_object_relations(self, onto, object_to_object_relations, reify=True):
        """takes an OCED ontology and object to object relation data, instantiates the data in the ontology"""
        if reify:
            onto = self.setup_classes(onto, object_to_object_relations.keys(), onto.ObjectObjectRelation)

            for relation_name in object_to_object_relations:
                logging.info("      instantiating object to object relation %s..", relation_name)
                df = object_to_object_relations[relation_name]["data"]
                from_column = object_to_object_relations[relation_name]["from_foreign_key"]            
                to_column = object_to_object_relations[relation_name]["to_foreign_key"]

                df.apply(
                    lambda x: self.instantiate_reified_object_to_object_relation(x, onto, relation_name, from_column, to_column),
                    axis=1,
                )
        else:
            onto = self.setup_classes(onto, object_to_object_relations.keys(), onto.ObjectObjectRelation)

            for relation_name in object_to_object_relations:
                logging.info("      instantiating object to object relation %s..", relation_name)
                df = object_to_object_relations[relation_name]["data"]
                from_column = object_to_object_relations[relation_name]["from_foreign_key"]            
                to_column = object_to_object_relations[relation_name]["to_foreign_key"]

                df.apply(
                    lambda x: self.instantiate_relation(x, onto, relation_name, from_column, to_column),
                    axis=1,
                )
        
        return onto
    
    def instantiate_relation(self, relation_row, onto, relation_name, from_column, to_column):
        """instantiates a single relation from row data, as well as the key columns"""
        from_instance = onto[relation_row[from_column]]
        to_instance = onto[relation_row[to_column]]
        getattr(from_instance, relation_name).append(to_instance)
        
    def instantiate_reified_object_to_object_relation(self, relation_row, onto, relation_name, from_column, to_column):
        """instantiates a single reified object to object relation from row data, as well as the key columns"""
        from_str = relation_row[from_column]
        to_str = relation_row[to_column]
        from_instance = onto[from_str]
        to_instance = onto[to_str]
        relation_instance = onto[relation_name](from_str + "_" + relation_name + "_" + to_str)
        
        from_instance.inRelation.append(relation_instance)
        relation_instance.with_ = to_instance
        

    def setup_classes(self, onto, classes, superclass, disjoint=True):
        """gets an OCED ontology, list of new concept names and their superclass and creates the new classes in the ontology as
        children of the superclass"""
        with onto:
            for class_ in classes:
                types.new_class(class_, (superclass,))
            if disjoint:
                owl.AllDisjoint(classes)

        return onto
