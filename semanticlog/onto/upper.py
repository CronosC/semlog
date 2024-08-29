import cython
import owlready2 as owl
import pandas as pd
import datetime
import semanticlog.onto.datatypes as dt

import logging

class OCEDUpperOntologyBuilder:
    ontoURI = "https://uni-bamberg.de/KIGA/OCED.owl"
    
    def __init__(self):
        """constructor which creates the upper level ontology and makes it available as .onto"""
        logging.info("rebuilding upper level ontology..")
        self.onto = self.build_oced_ul_ontology()


    def build_oced_ul_ontology(self):
        """sets up a generic upper level ontology according to the full OCED meta model for process mining"""
        onto = owl.get_ontology(self.ontoURI)
        onto_time = owl.get_ontology("data/ontos/Time.xml").load()

        with onto:
            #### Concepts ####
            class Log(owl.Thing):pass
            # attribute related classes
            class Attribute(owl.Thing):pass
            class EventAttribute(Attribute):pass
            class ObjectAttribute(Attribute):pass
            # eventand object related classes
            class Event(owl.Thing):pass
            class Trace(owl.Thing):pass
            class Start(Event):pass
            class End(Event):pass
            class Object(owl.Thing):pass
            class Time(owl.Thing):pass
            # reified object_to_object relations
            class ObjectObjectRelation(owl.Thing):pass
            class child_of(ObjectObjectRelation):pass
            class parent_of(ObjectObjectRelation):pass
            # misc
            class DeclareConstraint(Trace):pass

            #### Properties ####
            class partOf(owl.ObjectProperty):
                domain    = [Event | Trace]
                range     = [Log | Trace]
            # event_to_object relations
            class EventObjectRelation(owl.ObjectProperty):
                domain    = [Event]
                range     = [Object | ObjectObjectRelation | ObjectAttribute]
            class create(EventObjectRelation):
                domain    = [Event]
                range     = [Object | ObjectObjectRelation | ObjectAttribute]
            class modify(EventObjectRelation):
                domain    = [Event]
                range     = [Object | ObjectObjectRelation | ObjectAttribute]
            class delete(EventObjectRelation):
                domain    = [Event]
                range     = [Object | ObjectObjectRelation | ObjectAttribute]

            # time related relations
            class recordedAt(owl.ObjectProperty, owl.FunctionalProperty):
                domain    = [Event]
                range     = [Time]

            # relations to create reified object_to_object relations
            class inRelation(owl.ObjectProperty, owl.InverseFunctionalProperty):
                domain    = [Object]
                range     = [ObjectObjectRelation]
            class with_(owl.ObjectProperty, owl.FunctionalProperty):
                domain    = [ObjectObjectRelation]
                range     = [Object]

            # attribute assignment relations
            class hasAttribute(owl.ObjectProperty):
                domain    = [Event | Object]
                range     = [Attribute]
            class hasValue(owl.DataProperty, owl.FunctionalProperty):
                domain    = [Attribute]

            # event order relations
            class ef(owl.ObjectProperty, owl.TransitiveProperty):
                domain    = [Event]
                range     = [Event]
            class df(ef, owl.FunctionalProperty, owl.InverseFunctionalProperty):
                domain    = [Event]
                range     = [Event]


            #### Constraints ####
            Event.equivalent_to = [recordedAt.some(onto.Time)]
            Start.equivalent_to = [Event & owl.Not(owl.Inverse(df).some(Event))]
            End.equivalent_to = [Event & owl.Not(df.some(Event))]
            Attribute.equivalent_to = [EventAttribute | ObjectAttribute, hasValue.some(owl.Thing)]
            EventAttribute.equivalent_to = [owl.Inverse(hasAttribute).some(Event)]
            ObjectAttribute.equivalent_to = [owl.Inverse(hasAttribute).some(Object)]
            Time.equivalent_to = [onto_time.Instant & owl.Inverse(recordedAt).some(Event)]

            owl.AllDisjoint([Event, Object, ObjectObjectRelation, Attribute, Time])

        return onto