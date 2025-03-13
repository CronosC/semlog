import datetime
import pandas as pd
import owlapy as owl
from owlapy.iri import IRI
from owlapy.owl_axiom import OWLDeclarationAxiom, OWLSubClassOfAxiom
from owlapy.owl_ontology_manager import OntologyManager
from owlapy.class_expression import OWLClass

from semanticlog.onto.builder import OCEDBaseBuilder
from semanticlog.onto.transform import parse_manchester


class OCEDUpperOntoBuilder(OCEDBaseBuilder):

    def __init__(self, iri):
        super().__init__(iri)

    def construct_upper_ontology(self, path: str) -> None:
        self.add_classes(["Object", "Event", "Attribute", "Time"])
        self.add_classes(["ObjectAttribute", "EventAttribute"], "Attribute")

        self.add_property(
            "observed_at",
            domain_class="Event",
            range_class="Time",
            create_inverse=True,
            inverse_name="observes",
        )
        self.add_property(
            "event_object_relation",
            domain_class="Event",
            range_class="Object",
            create_inverse=True,
        )
        self.add_property(
            "object_object_relation",
            domain_class="Object",
            range_class="Object",
            create_inverse=True,
        )
        self.add_property(
            "has_attribute",
            domain_class="Object or Event",
            range_class="Attribute",
            create_inverse=True,
            inverse_name="attribute_of",
        )
        self.add_property(
            "has_value",
            domain_class="Attribute or Time",
            data_prop=True,
        )

        self.save_onto(path)


'''
class OCEDUpperOntologyBuilder:
    ontoURI = "https://uni-bamberg.de/KIGA/OCED.owl"
    
    def __init__(self):
        """constructor which creates the upper level ontology and makes it available as .onto"""
        logging.info("rebuilding upper level ontology..")
        self.onto = self.build_oced_ul_ontology()


    def build_oced_ul_ontology(self, include_time=False):
        """sets up a generic upper level ontology according to the full OCED meta model for process mining"""
        onto = owl.get_ontology(self.ontoURI)
        if include_time:
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
            class consistsOf(owl.ObjectProperty):
                inverse_property = partOf
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
            class records(owl.ObjectProperty):
                inverse_property = recordedAt

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
            class isAttributeOf(owl.ObjectProperty):
                inverse_property = hasAttribute
            class hasValue(owl.DataProperty, owl.FunctionalProperty):
                domain    = [Attribute]
                range     = [str]

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
            Attribute.equivalent_to = [EventAttribute | ObjectAttribute] #, hasValue.some(str)] leads to error in onto!
            EventAttribute.equivalent_to = [owl.Inverse(hasAttribute).some(Event)]
            ObjectAttribute.equivalent_to = [owl.Inverse(hasAttribute).some(Object)]
            if include_time:
                Time.equivalent_to = [onto_time.Instant & owl.Inverse(recordedAt).some(Event)]

            owl.AllDisjoint([Event, Object, ObjectObjectRelation, Attribute, Time])
            
            #### Data types ####
            
            onto = dt.setup_datatypes(onto)

        return onto
'''
