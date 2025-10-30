import os
import random
import unittest
from datetime import datetime, timedelta

import pandas as pd
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.instantiate import OCEDOntoBuilder
from semanticlog.onto.settings import OntoBuildSettings, OwlProfile, EventOrderRepresentation
from semanticlog.onto.upper import OCEDUpperOntoBuilder

from owlapy.iri import IRI
from owlapy.class_expression import OWLClass, OWLThing
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_ontology import Ontology
from owlapy.owl_property import OWLObjectProperty, OWLDataProperty
from owlapy.owl_reasoner import StructuralReasoner


NS = "https://oced#" 

class TestOntoBuilder(unittest.TestCase):

    def setUp(self):
        self.settings = OntoBuildSettings()
        self.settings.profile = OwlProfile.OWL2EL
        self.settings.order_representation = []
        self.settings.case_concepts = []
        
        if os.path.exists("data/tmp/tmp.owl"):
            os.remove("data/tmp/tmp.owl")
        builder = OCEDUpperOntoBuilder("https://oced#", self.settings)
        builder.construct_upper_ontology("data/tmp/tmp.owl")

        self.data = OCEDStore()
        self.data._events = {
            "activity1": {
                "id": "id",
                "time": "time",
                "data": pd.DataFrame(
                    {
                        "time": [
                            datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)
                        ],
                        "id": ["e" + str(i) for i in range(10)],
                        "resource": ["Bob", "Bob", "Alice", "Bob", "Alice", "Alice", "Alice", "Bob", "Alice", "Alice"],
                        "number": [10.0, 26.5, 50.1, 23.2, 50.3, 65.4, 72.5, 46.6, 96.7, 16.8],
                    }
                ),
            }
        }
        self.data._objects = {
            "thing1": {
                "id": "id",
                "data": pd.DataFrame(
                    {
                        "id": ["o" + str(i) for i in range(10)],
                        "type": ["Small", "Small", "Medium", "Large", "Large", "Large", "Small", "Medium", "Small", "Medium"],
                        "value": [120.0, None, 605.4, 243.2, 530.3, 605.4, 732.5, 416.6, 196.7, 136.8],
                    }
                ),
            }
        }
        self.data._relations = {
            "event_to_object": {
                "does": {
                    "from_id": "event_id",
                    "to_id": "object_id",
                    "to_type": ["thing1"],
                    "data": pd.DataFrame(
                        {
                            "event_id": ["e" + str(random.randint(1, 10)) for _ in range(10)],
                            "object_id": ["o" + str(random.randint(1, 10)) for _ in range(10)],
                        }
                    ),
                }
            },
            "object_to_object": {
                "connects": {
                    "from_id": "object_id1",
                    "from_type": ["thing1"],
                    "to_id": "object_id2",
                    "to_type": ["thing1"],
                    "data": pd.DataFrame(
                        {
                            "object_id1": ["o" + str(random.randint(1, 10)) for _ in range(10)],
                            "object_id2": ["o" + str(random.randint(1, 10)) for _ in range(10)],
                        }
                    ),
                }
            },
        }


    def _load_reasoner(self):
        onto = Ontology(IRI.create("file://data/tmp/tmp.owl"))
        return onto, StructuralReasoner(onto)

    def test_events_basic(self):
        builder = OCEDOntoBuilder("data/tmp/tmp.owl", self.settings)
        builder.instantiate_event_data(self.data)
        builder.save_onto("data/tmp/tmp.owl")

        onto, reasoner = self._load_reasoner()

        activity1 = OWLClass(IRI(NS, "activity1"))
        with self.subTest("class exists"):
            self.assertIn(activity1, list(onto.classes_in_signature()))

        instances = set(reasoner.instances(activity1))
        any_expected = any(OWLNamedIndividual(IRI(NS, f"e{i}")) in instances for i in range(10))
        with self.subTest("has instances"):
            self.assertTrue(any_expected, "No event individuals e0..e9 found as instances of activity1")

    def test_objects_basic(self):
        builder = OCEDOntoBuilder("data/tmp/tmp.owl", self.settings)
        builder.instantiate_object_data(self.data)
        builder.save_onto("data/tmp/tmp.owl")

        onto, reasoner = self._load_reasoner()

        thing1 = OWLClass(IRI(NS, "thing1"))
        with self.subTest("class exists"):
            self.assertIn(thing1, list(onto.classes_in_signature()))

        instances = set(reasoner.instances(thing1))
        any_expected = any(OWLNamedIndividual(IRI(NS, f"o{i}")) in instances for i in range(10))
        with self.subTest("has instances"):
            self.assertTrue(any_expected, "No object individuals o0..o9 found as instances of thing1")

    def test_e2o_relations(self):
        builder = OCEDOntoBuilder("data/tmp/tmp.owl", self.settings)
        builder.instantiate_event_data(self.data)
        builder.instantiate_object_data(self.data)
        builder.instantiate_event_object_relation_data(self.data)
        builder.save_onto("data/tmp/tmp.owl")

        onto, reasoner = self._load_reasoner()

        does = OWLObjectProperty(IRI(NS, "does"))
        with self.subTest("property exists"):
            self.assertIn(does, list(onto.object_properties_in_signature()))

        has_any_assertion = False
        for ind in list(onto.individuals_in_signature()):
            props = set(reasoner.ind_object_properties(ind, direct=False))
            if does in props:
                if list(reasoner.object_property_values(ind, does, direct=False)):
                    has_any_assertion = True
                    break
        with self.subTest("has assertions"):
            self.assertTrue(has_any_assertion, "No OWL object property assertions found for 'does'")

    def test_o2o_relations(self):
        builder = OCEDOntoBuilder("data/tmp/tmp.owl", self.settings)
        builder.instantiate_event_data(self.data)
        builder.instantiate_object_data(self.data)
        builder.instantiate_object_to_object_relation_data(self.data)
        builder.save_onto("data/tmp/tmp.owl")

        onto, reasoner = self._load_reasoner()

        connects = OWLObjectProperty(IRI(NS, "connects"))
        with self.subTest("property exists"):
            self.assertIn(connects, list(onto.object_properties_in_signature()))

        has_any_assertion = False
        for ind in onto.individuals_in_signature():
            props = set(reasoner.ind_object_properties(ind, direct=False))
            if connects in props:
                if list(reasoner.object_property_values(ind, connects, direct=False)):
                    has_any_assertion = True
                    break
        with self.subTest("has assertions"):
            self.assertTrue(has_any_assertion, "No OWL object property assertions found for 'connects'")

    def test_time_basic(self):
        builder = OCEDOntoBuilder("data/tmp/tmp.owl", self.settings)
        builder.instantiate_time_data(self.data)
        builder.save_onto("data/tmp/tmp.owl")

        onto, reasoner = self._load_reasoner()

        time_cls = OWLClass(IRI(NS, "Time"))
        with self.subTest("class exists"):
            self.assertIn(time_cls, list(onto.classes_in_signature()))

        with self.subTest("has instances"):
            time_instances = list(reasoner.instances(time_cls))
            self.assertTrue(len(time_instances) > 0, "No instances of Time were created")

        observed_at = OWLObjectProperty(IRI(NS, "observed_at"))
        with self.subTest("property exists"):
            self.assertIn(observed_at, list(onto.object_properties_in_signature()))

    def test_build_all(self):
        builder = OCEDOntoBuilder("data/tmp/tmp.owl", self.settings)
        builder.build_ontology(self.data)
        builder.save_onto("data/tmp/tmp.owl")

        onto, reasoner = self._load_reasoner()

        activity1 = OWLClass(IRI(NS, "activity1"))
        thing1 = OWLClass(IRI(NS, "thing1"))
        time_cls = OWLClass(IRI(NS, "Time"))

        for cls in [activity1, thing1, time_cls]:
            with self.subTest(f"class: {cls}"):
                self.assertIn(cls, list(onto.classes_in_signature()))

        with self.subTest("inst activity1"):
            self.assertTrue(list(reasoner.instances(activity1)), "No instances for activity1")
        with self.subTest("inst thing1"):
            self.assertTrue(list(reasoner.instances(thing1)), "No instances for thing1")
        with self.subTest("inst Time"):
            self.assertTrue(list(reasoner.instances(time_cls)), "No instances for Time")

    def test_declare_requires_file(self):
        builder = OCEDOntoBuilder("data/tmp/tmp.owl", self.settings)
        self.settings.order_representation = [EventOrderRepresentation.DECLARE]
        self.settings.declare_file = None

        with self.subTest("raises without file"):
            with self.assertRaises(ValueError):
                builder.instantiate_event_order(self.data)
            
    def test_declare_single_case(self):
        df = pd.DataFrame({
            "case_concept": ["thing1", "thing1", "thing1"],
            "trace_id": ["o1", "o3", "o7"],
            "constraintA": [True, False, True],
            "constraintB": [False, True, False],
        })
        declare_path = "tests/df_declare_testing.csv"
        df.to_csv(declare_path, index=False)

        self.settings.order_representation = [EventOrderRepresentation.DECLARE]
        self.settings.declare_file = declare_path

        builder = OCEDOntoBuilder("data/tmp/tmp.owl", self.settings)
        builder.instantiate_event_order(self.data)
        builder.save_onto("data/tmp/tmp.owl")

        onto, reasoner = self._load_reasoner()
        ns = "https://oced#"

        declare_constraint = OWLClass(IRI(ns, "DeclareConstraint"))
        constraintA = OWLClass(IRI(ns, "constraintA"))
        constraintB = OWLClass(IRI(ns, "constraintB"))

        with self.subTest("classes exist"):
            self.assertIn(declare_constraint, list(onto.classes_in_signature()))
            self.assertIn(constraintA, list(onto.classes_in_signature()))
            self.assertIn(constraintB, list(onto.classes_in_signature()))

        ind1 = OWLNamedIndividual(IRI(ns, "o1"))
        ind2 = OWLNamedIndividual(IRI(ns, "o3"))
        ind3 = OWLNamedIndividual(IRI(ns, "o7"))

        with self.subTest("constraintA instances"):
            self.assertIn(ind1, list(reasoner.instances(constraintA)))
            self.assertIn(ind3, list(reasoner.instances(constraintA)))
            self.assertNotIn(ind2, list(reasoner.instances(constraintA)))

        with self.subTest("constraintB instances"):
            self.assertIn(ind2, list(reasoner.instances(constraintB)))
            self.assertNotIn(ind1, list(reasoner.instances(constraintB)))
            self.assertNotIn(ind3, list(reasoner.instances(constraintB)))
        
    def test_declare_multi_case_raises(self):
        df = pd.DataFrame({
            "case_concept": ["thing1", "thing2", "thing1", "thing2"],
            "trace_id": ["o1", "o2", "o3", "o4"],
            "constraintA": [True, False, True, False],
            "constraintB": [False, True, False, True],
        })
        declare_path = "tests/df_declare_multi_case.csv"
        df.to_csv(declare_path, index=False)

        self.settings.order_representation = [EventOrderRepresentation.DECLARE]
        self.settings.declare_file = declare_path

        builder = OCEDOntoBuilder("data/tmp/tmp.owl", self.settings)

        with self.subTest("raises for multi-case"):
            with self.assertRaises(NotImplementedError):
                builder.instantiate_event_order(self.data)
        
        
    def test_class_hierarchy(self):
        builder = OCEDOntoBuilder("data/tmp/tmp.owl", self.settings)
        builder.instantiate_event_data(self.data)
        builder.instantiate_object_data(self.data)
        builder.save_onto("data/tmp/tmp.owl")

        onto, reasoner = self._load_reasoner()

        event_cls   = OWLClass(IRI(NS, "Event"))
        object_cls  = OWLClass(IRI(NS, "Object"))
        activity1   = OWLClass(IRI(NS, "activity1"))
        thing1      = OWLClass(IRI(NS, "thing1"))

        with self.subTest("classes exist"):
            for cls in [event_cls, object_cls, activity1, thing1]:
                self.assertIn(cls, list(onto.classes_in_signature()))

        with self.subTest("activity1 ⊑ Event"):
            super_of_activity1 = set(reasoner.super_classes(activity1, direct=False))
            self.assertIn(event_cls, super_of_activity1)

        with self.subTest("thing1 ⊑ Object"):
            super_of_thing1 = set(reasoner.super_classes(thing1, direct=False))
            self.assertIn(object_cls, super_of_thing1)
        
        
    def test_numeric_attrs_value_nodes(self):
        builder = OCEDOntoBuilder("data/tmp/tmp.owl", self.settings)
        builder.instantiate_object_data(self.data)
        builder.save_onto("data/tmp/tmp.owl")

        onto, reasoner = self._load_reasoner()

        # 1) attribute class exists
        value_cls = OWLClass(IRI(NS, "value"))
        with self.subTest("class exists"):
            self.assertIn(value_cls, list(onto.classes_in_signature()))

        # 2) value individuals exist
        inds = set(onto.individuals_in_signature())
        expected_inds = [
            "value_120.0", "value_605.4", "value_243.2", "value_530.3",
            "value_732.5", "value_416.6", "value_196.7", "value_136.8",
        ]
        with self.subTest("value individuals exist"):
            for name in expected_inds:
                self.assertIn(OWLNamedIndividual(IRI(NS, name)), inds, f"Missing {name}")

        # 3) specialized object property for the attribute exists
        #    (adjust the fallback name to whatever your generator produces if different)
        has_value_prop = OWLObjectProperty(IRI(NS, "has_value"))
        if has_value_prop not in list(onto.object_properties_in_signature()):
            has_value_prop = OWLObjectProperty(IRI(NS, "has_attribute_value"))

        with self.subTest("specialized has-<attr> property exists"):
            self.assertIn(has_value_prop, list(onto.object_properties_in_signature()))

        # 4) has_data_value data property exists
        has_data_value = OWLDataProperty(IRI(NS, "has_data_value"))
        with self.subTest("has_data_value exists"):
            self.assertIn(has_data_value, list(onto.data_properties_in_signature()))

        # 5) sample connectivity checks:
        #    o2 and o5 should both point to value_605.4 via the specialized property
        o2_ind = OWLNamedIndividual(IRI(NS, "o2"))
        o5_ind = OWLNamedIndividual(IRI(NS, "o5"))
        v605 = OWLNamedIndividual(IRI(NS, "value_605.4"))

        with self.subTest("o2/o5 link to value_605.4 via specialized property"):
            self.assertIn(v605, set(reasoner.object_property_values(o2_ind, has_value_prop, direct=False)))
            self.assertIn(v605, set(reasoner.object_property_values(o5_ind, has_value_prop, direct=False)))

        # 6) data literal attached to the value individual
        #    (We just assert that some literal is present; reasoners often wrap it.)
        with self.subTest("value individual has data literal"):
            # gather literals (implementation dependent; using reasoner’s data_property_values)
            literals = list(reasoner.data_property_values(v605, has_data_value, direct=False))
            self.assertTrue(len(literals) > 0, "No has_data_value literal on value_605.4")
        
    def test_string_attrs_value_nodes(self):
        # ensure resource column is present in the event df
        df = self.data._events["activity1"]["data"].copy()
        self.data._events["activity1"]["data"] = df

        builder = OCEDOntoBuilder("data/tmp/tmp.owl", self.settings)
        builder.instantiate_event_data(self.data)
        builder.save_onto("data/tmp/tmp.owl")

        onto, reasoner = self._load_reasoner()

        # 1) attribute class exists
        resource_cls = OWLClass(IRI(NS, "resource"))
        with self.subTest("attr class exists"):
            self.assertIn(resource_cls, list(onto.classes_in_signature()))

        # 2) value individuals exist and are typed as `resource`
        alice_ind = OWLNamedIndividual(IRI(NS, "resource_Alice"))
        bob_ind   = OWLNamedIndividual(IRI(NS, "resource_Bob"))
        inds = set(onto.individuals_in_signature())
        with self.subTest("value individuals exist"):
            self.assertIn(alice_ind, inds, "Expected individual 'resource_Alice' not found")
            self.assertIn(bob_ind,   inds, "Expected individual 'resource_Bob' not found")

        with self.subTest("value individuals typed as resource"):
            self.assertIn(alice_ind, set(reasoner.instances(resource_cls)))
            self.assertIn(bob_ind,   set(reasoner.instances(resource_cls)))

        # 3) specialized has-attribute property exists and is used
        #    (name depends on `generate_spezialized_has_attribute_name("resource")`;
        #     if your upper ontology names it "has_resource", this will be correct.)
        has_resource_prop = OWLObjectProperty(IRI(NS, "has_resource"))
        with self.subTest("has_resource property exists"):
            self.assertIn(has_resource_prop, list(onto.object_properties_in_signature()))

        # Spot-check that at least one event points to a resource value individual
        any_assertion = False
        for e_idx in range(10):
            ev_ind = OWLNamedIndividual(IRI(NS, f"e{e_idx}"))
            if ev_ind in inds:
                targets = set(reasoner.object_property_values(ev_ind, has_resource_prop, direct=False))
                if alice_ind in targets or bob_ind in targets:
                    any_assertion = True
                    break
        with self.subTest("events linked to resource value individuals"):
            self.assertTrue(any_assertion, "No has_resource assertions found from events to resource_* individuals")