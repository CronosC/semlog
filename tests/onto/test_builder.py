import unittest
from owlapy.class_expression import OWLClassExpression
from semanticlog.onto.builder import OCEDBaseBuilder
from owlapy.iri import IRI
import time
import pandas as pd


class TestBuilder(unittest.TestCase):

    def test_add_class(self):
        builder = OCEDBaseBuilder("data/test.owl#")
        self.assertEqual(len(list(builder.onto.classes_in_signature())), 0)
        builder.add_class("Human")
        builder.add_class("Parent", "Human")
        builder.add_class("Male")
        builder.add_class("Man", "Human and Male")
        self.assertEqual(len(list(builder.onto.classes_in_signature())), 4)

    def test_add_classes(self):
        builder = OCEDBaseBuilder("data/test.owl#")
        builder.add_classes(["Human", "Male", "Female"])
        builder.add_classes(["Man", "Women"], ["Human and Male", "Human and Female"])
        builder.add_classes(["Child"], [None])
        self.assertEqual(len(list(builder.onto.classes_in_signature())), 6)
        
    def test_assert_individual(self):
        builder = OCEDBaseBuilder("data/test.owl#")
        individuals =  ["a", "b", "c", "d", "e"]
        builder.add_class("Letter")
        for ind in individuals:
            builder.assert_individual(ind, "Letter")
        self.assertEqual(len(list(builder.onto.classes_in_signature())), 1)
        self.assertEqual(len(list(builder.onto.individuals_in_signature())), 5)  
        builder.save_onto("data/temp.owl")
        
    def test_assert_individuals(self):
        builder = OCEDBaseBuilder("data/test.owl#")
        individuals =  ["a", "b", "c", "d", "e"]
        builder.add_class("Letter")
        builder.assert_individuals(individuals, "Letter")
        self.assertEqual(len(list(builder.onto.classes_in_signature())), 1)
        self.assertEqual(len(list(builder.onto.individuals_in_signature())), 5)  
        builder.save_onto("data/temp.owl")
        
    def test_assert_individuals_scale(self):
        builder = OCEDBaseBuilder("data/test.owl#")
        individuals = [str(i) for i in range(100000)]
        builder.add_class("Letter")
        t0 = time.time()
        builder.assert_individuals(individuals, "Letter")
        t1 = time.time()
        print(f"\nAdding {len(individuals)} individuals took: {str(t1-t0)}s")
        self.assertEqual(len(list(builder.onto.classes_in_signature())), 1)
        self.assertEqual(len(list(builder.onto.individuals_in_signature())), len(individuals))    


    def test_add_property(self):
        builder = OCEDBaseBuilder("data/test.owl#")
        builder.add_property("isRelated")
        builder.add_class("Parent")
        builder.add_class("Child")
        builder.add_class("Boy", superclass="Child")
        self.assertEqual(len(list(builder.onto.classes_in_signature())), 3)
        builder.add_property(
            "hasChild",
            domain_class="Parent",
            range_class="Child",
            create_inverse=True,
            inverse_name="hasParent",
        )
        builder.add_property(
            "hasMaleChild",
            domain_class="Parent",
            range_class="Boy",
            create_inverse=True,
            inverse_name="hasParent",
        )
        builder.add_property(
            "hasAge", domain_class="Child", range_class="integer", data_prop=True
        )
        self.assertEqual(len(list(builder.onto.properties_in_signature())), 5)
        self.assertEqual(len(list(builder.onto.data_properties_in_signature())), 1)
        
    def test_assert_relation(self):
        builder = OCEDBaseBuilder("data/test.owl#")
        builder.add_class("Person")
        individuals =  ["Alice", "Bob"]
        builder.assert_individuals(individuals, "Person")
        builder.add_property(
            "knows",
            domain_class="Person",
            range_class="Person",
            create_inverse=False,
        )
        builder.assert_relation("knows", "Alice", "Bob")
        builder.add_property(
            "hasAge",
            domain_class="Person",
            range_class="integer",
            data_prop=True
        )
        builder.assert_relation("hasAge", "Alice", 25, data_prop=True)
        builder.assert_relation("hasAge", "Bob", 34, data_prop=True)
        builder.save_onto("data/temp.owl") 
        
    def test_assert_relations(self):
        builder = OCEDBaseBuilder("data/test.owl#")
        builder.add_class("Number")
        individuals = [str(i) for i in range(100)]
        individuals_int = [i for i in range(100)]
        builder.assert_individuals(individuals, "Letter")
 
        
        builder.add_property(
            "identity",
            domain_class="Number",
            range_class="Number",
        )
        relation_data = pd.DataFrame({"from": individuals, "to": individuals})
        builder.assert_relations(relation_data, from_key="from", to_key="to", prop="identity")
        
        builder.add_property(
            "hasValue",
            domain_class="Number",
            range_class="integer",
            data_prop=True
        )
        relation_data_int = pd.DataFrame({"from": individuals, "to": individuals_int})
        builder.assert_relations(relation_data_int, from_key="from", to_key="to", prop="hasValue", data_prop=True)
        
    def test_assert_relations_scale(self):
        builder = OCEDBaseBuilder("data/test.owl#")
        builder.add_class("Number")
        individuals = [str(i) for i in range(10000)]
        individuals_int = [i for i in range(10000)]
        
        t0 = time.time()
        builder.assert_individuals(individuals, "Letter")
        t1 = time.time()
        print(f"\nAdding {len(individuals)} individuals took: {str(t1-t0)}s")
        
        builder.add_property(
            "identity",
            domain_class="Number",
            range_class="Number",
        )
        relation_data = pd.DataFrame({"from": individuals, "to": individuals})
        t0 = time.time()
        builder.assert_relations(relation_data, from_key="from", to_key="to", prop="identity")
        t1 = time.time()
        print(f"\nAdding {len(individuals)} relations took: {str(t1-t0)}s")
        
        builder.add_property(
            "hasValue",
            domain_class="Number",
            range_class="integer",
            data_prop=True
        )
        relation_data_int = pd.DataFrame({"from": individuals, "to": individuals_int})
        t0 = time.time()
        builder.assert_relations(relation_data_int, from_key="from", to_key="to", prop="hasValue", data_prop=True)
        t1 = time.time()
        print(f"\nAdding {len(individuals)} data relations took: {str(t1-t0)}s")
        
        builder.save_onto("data/temp.owl")

if __name__ == "__main__":
    unittest.main()
