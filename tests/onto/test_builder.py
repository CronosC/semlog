import tempfile
import time
import unittest
from pathlib import Path

import pandas as pd

from semanticlog.onto.builder import OCEDBaseBuilder
from semanticlog.onto.settings import OntoBuildSettings, OwlProfile


class TestBuilder(unittest.TestCase):
    def setUp(self):
        self.settings = OntoBuildSettings()
        self.settings.profile = OwlProfile.OWL2DL
        self.settings.order_representation = []
        self.settings.case_concepts = []
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out_path = str(Path(self._tmp.name) / "temp.owl")

    def test_add_class(self):
        builder = OCEDBaseBuilder("https://test#", self.settings)
        self.assertEqual(len(list(builder.onto.classes_in_signature())), 0)
        builder.add_class("Human")
        builder.add_class("Parent", "Human")
        builder.add_class("Male")
        builder.add_class("Man", "Human and Male")
        self.assertEqual(len(list(builder.onto.classes_in_signature())), 4)

    def test_add_classes(self):
        builder = OCEDBaseBuilder("https://test#", self.settings)
        builder.add_classes(["Human", "Male", "Female"])
        builder.add_classes(["Man", "Women"], ["Human and Male", "Human and Female"])
        builder.add_classes(["Child"], [None])
        self.assertEqual(len(list(builder.onto.classes_in_signature())), 6)

    def test_assert_individual(self):
        builder = OCEDBaseBuilder("https://test#", self.settings)
        individuals = ["a", "b", "c", "d", "e"]
        builder.add_class("Letter")
        for ind in individuals:
            builder.assert_individual(ind, "Letter")
        self.assertEqual(len(list(builder.onto.classes_in_signature())), 1)
        self.assertEqual(len(list(builder.onto.individuals_in_signature())), 5)
        builder.save_onto(self.out_path)

    def test_assert_individuals(self):
        builder = OCEDBaseBuilder("https://test#", self.settings)
        individuals = ["a", "b", "c", "d", "e"]
        builder.add_class("Letter")
        builder.assert_individuals(individuals, "Letter")
        self.assertEqual(len(list(builder.onto.classes_in_signature())), 1)
        self.assertEqual(len(list(builder.onto.individuals_in_signature())), 5)
        builder.save_onto(self.out_path)

    def test_assert_individuals_scale(self):
        builder = OCEDBaseBuilder("https://test#", self.settings)
        individuals = [str(i) for i in range(100000)]
        builder.add_class("Letter")
        t0 = time.time()
        builder.assert_individuals(individuals, "Letter")
        t1 = time.time()
        print(f"\nAdding {len(individuals)} individuals took: {str(t1 - t0)}s")
        self.assertEqual(len(list(builder.onto.classes_in_signature())), 1)
        self.assertEqual(
            len(list(builder.onto.individuals_in_signature())), len(individuals)
        )

    def test_add_property(self):
        builder = OCEDBaseBuilder("https://test#", self.settings)
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
        builder = OCEDBaseBuilder("https://test#", self.settings)
        builder.add_class("Person")
        individuals = ["Alice", "Bob"]
        builder.assert_individuals(individuals, "Person")
        builder.add_property(
            "knows",
            domain_class="Person",
            range_class="Person",
            create_inverse=False,
        )
        builder.assert_relation("knows", "Alice", "Bob")
        builder.add_property(
            "hasAge", domain_class="Person", range_class="integer", data_prop=True
        )
        builder.assert_relation("hasAge", "Alice", 25, data_prop=True)
        builder.assert_relation("hasAge", "Bob", 34, data_prop=True)
        builder.save_onto(self.out_path)

    def test_assert_relations(self):
        builder = OCEDBaseBuilder("https://test#", self.settings)
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
        builder.assert_relations(
            relation_data, from_key="from", to_key="to", prop="identity"
        )

        builder.add_property(
            "hasValue", domain_class="Number", range_class="integer", data_prop=True
        )
        relation_data_int = pd.DataFrame({"from": individuals, "to": individuals_int})
        builder.assert_relations(
            relation_data_int,
            from_key="from",
            to_key="to",
            prop="hasValue",
            data_prop=True,
        )

    def test_assert_relations_scale(self):
        builder = OCEDBaseBuilder("https://test#", self.settings)
        builder.add_class("Number")
        individuals = [str(i) for i in range(10000)]
        individuals_int = [i for i in range(10000)]

        t0 = time.time()
        builder.assert_individuals(individuals, "Letter")
        t1 = time.time()
        print(f"\nAdding {len(individuals)} individuals took: {str(t1 - t0)}s")

        builder.add_property(
            "identity",
            domain_class="Number",
            range_class="Number",
        )
        relation_data = pd.DataFrame({"from": individuals, "to": individuals})
        t0 = time.time()
        builder.assert_relations(
            relation_data, from_key="from", to_key="to", prop="identity"
        )
        t1 = time.time()
        print(f"\nAdding {len(individuals)} relations took: {str(t1 - t0)}s")

        builder.add_property(
            "hasValue", domain_class="Number", range_class="integer", data_prop=True
        )
        relation_data_int = pd.DataFrame({"from": individuals, "to": individuals_int})
        t0 = time.time()
        builder.assert_relations(
            relation_data_int,
            from_key="from",
            to_key="to",
            prop="hasValue",
            data_prop=True,
        )
        t1 = time.time()
        print(f"\nAdding {len(individuals)} data relations took: {str(t1 - t0)}s")

        builder.save_onto(self.out_path)

    def test_assert_data_relations_with_datetimes(self):
        builder = OCEDBaseBuilder("https://test#", self.settings)
        # Class + individuals
        builder.add_class("Event")
        builder.assert_individuals(["e1", "e2"], "Event")

        # Data property without explicit range (allowed by implementation)
        builder.add_property(
            "occurredAt",
            domain_class="Event",
            range_class=None,
            data_prop=True,
        )

        # Datetime column as pandas datetime64
        df = pd.DataFrame(
            {
                "from": ["e1", "e2"],
                "to": pd.date_range("2022-01-01", periods=2, freq="D"),  # datetime64
            }
        )

        # Should NOT raise;
        builder.assert_relations(
            df, prop="occurredAt", from_key="from", to_key="to", data_prop=True
        )

        # If we got here, saving should also not raise
        builder.save_onto(self.out_path)

    def test_add_property_validation(self):
        builder = OCEDBaseBuilder("https://test#", self.settings)
        builder.add_class("A")
        builder.add_class("B")

        # 1) inverse_name without create_inverse -> error
        with self.assertRaises(ValueError):
            builder.add_property(
                "p_bad_inv_name",
                domain_class="A",
                range_class="B",
                create_inverse=False,
                inverse_name="p_bad_inv",  # illegal combo
            )

        # 2) data property cannot have inverse -> error
        with self.assertRaises(ValueError):
            builder.add_property(
                "age_with_inverse",
                domain_class="A",
                range_class="integer",
                data_prop=True,
                create_inverse=True,
            )

        # 3) object prop cannot have a data range -> error
        with self.assertRaises(ValueError):
            builder.add_property(
                "p_obj_with_data_range",
                domain_class="A",
                range_class="integer",  # data range on object prop
                data_prop=False,
            )

        # 4) data prop with unknown datatype -> error
        with self.assertRaises(ValueError):
            builder.add_property(
                "p_data_with_unknown_range",
                domain_class="A",
                range_class="stringy",  # not in {"integer","double","boolean"}
                data_prop=True,
            )

        # 5) happy path: object prop with inverse (auto name), and data prop with known range
        builder.add_property(
            "relatesTo", domain_class="A", range_class="B", create_inverse=True
        )
        builder.add_property(
            "hasAge", domain_class="A", range_class="integer", data_prop=True
        )

        # We should now have: relatesTo + its inverse + hasAge in signatures
        obj_props = list(builder.onto.object_properties_in_signature())
        data_props = list(builder.onto.data_properties_in_signature())
        self.assertGreaterEqual(len(obj_props), 2)  # prop + inverse
        self.assertGreaterEqual(len(data_props), 1)  # hasAge present


if __name__ == "__main__":
    unittest.main()
