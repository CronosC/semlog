import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from semanticlog.data.oced import OCEDStore


class TestOCEDStore(unittest.TestCase):

    def setUp(self):
        # Initialize the OCEDStore object and sample data.
        self.oced = OCEDStore()
        self.event_1= pd.DataFrame(
            {
                "event": ["e1", "e2", "e5"],
                "timestamp": pd.date_range("2022-01-01", periods=3, freq="D"),
            }
        )
        self.event_2= pd.DataFrame(
            {
                "event": ["e3", "e4"],
                "timestamp": pd.date_range("2022-01-01", periods=2, freq="D"),
            }
        )
        self.object_A_type = "A"
        self.object_A_data = pd.DataFrame(
            {"object": [1, 2, 3], "object_attribute": ["attr1", "attr2", "attr3"]}
        )
        self.object_B_type = "B"
        self.object_B_data = pd.DataFrame(
            {"object": [4, 5], "object_attribute": ["attr4", "attr5"]}
        )
        self.e2o_relation_data = pd.DataFrame(
            {"from_id": ["e1", "e2", "e3", "e4", "e5"], "to_id": [1, 1, 2, 2, 2]}
        )
        self.o2o_relation_data = pd.DataFrame(
            {"from_id": [1, 1, 4, 5], "to_id": [1, 3, 2, 1]}
        )

    def test_add_and_get_events(self):
        # Add events and verify stored metadata and data.
        self.oced.add_event("type_1", "event", "timestamp", self.event_1)
        events = self.oced.get_events("type_1")
        assert_frame_equal(events["data"], self.event_1)

    def test_add_event_invalid_columns(self):
        # Missing a required column should trigger an error.
        invalid_data = self.event_1.drop("timestamp", axis=1)
        with self.assertRaises(ValueError):
            self.oced.add_event("type1", "event", "timestamp", invalid_data)

    def test_events_overwrite(self):
        self.oced.add_event("type2", "event", "timestamp", self.event_2)
        # Duplicate add without overwrite should error.
        with self.assertRaises(ValueError):
            self.oced.add_event("type2", "event", "timestamp", self.event_2)
        # With overwrite, it should update.
        self.oced.add_event(
            "type2", "event", "timestamp", self.event_2, overwrite=True
        )
        events = self.oced.get_events("type2")
        assert_frame_equal(events["data"], self.event_2)

    def test_add_and_get_objects(self):
        # Initially, objects should be empty.
        self.assertEqual(self.oced.get_objects(), {})
        # Add object A and B, then verify.
        self.oced.add_object(self.object_A_type, "object", self.object_A_data)
        obj_A = self.oced.get_object(self.object_A_type)
        self.assertEqual(obj_A["id"], "object")
        assert_frame_equal(obj_A["data"], self.object_A_data)
        self.oced.add_object(self.object_B_type, "object", self.object_B_data)
        objects = self.oced.get_objects()
        self.assertIn(self.object_A_type, objects)
        self.assertIn(self.object_B_type, objects)
        object_types = list(self.oced.get_object_types())
        self.assertCountEqual(object_types, [self.object_A_type, self.object_B_type])
        # Adding duplicate object without overwrite should error.
        with self.assertRaises(ValueError):
            self.oced.add_object(self.object_A_type, "object", self.object_A_data)

    def test_add_event_to_object_relation(self):
        # Object must exist for relation.
        self.oced.add_object(self.object_A_type, "object", self.object_A_data)
        # Add a valid event-to-object relation.
        self.oced.add_event_to_object_relation(
            "rel1", "from_id", "to_id", self.object_A_type, self.e2o_relation_data
        )
        rel = self.oced.get_relation("rel1")
        self.assertEqual(rel["from_id"], "from_id")
        self.assertEqual(rel["to_id"], "to_id")
        self.assertEqual(rel["to_type"], self.object_A_type)
        assert_frame_equal(rel["data"], self.e2o_relation_data)
        # Duplicate relation addition should error.
        with self.assertRaises(ValueError):
            self.oced.add_event_to_object_relation(
                "rel1", "from_id", "to_id", self.object_A_type, self.e2o_relation_data
            )
        # Unknown object type for relation.
        with self.assertRaises(ValueError):
            self.oced.add_event_to_object_relation(
                "rel2", "from_id", "to_id", "Unknown", self.e2o_relation_data
            )

    def test_add_object_to_object_relation(self):
        # Add both objects.
        self.oced.add_object(self.object_A_type, "object", self.object_A_data)
        self.oced.add_object(self.object_B_type, "object", self.object_B_data)
        # Add a valid object-to-object relation.
        self.oced.add_object_to_object_relation(
            "rel_o2o",
            "from_id",
            self.object_A_type,
            "to_id",
            self.object_B_type,
            self.o2o_relation_data,
        )
        rel = self.oced.get_relation("rel_o2o")
        self.assertEqual(rel["from_id"], "from_id")
        self.assertEqual(rel["from_type"], self.object_A_type)
        self.assertEqual(rel["to_id"], "to_id")
        self.assertEqual(rel["to_type"], self.object_B_type)
        assert_frame_equal(rel["data"], self.o2o_relation_data)
        # Duplicate addition should error.
        with self.assertRaises(ValueError):
            self.oced.add_object_to_object_relation(
                "rel_o2o",
                "from_id",
                self.object_A_type,
                "to_id",
                self.object_B_type,
                self.o2o_relation_data,
            )
        # Unknown object type in relation.
        with self.assertRaises(ValueError):
            self.oced.add_object_to_object_relation(
                "rel_invalid",
                "from_id",
                "X",
                "to_id",
                self.object_B_type,
                self.o2o_relation_data,
            )

    def test_get_log(self):
        # Setup events, object, and relation for log generation.
        self.oced.add_event("type1", "event", "timestamp", self.event_1)
        self.oced.add_event("type2", "event", "timestamp", self.event_2)
        self.oced.add_object(self.object_A_type, "object", self.object_A_data)
        self.oced.add_event_to_object_relation(
            "rel_log", "event", "object", self.object_A_type, self.e2o_relation_data
        )
        # Get log and verify its type and expected columns.
        log = self.oced.get_log(self.object_A_type)

        self.assertIsInstance(log, pd.DataFrame)
        # Check for pm4py formatted column names.
        self.assertIn("case:concept:name", log.columns)
        self.assertIn("concept:name", log.columns)
        self.assertIn("time:timestamp", log.columns)
        
    def test_get_type(self):
        # Convert object_A_data id column to strings for compatibility
        self.object_A_data["object"] = self.object_A_data["object"].astype(str)
        # Add an object with type "A" and an event with type "event_type".
        self.oced.add_object(self.object_A_type, "object", self.object_A_data)
        self.oced.add_event("event_type", "event", "timestamp", self.event_1)

        # Test get_type with an id present in the object dataset.
        self.assertEqual(self.oced.get_type(["1"]), self.object_A_type)

        # Test get_type with an id present in the event dataset.
        self.assertEqual(self.oced.get_type(["e2"]), "event_type")

        # Test get_type with an id not present in any dataset.
        self.assertIsNone(self.oced.get_type(["nonexistent"]))

if __name__ == "__main__":
    unittest.main()
