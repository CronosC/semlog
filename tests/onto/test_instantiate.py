import unittest
import os
from semanticlog.onto.instantiate import OCEDOntoBuilder
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.upper import OCEDUpperOntoBuilder
from datetime import datetime, timedelta
import pandas as pd
import random


class TestOntoBuilder(unittest.TestCase):

    def setUp(self):
        if os.path.exists("data/tmp.owl"):
            os.remove("data/tmp.owl")
        builder = OCEDUpperOntoBuilder("https:/oced#")
        builder.construct_upper_ontology("data/tmp.owl")

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
                        "resource": random.choices(["Alice", "Bob"], k=10),
                        "number": [random.uniform(0, 100) for _ in range(10)],
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
                        "type": random.choices(["Small", "Medium", "Large"], k=10),
                        "number": [random.uniform(0, 100) for _ in range(10)],
                    }
                ),
            }
        }
        self.data._relations = {
            "event_to_object": {
                "does": {
                    "from_id": "event_id",
                    "to_id": "object_id",
                    "to_type": "thing1",
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
                    "from_type": "thing1",
                    "to_id": "object_id2",
                    "to_type": "thing1",
                    "data": pd.DataFrame(
                        {
                            "object_id1": ["o" + str(random.randint(1, 10)) for _ in range(10)],
                            "object_id2": ["o" + str(random.randint(1, 10)) for _ in range(10)],
                        }
                    ),
                }
            },
        }

    def test_instantiate_events(self):
        builder = OCEDOntoBuilder("data/tmp.owl")
        builder.instantiate_event_data(self.data)
        builder.save_onto("data/tmp.owl")

    def test_instantiate_objects(self):
        builder = OCEDOntoBuilder("data/tmp.owl")
        builder.instantiate_object_data(self.data)
        builder.save_onto("data/tmp.owl")
        
    def test_instantiate_e2o(self):
        builder = OCEDOntoBuilder("data/tmp.owl")
        builder.instantiate_event_data(self.data)
        builder.instantiate_object_data(self.data)
        builder.instantiate_event_object_relation_data(self.data)
        builder.save_onto("data/tmp.owl")
        
    def test_instantiate_o2o(self):
        builder = OCEDOntoBuilder("data/tmp.owl")
        builder.instantiate_event_data(self.data)
        builder.instantiate_object_data(self.data)
        builder.instantiate_object_to_object_relation_data(self.data)
        builder.save_onto("data/tmp.owl")
        
    def test_instantiate_time(self):
        builder = OCEDOntoBuilder("data/tmp.owl")
        builder.instantiate_time_data(self.data)
        builder.save_onto("data/tmp.owl")
        
        
    def test_instantiate_all(self):
        builder = OCEDOntoBuilder("data/tmp.owl")
        builder.build_ontology(self.data)
        builder.save_onto("data/tmp.owl")


if __name__ == "__main__":
    unittest.main()
