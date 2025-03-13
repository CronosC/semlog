import unittest
import pandas as pd
from datetime import datetime

# Import the function to test.
# Adjust the import according to your project structure.
from semanticlog.data.misc import get_weighted_variant_mapping

class TestGetWeightedVariantMapping(unittest.TestCase):
    def test_variant_mapping_normal(self):
        # Create an unsorted sample event log.
        data = [
            {"case:concept:name": "case2", "time:timestamp": datetime(2021, 1, 2, 12, 0), "concept:name": "C", "event": 1},
            {"case:concept:name": "case1", "time:timestamp": datetime(2021, 1, 1, 8, 0), "concept:name": "A", "event": 2},
            {"case:concept:name": "case1", "time:timestamp": datetime(2021, 1, 1, 9, 0), "concept:name": "B", "event": 3},
            {"case:concept:name": "case2", "time:timestamp": datetime(2021, 1, 2, 10, 0), "concept:name": "A", "event": 4},
            {"case:concept:name": "case2", "time:timestamp": datetime(2021, 1, 2, 11, 0), "concept:name": "B", "event": 5},
            {"case:concept:name": "case3", "time:timestamp": datetime(2021, 1, 1, 8, 0), "concept:name": "A", "event": 6},
            {"case:concept:name": "case3", "time:timestamp": datetime(2021, 1, 1, 9, 0), "concept:name": "B", "event": 7},
        ]
        log = pd.DataFrame(data)
        
        # Call the function under test.
        result = get_weighted_variant_mapping(log)
        
        # Expected: for case1 the activity sequence is ("A", "B")
        # and for case2 the sequence is ("A", "B", "C").
        print("#"*50)
        print(result)
        print("#"*50)
        expected = {
            ("A", "B"): 2,
            ("A", "B", "C"): 1
        }
        
        # Check that the returned variants mapping matches our expectation.
        # (Using assertEqual directly for dicts; if order does not matter, you could also compare keys and values separately.)
        self.assertEqual(result, expected)

    def test_variant_mapping_empty(self):
        # Test that an empty log returns an empty mapping.
        log = pd.DataFrame(columns=["case:concept:name", "time:timestamp", "concept:name", "event"])
        result = get_weighted_variant_mapping(log)
        expected = {}
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()