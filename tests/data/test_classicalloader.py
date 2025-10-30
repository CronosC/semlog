import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from semanticlog.data.pm4pyloader import PM4PYLoader
from semanticlog.data.ocelloader import OCELLoader
from semanticlog.data.classicalloader import ClassicalLoader
import os

pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class TestClassicalLoader(unittest.TestCase):
    
    def test_load_classical(self):
        path = "simulated.csv"
        loader = ClassicalLoader(path, {"object_cols": ["resource"]})
        oced = loader.get_oced()
        print(oced._events)
        self.assertIsNotNone(oced._events)
        self.assertIsNotNone(oced._objects)
        self.assertIsNotNone(oced._relations)


if __name__ == "__main__":
    unittest.main()
