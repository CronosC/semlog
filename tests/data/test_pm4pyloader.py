import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from semanticlog.data.pm4pyloader import PM4PYLoader
from semanticlog.data.ocelloader import OCELLoader
import os

pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class TestPM4PYLoader(unittest.TestCase):
    
    def test_load_all_pm4py(self):
        folder_path = "data/OCEL"
        for file_name in os.listdir(folder_path):
            if file_name not in ["AOE2.sqlite", "AOE2_small.json", "blockchain.xmlocel"]:
                with self.subTest(file=file_name): 
                    file_path = os.path.join(folder_path, file_name)
                    loader = PM4PYLoader(file_path, {})
                    oced = loader.get_oced()
                    self.assertIsNotNone(oced)
'''
class TestPM4PYLoader(unittest.TestCase):

    def test_load_ocel_sqlite_github_pm4py(self):
        loader = PM4PYLoader("data/github_pm4py.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        self.assertIsNotNone(oced)

    def test_load_ocel_sqlite_o2c(self):
        loader = PM4PYLoader("data/o2c.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        self.assertIsNotNone(oced)

    def test_load_ocel_sqlite_ocel2_p2p(self):
        loader = PM4PYLoader("data/ocel2-p2p.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        self.assertIsNotNone(oced)

    def test_load_ocel_sqlite_order_management(self):
        loader = PM4PYLoader("data/order-management.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        self.assertIsNotNone(oced)

    def test_load_ocel_sqlite_p2p(self):
        loader = PM4PYLoader("data/p2p.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        self.assertIsNotNone(oced)

    def test_load_ocel_sqlite_recruiting(self):
        loader = PM4PYLoader("data/recruiting.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        self.assertIsNotNone(oced)

    def test_load_ocel_sqlite_running_example(self):
        loader = PM4PYLoader("data/running-example.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        self.assertIsNotNone(oced)

    def test_load_ocel_sqlite_windows_events(self):
        loader = PM4PYLoader("data/windows_events.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        self.assertIsNotNone(oced)

    def test_load_ocel_sqlite_transfer_order(self):
        loader = PM4PYLoader("data/transfer_order.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        self.assertIsNotNone(oced)
'''
 

if __name__ == "__main__":
    unittest.main()
