import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from semanticlog.data.ocelloader import OCELLoader

pd.set_option('display.max_columns', None)

class TestOCEDStore(unittest.TestCase):

    def test_load_ocel_sqlite_github_pm4py(self):
        loader = OCELLoader("data/github_pm4py.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        self.assertIsNotNone(oced)

    def test_load_ocel_sqlite_o2c(self):
        loader = OCELLoader("data/o2c.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        self.assertIsNotNone(oced)

    def test_load_ocel_sqlite_ocel2_p2p(self):
        loader = OCELLoader("data/ocel2-p2p.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        self.assertIsNotNone(oced)

    def test_load_ocel_sqlite_order_management(self):
        loader = OCELLoader("data/order-management.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        self.assertIsNotNone(oced)

    def test_load_ocel_sqlite_p2p(self):
        loader = OCELLoader("data/p2p.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        self.assertIsNotNone(oced)

    def test_load_ocel_sqlite_recruiting(self):
        loader = OCELLoader("data/recruiting.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        self.assertIsNotNone(oced)

    def test_load_ocel_sqlite_running_example(self):
        loader = OCELLoader("data/running-example.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        self.assertIsNotNone(oced)

    def test_load_ocel_sqlite_windows_events(self):
        loader = OCELLoader("data/windows_events.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        self.assertIsNotNone(oced)

    def test_load_ocel_sqlite_transfer_order(self):
        loader = OCELLoader("data/transfer_order.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        self.assertIsNotNone(oced)
        
 

if __name__ == "__main__":
    unittest.main()
