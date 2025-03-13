import unittest
import pandas as pd
import random
from datetime import datetime
from unittest.mock import patch
from dateutil import tz
from semanticlog.data.filtering import *

# --- Test Suite ---
class TestPM4pyLogFilters(unittest.TestCase):

    def test_filter_to_timeintervall_timezone_aware(self):
        # Create a timezone aware DataFrame.
        # Use UTC timestamps.
        base_time = pd.Timestamp("2023-03-15 12:00:00", tz="UTC")
        times = [
            base_time - pd.Timedelta(minutes=30),
            base_time + pd.Timedelta(minutes=30),
            base_time + pd.Timedelta(hours=1)
        ]
        log = pd.DataFrame({
            "time:timestamp": pd.to_datetime(times, utc=True),
            "case:concept:name": ["case1", "case2", "case3"]
        })
        # Define mintime and maxtime (as unixtime seconds)
        mintime = int((base_time - pd.Timedelta(minutes=15)).timestamp())
        maxtime = int((base_time + pd.Timedelta(minutes=45)).timestamp())
        filtered_log = filter_to_timeintervall(log, mintime, maxtime)
        # Only the row with time base_time + 30 minutes should remain.
        self.assertEqual(len(filtered_log), 1)
        self.assertEqual(filtered_log.iloc[0]["case:concept:name"], "case2")

    def test_filter_to_timeintervall_timezone_naive(self):
        # Create a timezone naive DataFrame.
        base_time = pd.Timestamp("2023-03-15 12:00:00")
        times = [
            base_time - pd.Timedelta(minutes=30),
            base_time + pd.Timedelta(minutes=30),
            base_time + pd.Timedelta(hours=1)
        ]
        log = pd.DataFrame({
            "time:timestamp": pd.to_datetime(times),  # naive timestamps
            "case:concept:name": ["case1", "case2", "case3"]
        })
        mintime = int((base_time - pd.Timedelta(minutes=15)).timestamp())
        maxtime = int((base_time + pd.Timedelta(minutes=45)).timestamp())
        filtered_log = filter_to_timeintervall(log, mintime, maxtime)
        self.assertEqual(len(filtered_log), 1)
        self.assertEqual(filtered_log.iloc[0]["case:concept:name"], "case2")

    def test_filter_remove_months(self):
        # Create a DataFrame with events across different months.
        dates = pd.to_datetime([
            "2023-01-15", "2023-02-20", "2023-03-10", "2023-04-05"
        ])
        log = pd.DataFrame({
            "time:timestamp": dates,
            "case:concept:name": ["case1", "case2", "case3", "case1"]
        })
        # Remove events from March (3) and April (4)
        filtered_log = filter_remove_months(log, months=[3, 4])
        # case3 appears in March and case1 appears in April (as well as January),
        # so both should be removed; only case2 remains.
        self.assertTrue("case2" in filtered_log["case:concept:name"].unique())
        self.assertFalse("case1" in filtered_log["case:concept:name"].unique())
        self.assertFalse("case3" in filtered_log["case:concept:name"].unique())
        
if __name__ == '__main__':
    unittest.main()