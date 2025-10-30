import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from semanticlog.data.oced import OCEDStore
import numpy as np
from pandas.testing import assert_series_equal

class TestOCEDStore(unittest.TestCase):

    def setUp(self):
        # Initialize the OCEDStore object and sample data.
        self.oced = OCEDStore()
        self.event_1= pd.DataFrame(
            {
                "event": ["e1", "e2", "e5"],
                "timestamp": pd.date_range("2022-01-01", periods=3, freq="D", tz="UTC"),
            }
        )
        self.event_2= pd.DataFrame(
            {
                "event": ["e3", "e4"],
                "timestamp": pd.date_range("2022-01-01", periods=2, freq="D", tz="UTC"),
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
        
    def test_rename_relation(self):
        self.oced.add_event("type1", "event", "timestamp", self.event_1)
        self.oced.add_event("type2", "event", "timestamp", self.event_2)
        self.oced.add_object(self.object_A_type, "object", self.object_A_data)
        self.oced.add_object(self.object_B_type, "object", self.object_B_data)
        self.oced.add_event_to_object_relation("rel1", "from_id", "to_id", [self.object_A_type], self.e2o_relation_data)
        self.oced.add_object_to_object_relation("rel2", "from_id", [self.object_A_type], "to_id", [self.object_B_type], self.o2o_relation_data)
        
        self.oced.rename_relation('rel1', 'rel1_new')
        self.assertIn('rel1_new', self.oced._relations['event_to_object'])
        self.assertNotIn('rel1', self.oced._relations['event_to_object'])
        with self.assertRaises(ValueError):
            self.oced.rename_relation('nonexistent_rel', 'rel_new')

    def test_rename_object(self):
        self.oced.add_event("type1", "event", "timestamp", self.event_1)
        self.oced.add_event("type2", "event", "timestamp", self.event_2)
        self.oced.add_object(self.object_A_type, "object", self.object_A_data)
        self.oced.add_object(self.object_B_type, "object", self.object_B_data)
        self.oced.add_event_to_object_relation("rel1", "from_id", "to_id", [self.object_A_type], self.e2o_relation_data)
        self.oced.add_object_to_object_relation("rel2", "from_id", [self.object_A_type], "to_id", [self.object_B_type], self.o2o_relation_data)
        
        self.oced.rename_object(self.object_A_type, 'C')
        self.assertIn('C', self.oced._objects)
        self.assertNotIn(self.object_A_type, self.oced._objects)

        for relation in self.oced._relations['event_to_object'].values():
            self.assertNotIn(self.object_A_type, relation['to_type'])
        for relation in self.oced._relations['object_to_object'].values():
            self.assertNotIn(self.object_A_type, relation['from_type'])
            self.assertNotIn(self.object_A_type, relation['to_type'])
        with self.assertRaises(ValueError):
            self.oced.rename_object('nonexistent_object', 'object_new')

    def test_rename_activity(self):
        self.oced.add_event("type1", "event", "timestamp", self.event_1)
        self.oced.add_event("type2", "event", "timestamp", self.event_2)
        self.oced.add_object(self.object_A_type, "object", self.object_A_data)
        self.oced.add_object(self.object_B_type, "object", self.object_B_data)
        self.oced.add_event_to_object_relation("rel1", "from_id", "to_id", [self.object_A_type], self.e2o_relation_data)
        self.oced.add_object_to_object_relation("rel2", "from_id", [self.object_A_type], "to_id", [self.object_B_type], self.o2o_relation_data)
        
        self.oced.rename_activity('type1', 'type1_new')
        self.assertIn('type1_new', self.oced._events)
        self.assertNotIn('type1', self.oced._events)

        with self.assertRaises(ValueError):
            self.oced.rename_activity('nonexistent_activity', 'activity_new')

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
            "rel1", "from_id", "to_id", [self.object_A_type], self.e2o_relation_data
        )
        rel = self.oced.get_relation("rel1")
        self.assertEqual(rel["from_id"], "from_id")
        self.assertEqual(rel["to_id"], "to_id")
        self.assertEqual(rel["to_type"], [self.object_A_type])
        assert_frame_equal(rel["data"], self.e2o_relation_data)
        # Duplicate relation addition should error.
        with self.assertRaises(ValueError):
            self.oced.add_event_to_object_relation(
                "rel1", "from_id", "to_id", [self.object_A_type], self.e2o_relation_data
            )
        # Unknown object type for relation.
        with self.assertRaises(ValueError):
            self.oced.add_event_to_object_relation(
                "rel2", "from_id", "to_id", ["Unknown"], self.e2o_relation_data
            )

    def test_add_object_to_object_relation(self):
        # Add both objects.
        self.oced.add_object(self.object_A_type, "object", self.object_A_data)
        self.oced.add_object(self.object_B_type, "object", self.object_B_data)
        # Add a valid object-to-object relation.
        self.oced.add_object_to_object_relation(
            "rel_o2o",
            "from_id",
            [self.object_A_type],
            "to_id",
            [self.object_B_type],
            self.o2o_relation_data,
        )
        rel = self.oced.get_relation("rel_o2o")
        self.assertEqual(rel["from_id"], "from_id")
        self.assertEqual(rel["from_type"], [self.object_A_type])
        self.assertEqual(rel["to_id"], "to_id")
        self.assertEqual(rel["to_type"], [self.object_B_type])
        assert_frame_equal(rel["data"], self.o2o_relation_data)
        # Duplicate addition should error.
        with self.assertRaises(ValueError):
            self.oced.add_object_to_object_relation(
                "rel_o2o",
                "from_id",
                [self.object_A_type],
                "to_id",
                [self.object_B_type],
                self.o2o_relation_data,
            )
        # Unknown object type in relation.
        with self.assertRaises(ValueError):
            self.oced.add_object_to_object_relation(
                "rel_invalid",
                "from_id",
                ["X"],
                "to_id",
                [self.object_B_type],
                self.o2o_relation_data,
            )

    def test_get_log(self):
        # Setup events, object, and relation for log generation.
        self.oced.add_event("type1", "event", "timestamp", self.event_1)
        self.oced.add_event("type2", "event", "timestamp", self.event_2)
        self.oced.add_object(self.object_A_type, "object", self.object_A_data)
        self.oced.add_event_to_object_relation(
            "rel_log", "from_id", "to_id", self.object_A_type, self.e2o_relation_data
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
        self.assertEqual(self.oced.get_types(["1"]), [self.object_A_type])

        # Test get_type with an id present in the event dataset.
        self.assertEqual(self.oced.get_types(["e2"]), ["event_type"])

        # Test get_type with an id not present in any dataset.
        self.assertEqual(self.oced.get_types(["nonexistent"]), [])
        
    def test_get_most_common_objects(self):
        # Add two object types with different sizes
        self.oced.add_object(self.object_A_type, "object", self.object_A_data)  # len=3
        self.oced.add_object(self.object_B_type, "object", self.object_B_data)  # len=2

        # Top-1 should be A
        top1 = self.oced.get_most_common_objects(n=1)
        self.assertEqual(top1, [self.object_A_type])

        # Top-2 should preserve order by size: A then B
        top2 = self.oced.get_most_common_objects(n=2)
        self.assertEqual(top2, [self.object_A_type, self.object_B_type])
        
    def test_change_object_ids_propagates_everywhere(self):
        # Objects
        self.oced.add_object("A", "object", pd.DataFrame({"object": [1, 2, 3]}))
        self.oced.add_object("B", "object", pd.DataFrame({"object": [10, 11]}))

        # Events
        ev = pd.DataFrame(
            {"event": ["e1", "e2", "e3"], "timestamp": pd.date_range("2022-01-01", periods=3, freq="D")}
        )
        self.oced.add_event("E", "event", "timestamp", ev)

        # Relations:
        # e2o: events linked to A objects
        e2o = pd.DataFrame({"from_id": ["e1", "e2", "e3"], "to_id": [1, 2, 3]})
        self.oced.add_event_to_object_relation("rel_e2o", "from_id", "to_id", ["A"], e2o)

        # o2o: A -> A and A -> B mixes
        o2o = pd.DataFrame(
            {
                "from_id": [1, 2, 3, 1],
                "to_id":   [2, 3, 1, 10],  # last one points to a B object
            }
        )
        self.oced.add_object_to_object_relation(
            "rel_o2o",
            "from_id", ["A"],
            "to_id",   ["A", "B"],
            o2o,
        )

        # Change function: prepend "A-" to IDs (stringify first to be safe)
        def change_fn(series):
            return series.astype(str).map(lambda x: f"A-{x}")

        # Execute
        self.oced.change_object_ids("A", change_fn)

        # --- Assertions ---

        # 1) Object table updated
        a_table = self.oced.get_object("A")["data"]
        self.assertCountEqual(a_table["object"].tolist(), ["A-1", "A-2", "A-3"])

        # 2) e2o updated wherever to_type includes "A"
        rel_e2o = self.oced.get_relation("rel_e2o")
        self.assertTrue(set(rel_e2o["data"]["to_id"].tolist()) == {"A-1", "A-2", "A-3"})

        # 3) o2o updated in both from_id and to_id where types include "A"
        rel_o2o = self.oced.get_relation("rel_o2o")
        # from_id (all A) should be changed:
        self.assertCountEqual(rel_o2o["data"]["from_id"].tolist(), ["A-1", "A-2", "A-3", "A-1"])
        # to_id includes A and B; only A ids should change, B ids (10) stay as is
        self.assertCountEqual(rel_o2o["data"]["to_id"].tolist(), ["A-2", "A-3", "A-1", 10])
        
    def test_prevent_cross_domain_id_reuse(self):
        # Prepare overlapping IDs as strings to ensure exact matching
        obj_df = pd.DataFrame({"object": ["e1", "e2", "o3"]})
        ev_df_1 = pd.DataFrame({
            "event": ["e2", "e4"],  # "e2" overlaps with object IDs
            "timestamp": pd.date_range("2022-01-01", periods=2, freq="D"),
        })
        ev_df_2 = pd.DataFrame({
            "event": ["e5", "e6"],
            "timestamp": pd.date_range("2022-01-03", periods=2, freq="D"),
        })
        obj_df_overlap = pd.DataFrame({"object": ["e6", "o7"]})  # "e6" overlaps with ev_df_2

        # 1) Add object first, then try to add event that reuses an ID -> should fail
        self.oced.add_object("A", "object", obj_df)
        with self.assertRaises(ValueError):
            self.oced.add_event("E_bad", "event", "timestamp", ev_df_1)
            
    def test_relation_name_clash_between_e2o_and_o2o(self):
        # Setup minimal objects needed
        self.oced.add_object(self.object_A_type, "object", self.object_A_data)
        self.oced.add_object(self.object_B_type, "object", self.object_B_data)

        # Add an e2o relation named "relX"
        e2o = pd.DataFrame({"from_id": ["e1", "e2"], "to_id": [1, 2]})
        self.oced.add_event("E", "event", "timestamp",
                            pd.DataFrame({"event": ["e1", "e2"], "timestamp": pd.date_range("2022-01-01", periods=2)}))
        self.oced.add_event_to_object_relation("relX", "from_id", "to_id", [self.object_A_type], e2o)

        # Attempt to add an o2o relation with the SAME name "relX" -> must fail
        o2o = pd.DataFrame({"from_id": [1], "to_id": [4]})
        with self.assertRaises(ValueError):
            self.oced.add_object_to_object_relation(
                "relX",
                "from_id", [self.object_A_type],
                "to_id",   [self.object_B_type],
                o2o,
            )

        # Now the other direction: create an o2o named "relY" first
        self.oced.add_object_to_object_relation(
            "relY",
            "from_id", [self.object_A_type],
            "to_id",   [self.object_B_type],
            pd.DataFrame({"from_id": [2], "to_id": [5]}),
        )

        # Trying to add an e2o with name "relY" should fail
        with self.assertRaises(ValueError):
            self.oced.add_event_to_object_relation(
                "relY", "from_id", "to_id", [self.object_A_type],
                pd.DataFrame({"from_id": ["e2"], "to_id": [2]}),
            )
            
    def test_change_event_ids_propagates_to_e2o_only_for_target_activity(self):
        # Objects
        self.oced.add_object("A", "object", pd.DataFrame({"object": [1, 2]}))

        # Two activities (event types) with disjoint IDs
        pay_df = pd.DataFrame({
            "event": ["e1", "e2"],
            "timestamp": pd.date_range("2022-01-01", periods=2, freq="D"),
        })
        ship_df = pd.DataFrame({
            "event": ["s1", "s2"],
            "timestamp": pd.date_range("2022-01-03", periods=2, freq="D"),
        })
        self.oced.add_event("Pay", "event", "timestamp", pay_df)
        self.oced.add_event("Ship", "event", "timestamp", ship_df)

        # e2o: link both Pay and Ship events to objects of type A
        e2o = pd.DataFrame({"from_id": ["e1", "e2", "s1"], "to_id": [1, 2, 1]})
        self.oced.add_event_to_object_relation("rel_e2o", "from_id", "to_id", ["A"], e2o)

        # Change function: add prefix to the targeted activity's IDs
        def change_fn(series):
            return series.astype(str).map(lambda x: f"P-{x}")

        # Execute for activity "Pay" only
        self.oced.change_event_ids("Pay", change_fn)

        # 1) Event table for Pay updated; Ship unchanged
        pay_after = self.oced.get_event("Pay")["data"]
        ship_after = self.oced.get_event("Ship")["data"]
        self.assertCountEqual(pay_after["event"].tolist(), ["P-e1", "P-e2"])
        self.assertCountEqual(ship_after["event"].tolist(), ["s1", "s2"])  # untouched

        # 2) e2o updated only for the Pay events (from_id)
        rel = self.oced.get_relation("rel_e2o")["data"]
        # Expect P-e1, P-e2 mapped to 1,2; s1->1 stays unchanged
        self.assertCountEqual(rel["from_id"].tolist(), ["P-e1", "P-e2", "s1"])
        self.assertCountEqual(rel["to_id"].tolist(), [1, 2, 1])

        # 3) get_log still works and yields three rows, with two "Pay" and one "Ship"
        log = self.oced.get_log("A")
        self.assertEqual(len(log), 3)
        # Concept names reflect event types: two Pay, one Ship
        self.assertEqual((log["concept:name"] == "Pay").sum(), 2)
        self.assertEqual((log["concept:name"] == "Ship").sum(), 1)
        
    def test_used_ids_updates_on_change_ids(self):
        # Setup: A has ids 1,2; we will rename them to A-1,A-2
        self.oced.add_object("A", "object", pd.DataFrame({"object": [1, 2]}))
        # Minimal event/activity + e2o so we exercise typical flows (not strictly needed here)
        ev = pd.DataFrame({"event": ["e1"], "timestamp": pd.date_range("2022-01-01", periods=1)})
        self.oced.add_event("E", "event", "timestamp", ev)
        self.oced.add_event_to_object_relation(
            "rel_e2o", "from_id", "to_id", ["A"], pd.DataFrame({"from_id": ["e1"], "to_id": [1]})
        )

        # Change function
        def change_fn(series):
            return series.astype(str).map(lambda x: f"A-{x}")

        # Execute rename
        self.oced.change_object_ids("A", change_fn)

        # Sanity: A now has A-1,A-2
        a_ids = set(self.oced.get_object("A")["data"]["object"].tolist())
        self.assertEqual(a_ids, {"A-1", "A-2"})

        # Now try to add a NEW object type C that (re)uses the OLD id "1".
        # If _used_ids wasn't updated, this should wrongly raise; we expect it to SUCCEED.
        self.oced.add_object("C", "cid", pd.DataFrame({"cid": [1]}))  # should be allowed

        # And adding an event that reuses old id "2" as an event id should also be allowed
        ev2 = pd.DataFrame({"event": ["2"], "timestamp": pd.date_range("2022-01-03", periods=1)})
        self.oced.add_event("E2", "event", "timestamp", ev2)  # should be allowed

        # Final quick check: ids present as expected
        self.assertIn("C", self.oced.get_object_types())
        self.assertIn("E2", self.oced.get_activities())
        
class TestOCEDStoreDataCheck(unittest.TestCase):
    def setUp(self):
        self.oced = OCEDStore()

    def test_passthrough_numeric_and_bool(self):
        df = pd.DataFrame({
            "ints": [1, 2, 3],
            "floats": [1.1, 2.2, 3.3],
            "bools": [True, False, True],
        })
        before = df.copy(deep=True)
        out = self.oced.data_check(df)

        # values unchanged (and same object shape)
        assert_series_equal(out["ints"], before["ints"], check_dtype=True)
        assert_series_equal(out["floats"], before["floats"], check_dtype=True)
        assert_series_equal(out["bools"], before["bools"], check_dtype=True)

    def test_normalize_datetime_dtype_tz_and_nanoseconds(self):
        # tz-aware + nanoseconds -> naive, floored to seconds
        ts = pd.to_datetime(
            ["2022-01-01 12:00:00.987654321+00:00",
             "2022-01-01 12:00:01.123456789+00:00"],
            utc=True
        )
        df = pd.DataFrame({"dt": ts})
        out = self.oced.data_check(df, min_parse_ratio=1.0)

        self.assertTrue(pd.api.types.is_datetime64_ns_dtype(out["dt"]))
        # floored to seconds
        expected = pd.to_datetime(["2022-01-01 12:00:00", "2022-01-01 12:00:01"], utc=True)
        assert_series_equal(out["dt"], pd.Series(expected, name="dt"))

    def test_parse_datetime_strings_when_ratio_met(self):
        df = pd.DataFrame({
            "maybe_dt": ["2022-01-01 08:00:00",
                         "2022-01-01 09:00:05",
                         None,
                         "garbage"]
        })
        # 2 parseable of 3 non-nan -> ratio = 2/3 ≈ 0.666
        out = self.oced.data_check(df, min_parse_ratio=0.66)

        # Row with None was dropped at the *start* (all-NA row in a 1-col DF)
        # "garbage" stays, becomes NaT after conversion.
        self.assertTrue(pd.api.types.is_datetime64_ns_dtype(out["maybe_dt"]))
        expected = pd.Series(pd.to_datetime(
            ["2022-01-01 08:00:00", "2022-01-01 09:00:05", pd.NaT], utc=True
        ), name="maybe_dt")
        assert_series_equal(out["maybe_dt"].reset_index(drop=True), expected)

    def test_parse_numeric_strings_when_ratio_met(self):
        df = pd.DataFrame({"maybe_num": ["10.1", "11.7", None, "x"]})
        out = self.oced.data_check(df, min_parse_ratio=0.66)

        self.assertTrue(pd.api.types.is_numeric_dtype(out["maybe_num"]))
        expected = pd.Series([10.1, 11.7, np.nan], name="maybe_num")

        # reset index to compare values only
        assert_series_equal(out["maybe_num"].reset_index(drop=True), expected)

    def test_do_not_convert_when_ratio_too_low(self):
        # Only 1 of 3 non-nan parseable => 0.333 < 0.5 threshold
        df = pd.DataFrame({
            "weak_dt": ["2022-01-01 00:00:00", "bad", "worse", None]
        })
        before = df.dropna(how="all")  # mimic initial drop behavior for fair compare
        out = self.oced.data_check(df, min_parse_ratio=0.5)

        # Should remain object dtype (no datetime conversion) and same values (except the dropped None row)
        self.assertTrue(pd.api.types.is_object_dtype(out["weak_dt"]))
        assert_series_equal(out["weak_dt"].reset_index(drop=True),
                            before["weak_dt"].reset_index(drop=True))

    def test_ignore_cols_keeps_column_untouched(self):
        df = pd.DataFrame({
            "ignored": ["2022-01-01 12:00:00", "2022-01-01 12:00:01"],
            "convert_me": ["2022-01-01 12:00:00.123", "2022-01-01 12:00:01.999"],
        })
        before_ignored = df["ignored"].copy(deep=True)
        out = self.oced.data_check(df, ignore_cols=["ignored"])

        # ignored stays object strings
        self.assertTrue(pd.api.types.is_object_dtype(out["ignored"]))
        assert_series_equal(out["ignored"], before_ignored)

        # other column becomes naive datetime floored to seconds
        self.assertTrue(pd.api.types.is_datetime64_ns_dtype(out["convert_me"]))
        expected = pd.Series(pd.to_datetime(
            ["2022-01-01 12:00:00", "2022-01-01 12:00:01"], utc=True
        ), name="convert_me")
        assert_series_equal(out["convert_me"], expected)

    def test_drop_all_na_rows_and_columns(self):
        df = pd.DataFrame({
            "all_na_col": [np.nan, np.nan, np.nan],
            "mix": [1, np.nan, 3],
        })
        # Also include a fully NA row (after dropping all_na_col)
        df.loc[1, "mix"] = np.nan  # row index 1 becomes all-NA across columns

        out = self.oced.data_check(df)

        # all_na_col should be gone
        self.assertNotIn("all_na_col", out.columns)
        # fully NA row (index 1) should be dropped
        self.assertEqual(len(out), 2)
        self.assertListEqual(out["mix"].tolist(), [1, 3])

if __name__ == "__main__":
    unittest.main()
