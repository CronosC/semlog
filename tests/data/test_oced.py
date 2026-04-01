import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from semanticlog.data.oced import OCEDStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_oced_with_e2o():
    """Minimal OCEDStore: one event type, one object type, one e2o relation."""
    oced = OCEDStore()
    oced.add_object("A", "oid", pd.DataFrame({"oid": [1, 2]}))
    ev = pd.DataFrame(
        {
            "eid": ["e1", "e2"],
            "ts": pd.date_range("2022-01-01", periods=2, tz="UTC"),
        }
    )
    oced.add_event("E", "eid", "ts", ev)
    oced.add_event_to_object_relation(
        "rel", "eid", "oid", ["A"], pd.DataFrame({"eid": ["e1", "e2"], "oid": [1, 2]})
    )
    return oced


# ---------------------------------------------------------------------------
# add_event / get_event / get_activities
# ---------------------------------------------------------------------------


def test_add_event_stores_metadata_and_data(event_df_1):
    oced = OCEDStore()
    oced.add_event("type1", "event", "timestamp", event_df_1)
    ev = oced.get_events("type1")
    assert ev["id"] == "event"
    assert ev["time"] == "timestamp"
    assert_frame_equal(ev["data"], event_df_1)


def test_add_event_raises_for_missing_required_columns(event_df_1):
    oced = OCEDStore()
    bad = event_df_1.drop("timestamp", axis=1)
    with pytest.raises(ValueError):
        oced.add_event("type1", "event", "timestamp", bad)


def test_add_event_raises_on_duplicate_without_overwrite(event_df_2):
    oced = OCEDStore()
    oced.add_event("type2", "event", "timestamp", event_df_2)
    with pytest.raises(ValueError):
        oced.add_event("type2", "event", "timestamp", event_df_2)


def test_add_event_overwrites_existing_when_flag_set(event_df_2):
    oced = OCEDStore()
    oced.add_event("type2", "event", "timestamp", event_df_2)
    oced.add_event("type2", "event", "timestamp", event_df_2, overwrite=True)
    assert_frame_equal(oced.get_events("type2")["data"], event_df_2)


# ---------------------------------------------------------------------------
# add_object / get_object / get_object_types
# ---------------------------------------------------------------------------


def test_add_object_stores_metadata_and_data(object_df_a):
    oced = OCEDStore()
    oced.add_object("A", "object", object_df_a)
    obj = oced.get_object("A")
    assert obj["id"] == "object"
    assert_frame_equal(obj["data"], object_df_a)


def test_add_object_raises_for_missing_id_column(object_df_a):
    oced = OCEDStore()
    with pytest.raises(ValueError):
        oced.add_object("A", "nonexistent_col", object_df_a)


def test_add_object_raises_on_duplicate_without_overwrite(object_df_a):
    oced = OCEDStore()
    oced.add_object("A", "object", object_df_a)
    with pytest.raises(ValueError):
        oced.add_object("A", "object", object_df_a)


def test_add_object_overwrite_replaces_data():
    oced = OCEDStore()
    oced.add_object("A", "oid", pd.DataFrame({"oid": [1, 2]}))
    replacement = pd.DataFrame({"oid": [10, 20]})
    oced.add_object("A", "oid", replacement, overwrite=True)
    result_ids = list(oced.get_object("A")["data"]["oid"])
    assert result_ids == [10, 20]


def test_get_object_types_returns_all_added_types(object_df_a, object_df_b):
    oced = OCEDStore()
    oced.add_object("A", "object", object_df_a)
    oced.add_object("B", "object", object_df_b)
    assert set(oced.get_object_types()) == {"A", "B"}


def test_get_most_common_objects_returns_sorted_by_size(object_df_a, object_df_b):
    oced = OCEDStore()
    oced.add_object("A", "object", object_df_a)  # 3 rows
    oced.add_object("B", "object", object_df_b)  # 2 rows
    assert oced.get_most_common_objects(n=1) == ["A"]
    assert oced.get_most_common_objects(n=2) == ["A", "B"]


# ---------------------------------------------------------------------------
# rename_activity / rename_object / rename_relation
# ---------------------------------------------------------------------------


def test_rename_activity_updates_key(minimal_oced):
    minimal_oced.rename_activity("type1", "type1_new")
    assert "type1_new" in minimal_oced.get_activities()
    assert "type1" not in minimal_oced.get_activities()


def test_rename_activity_raises_for_nonexistent(minimal_oced):
    with pytest.raises(ValueError):
        minimal_oced.rename_activity("no_such", "x")


def test_rename_object_updates_key(minimal_oced):
    minimal_oced.rename_object("A", "C")
    assert "C" in minimal_oced.get_object_types()
    assert "A" not in minimal_oced.get_object_types()


def test_rename_object_propagates_type_in_relations(minimal_oced):
    minimal_oced.rename_object("A", "C")
    for rel in minimal_oced.get_relations("e2o").values():
        assert "A" not in rel["to_type"]
        assert "C" in rel["to_type"]
    for rel in minimal_oced.get_relations("o2o").values():
        assert "A" not in rel["from_type"]
        assert "A" not in rel["to_type"]


def test_rename_object_raises_for_nonexistent(minimal_oced):
    with pytest.raises(ValueError):
        minimal_oced.rename_object("no_such", "x")


def test_rename_relation_updates_key(minimal_oced):
    minimal_oced.rename_relation("rel1", "rel1_new")
    assert "rel1_new" in minimal_oced.get_relations("e2o")
    assert "rel1" not in minimal_oced.get_relations("e2o")


def test_rename_relation_raises_for_nonexistent(minimal_oced):
    with pytest.raises(ValueError):
        minimal_oced.rename_relation("no_such", "x")


def test_rename_relation_raises_when_target_name_exists(minimal_oced):
    # "rel2" already exists; renaming "rel1" to "rel2" should fail
    with pytest.raises(ValueError):
        minimal_oced.rename_relation("rel1", "rel2")


# ---------------------------------------------------------------------------
# add_event_to_object_relation / add_object_to_object_relation
# ---------------------------------------------------------------------------


def test_add_e2o_relation_stores_metadata_and_data(object_df_a, e2o_data):
    oced = OCEDStore()
    oced.add_object("A", "object", object_df_a)
    oced.add_event_to_object_relation("rel1", "from_id", "to_id", ["A"], e2o_data)
    rel = oced.get_relation("rel1")
    assert rel["from_id"] == "from_id"
    assert rel["to_id"] == "to_id"
    assert rel["to_type"] == ["A"]
    assert_frame_equal(rel["data"], e2o_data)


def test_add_e2o_relation_raises_on_duplicate_without_overwrite(object_df_a, e2o_data):
    oced = OCEDStore()
    oced.add_object("A", "object", object_df_a)
    oced.add_event_to_object_relation("rel1", "from_id", "to_id", ["A"], e2o_data)
    with pytest.raises(ValueError):
        oced.add_event_to_object_relation("rel1", "from_id", "to_id", ["A"], e2o_data)


def test_add_e2o_relation_raises_for_unknown_object_type(e2o_data):
    oced = OCEDStore()
    with pytest.raises(ValueError):
        oced.add_event_to_object_relation(
            "rel1", "from_id", "to_id", ["Unknown"], e2o_data
        )


def test_add_e2o_relation_raises_when_from_id_col_missing(object_df_a):
    oced = OCEDStore()
    oced.add_object("A", "object", object_df_a)
    bad_data = pd.DataFrame({"to_id": [1, 2]})
    with pytest.raises(ValueError, match="from_id_col"):
        oced.add_event_to_object_relation("rel1", "from_id", "to_id", ["A"], bad_data)


def test_add_e2o_relation_raises_when_to_id_col_missing(object_df_a):
    oced = OCEDStore()
    oced.add_object("A", "object", object_df_a)
    bad_data = pd.DataFrame({"from_id": ["e1", "e2"]})
    with pytest.raises(ValueError, match="to_id_col"):
        oced.add_event_to_object_relation("rel1", "from_id", "to_id", ["A"], bad_data)


def test_add_e2o_relation_raises_if_name_exists_as_o2o(minimal_oced):
    # "rel2" is already an o2o relation; adding an e2o with the same name must fail
    new_data = pd.DataFrame({"from_id": ["e1"], "to_id": [1]})
    with pytest.raises(ValueError):
        minimal_oced.add_event_to_object_relation(
            "rel2", "from_id", "to_id", ["A"], new_data
        )


def test_add_o2o_relation_stores_metadata_and_data(object_df_a, object_df_b, o2o_data):
    oced = OCEDStore()
    oced.add_object("A", "object", object_df_a)
    oced.add_object("B", "object", object_df_b)
    oced.add_object_to_object_relation(
        "rel2", "from_id", ["A"], "to_id", ["B"], o2o_data
    )
    rel = oced.get_relation("rel2")
    assert rel["from_id"] == "from_id"
    assert rel["from_type"] == ["A"]
    assert rel["to_id"] == "to_id"
    assert rel["to_type"] == ["B"]
    assert_frame_equal(rel["data"], o2o_data)


def test_add_o2o_relation_raises_on_duplicate_without_overwrite(
    object_df_a, object_df_b, o2o_data
):
    oced = OCEDStore()
    oced.add_object("A", "object", object_df_a)
    oced.add_object("B", "object", object_df_b)
    oced.add_object_to_object_relation(
        "rel2", "from_id", ["A"], "to_id", ["B"], o2o_data
    )
    with pytest.raises(ValueError):
        oced.add_object_to_object_relation(
            "rel2", "from_id", ["A"], "to_id", ["B"], o2o_data
        )


def test_add_o2o_relation_raises_for_unknown_object_type(object_df_a, o2o_data):
    oced = OCEDStore()
    oced.add_object("A", "object", object_df_a)
    with pytest.raises(ValueError):
        oced.add_object_to_object_relation(
            "rel_bad", "from_id", ["X"], "to_id", ["A"], o2o_data
        )


def test_add_o2o_relation_raises_if_name_exists_as_e2o(minimal_oced):
    # "rel1" is already an e2o relation; adding an o2o with same name must fail
    with pytest.raises(ValueError):
        minimal_oced.add_object_to_object_relation(
            "rel1",
            "from_id",
            ["A"],
            "to_id",
            ["B"],
            pd.DataFrame({"from_id": [1], "to_id": [4]}),
        )


# ---------------------------------------------------------------------------
# get_relations / get_relation_types
# ---------------------------------------------------------------------------


def test_get_relations_filter_e2o_only(minimal_oced):
    result = minimal_oced.get_relations("e2o")
    assert "rel1" in result
    assert "rel2" not in result


def test_get_relations_filter_o2o_only(minimal_oced):
    result = minimal_oced.get_relations("o2o")
    assert "rel2" in result
    assert "rel1" not in result


def test_get_relations_all_contains_both(minimal_oced):
    result = minimal_oced.get_relations("all")
    assert "event_to_object" in result
    assert "object_to_object" in result


def test_get_relations_invalid_type_raises(minimal_oced):
    with pytest.raises(ValueError):
        minimal_oced.get_relations("bad")


def test_get_relation_types_all_returns_both(minimal_oced):
    types = minimal_oced.get_relation_types("all")
    assert "rel1" in types
    assert "rel2" in types


def test_get_relation_types_e2o_returns_only_e2o(minimal_oced):
    types = minimal_oced.get_relation_types("e2o")
    assert "rel1" in types
    assert "rel2" not in types


def test_get_relation_types_o2o_returns_only_o2o(minimal_oced):
    types = minimal_oced.get_relation_types("o2o")
    assert "rel2" in types
    assert "rel1" not in types


def test_get_relation_types_invalid_raises(minimal_oced):
    with pytest.raises(ValueError):
        minimal_oced.get_relation_types("bad")


# ---------------------------------------------------------------------------
# get_types
# ---------------------------------------------------------------------------


def test_get_types_returns_object_type_for_known_object_id():
    oced = OCEDStore()
    oced.add_object("A", "object", pd.DataFrame({"object": ["o1", "o2"]}))
    assert oced.get_types(["o1"]) == ["A"]


def test_get_types_returns_activity_for_known_event_id(event_df_1):
    oced = OCEDStore()
    oced.add_event("event_type", "event", "timestamp", event_df_1)
    assert oced.get_types(["e2"]) == ["event_type"]


def test_get_types_returns_empty_for_unknown_id():
    oced = OCEDStore()
    assert oced.get_types(["nonexistent"]) == []


# ---------------------------------------------------------------------------
# get_ids
# ---------------------------------------------------------------------------


def test_get_ids_event_returns_correct_ids(event_df_1):
    oced = OCEDStore()
    oced.add_event("type1", "event", "timestamp", event_df_1)
    ids = set(oced.get_ids("event", "type1"))
    assert ids == {"e1", "e2", "e5"}


def test_get_ids_object_returns_correct_ids(object_df_a):
    oced = OCEDStore()
    oced.add_object("A", "object", object_df_a)
    ids = set(oced.get_ids("object", "A"))
    assert ids == {1, 2, 3}


def test_get_ids_invalid_of_raises():
    oced = OCEDStore()
    with pytest.raises(ValueError):
        oced.get_ids("bad", "whatever")


def test_get_ids_unknown_activity_raises():
    oced = OCEDStore()
    with pytest.raises(ValueError):
        oced.get_ids("event", "no_such_activity")


def test_get_ids_unknown_object_raises():
    oced = OCEDStore()
    with pytest.raises(ValueError):
        oced.get_ids("object", "no_such_type")


# ---------------------------------------------------------------------------
# ID uniqueness constraints
# ---------------------------------------------------------------------------


def test_add_event_raises_when_ids_already_used_by_object():
    oced = OCEDStore()
    oced.add_object("A", "object", pd.DataFrame({"object": ["e1", "e2", "o3"]}))
    ev_bad = pd.DataFrame(
        {
            "event": ["e2", "e4"],
            "timestamp": pd.date_range("2022-01-01", periods=2),
        }
    )
    with pytest.raises(ValueError):
        oced.add_event("E_bad", "event", "timestamp", ev_bad)


# ---------------------------------------------------------------------------
# change_ids / change_event_ids / change_object_ids
# ---------------------------------------------------------------------------


def test_change_ids_dispatcher_routes_to_event():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["e1", "e2"], "ts": pd.date_range("2022-01-01", periods=2)}
    )
    oced.add_event("act", "eid", "ts", ev)
    oced.change_ids("event", "act", lambda s: s.str.upper())
    assert set(oced.get_ids("event", "act")) == {"E1", "E2"}


def test_change_ids_dispatcher_routes_to_object():
    oced = OCEDStore()
    oced.add_object("A", "oid", pd.DataFrame({"oid": [1, 2]}))
    oced.change_ids("object", "A", lambda s: s.astype(str).map(lambda x: f"A-{x}"))
    assert set(oced.get_ids("object", "A")) == {"A-1", "A-2"}


def test_change_ids_invalid_of_raises():
    oced = OCEDStore()
    with pytest.raises(ValueError):
        oced.change_ids("bad", "A", lambda s: s)


def test_change_object_ids_propagates_to_e2o_and_o2o():
    oced = OCEDStore()
    oced.add_object("A", "object", pd.DataFrame({"object": [1, 2, 3]}))
    oced.add_object("B", "object", pd.DataFrame({"object": [10, 11]}))
    ev = pd.DataFrame(
        {
            "event": ["e1", "e2", "e3"],
            "timestamp": pd.date_range("2022-01-01", periods=3, freq="D"),
        }
    )
    oced.add_event("E", "event", "timestamp", ev)
    e2o = pd.DataFrame({"from_id": ["e1", "e2", "e3"], "to_id": [1, 2, 3]})
    oced.add_event_to_object_relation("rel_e2o", "from_id", "to_id", ["A"], e2o)
    o2o = pd.DataFrame({"from_id": [1, 2, 3, 1], "to_id": [2, 3, 1, 10]})
    oced.add_object_to_object_relation(
        "rel_o2o", "from_id", ["A"], "to_id", ["A", "B"], o2o
    )

    def change_fn(series):
        return series.astype(str).map(lambda x: f"A-{x}")

    oced.change_object_ids("A", change_fn)

    a_table = oced.get_object("A")["data"]
    assert set(a_table["object"].tolist()) == {"A-1", "A-2", "A-3"}

    rel_e2o = oced.get_relation("rel_e2o")
    assert set(rel_e2o["data"]["to_id"].tolist()) == {"A-1", "A-2", "A-3"}

    rel_o2o = oced.get_relation("rel_o2o")
    assert set(rel_o2o["data"]["from_id"].tolist()) == {"A-1", "A-2", "A-3", "A-1"}
    # B-side id (10) must be unchanged
    assert 10 in rel_o2o["data"]["to_id"].tolist()


def test_change_object_ids_rolls_back_on_error():
    oced = OCEDStore()
    oced.add_object("A", "oid", pd.DataFrame({"oid": [1, 2, 3]}))
    original_ids = set(oced.get_ids("object", "A"))

    def bad_fn(series):
        raise RuntimeError("deliberate failure")

    with pytest.raises(RuntimeError):
        oced.change_object_ids("A", bad_fn)

    assert set(oced.get_ids("object", "A")) == original_ids


def test_change_event_ids_propagates_to_e2o_for_target_activity_only():
    oced = OCEDStore()
    oced.add_object("A", "object", pd.DataFrame({"object": [1, 2]}))
    pay_df = pd.DataFrame(
        {
            "event": ["e1", "e2"],
            "timestamp": pd.date_range("2022-01-01", periods=2, freq="D"),
        }
    )
    ship_df = pd.DataFrame(
        {
            "event": ["s1", "s2"],
            "timestamp": pd.date_range("2022-01-03", periods=2, freq="D"),
        }
    )
    oced.add_event("Pay", "event", "timestamp", pay_df)
    oced.add_event("Ship", "event", "timestamp", ship_df)
    e2o = pd.DataFrame({"from_id": ["e1", "e2", "s1"], "to_id": [1, 2, 1]})
    oced.add_event_to_object_relation("rel_e2o", "from_id", "to_id", ["A"], e2o)

    oced.change_event_ids("Pay", lambda s: s.astype(str).map(lambda x: f"P-{x}"))

    pay_after = oced.get_event("Pay")["data"]
    ship_after = oced.get_event("Ship")["data"]
    assert set(pay_after["event"].tolist()) == {"P-e1", "P-e2"}
    assert set(ship_after["event"].tolist()) == {"s1", "s2"}

    rel = oced.get_relation("rel_e2o")["data"]
    assert set(rel["from_id"].tolist()) == {"P-e1", "P-e2", "s1"}


def test_used_ids_updated_after_change_object_ids():
    oced = OCEDStore()
    oced.add_object("A", "object", pd.DataFrame({"object": [1, 2]}))

    oced.change_object_ids("A", lambda s: s.astype(str).map(lambda x: f"A-{x}"))

    # Old IDs (1, 2) should no longer be reserved — adding them to new type is allowed
    oced.add_object("C", "cid", pd.DataFrame({"cid": [1]}))
    oced.add_event(
        "E2",
        "event",
        "ts",
        pd.DataFrame({"event": ["2"], "ts": pd.date_range("2022-01-03", periods=1)}),
    )
    assert "C" in oced.get_object_types()
    assert "E2" in oced.get_activities()


# ---------------------------------------------------------------------------
# get_log
# ---------------------------------------------------------------------------


def test_get_log_raises_for_unknown_case_concept(minimal_oced):
    with pytest.raises(ValueError, match="Unknown object type"):
        minimal_oced.get_log("no_such_type")


def test_get_log_raises_when_no_e2o_references_case_concept():
    oced = OCEDStore()
    oced.add_object("B", "bid", pd.DataFrame({"bid": ["b1"]}))
    oced.add_object("A", "oid", pd.DataFrame({"oid": [1]}))
    ev = pd.DataFrame({"eid": ["e1"], "ts": pd.to_datetime(["2022-01-01"], utc=True)})
    oced.add_event("E", "eid", "ts", ev)
    # e2o points to A only, not B
    oced.add_event_to_object_relation(
        "rel", "eid", "oid", ["A"], pd.DataFrame({"eid": ["e1"], "oid": [1]})
    )
    with pytest.raises(ValueError, match="No event-to-object relations"):
        oced.get_log("B")


def test_get_log_returns_dataframe_with_pm4py_columns(minimal_oced):
    log = minimal_oced.get_log("A")
    assert isinstance(log, pd.DataFrame)
    assert "case:concept:name" in log.columns
    assert "concept:name" in log.columns
    assert "time:timestamp" in log.columns


def test_get_log_row_count_and_activity_column(minimal_oced):
    # minimal_oced: 5 events (e1,e2,e5 of type1; e3,e4 of type2)
    # e2o maps all 5 events to A objects
    log = minimal_oced.get_log("A")
    assert len(log) == 5
    activity_counts = log["concept:name"].value_counts()
    assert activity_counts["type1"] == 3
    assert activity_counts["type2"] == 2


# ---------------------------------------------------------------------------
# data_check
# ---------------------------------------------------------------------------


def test_data_check_passes_through_numeric_and_bool():
    oced = OCEDStore()
    df = pd.DataFrame(
        {"ints": [1, 2, 3], "floats": [1.1, 2.2, 3.3], "bools": [True, False, True]}
    )
    before = df.copy(deep=True)
    out = oced.data_check(df)
    assert_series_equal(out["ints"], before["ints"], check_dtype=True)
    assert_series_equal(out["floats"], before["floats"], check_dtype=True)
    assert_series_equal(out["bools"], before["bools"], check_dtype=True)


def test_data_check_normalizes_datetime_to_seconds_floored():
    oced = OCEDStore()
    ts = pd.to_datetime(
        [
            "2022-01-01 12:00:00.987654321+00:00",
            "2022-01-01 12:00:01.123456789+00:00",
        ],
        utc=True,
    )
    out = oced.data_check(pd.DataFrame({"dt": ts}), min_parse_ratio=1.0)
    assert pd.api.types.is_datetime64_ns_dtype(out["dt"])
    expected = pd.to_datetime(["2022-01-01 12:00:00", "2022-01-01 12:00:01"], utc=True)
    assert_series_equal(out["dt"], pd.Series(expected, name="dt"))


def test_data_check_parses_datetime_strings_when_ratio_met():
    oced = OCEDStore()
    df = pd.DataFrame(
        {"maybe_dt": ["2022-01-01 08:00:00", "2022-01-01 09:00:05", None, "garbage"]}
    )
    out = oced.data_check(df, min_parse_ratio=0.66)
    assert pd.api.types.is_datetime64_ns_dtype(out["maybe_dt"])


def test_data_check_parses_numeric_strings_when_ratio_met():
    oced = OCEDStore()
    df = pd.DataFrame({"maybe_num": ["10.1", "11.7", None, "x"]})
    out = oced.data_check(df, min_parse_ratio=0.66)
    assert pd.api.types.is_numeric_dtype(out["maybe_num"])
    assert out["maybe_num"].dropna().tolist() == pytest.approx([10.1, 11.7])


def test_data_check_does_not_convert_when_ratio_too_low():
    oced = OCEDStore()
    df = pd.DataFrame({"weak_dt": ["2022-01-01 00:00:00", "bad", "worse", None]})
    out = oced.data_check(df, min_parse_ratio=0.5)
    assert pd.api.types.is_object_dtype(out["weak_dt"])


def test_data_check_respects_ignore_cols():
    oced = OCEDStore()
    df = pd.DataFrame(
        {
            "ignored": ["2022-01-01 12:00:00", "2022-01-01 12:00:01"],
            "convert_me": ["2022-01-01 12:00:00.123", "2022-01-01 12:00:01.999"],
        }
    )
    out = oced.data_check(df, ignore_cols=["ignored"])
    assert pd.api.types.is_object_dtype(out["ignored"])
    assert pd.api.types.is_datetime64_ns_dtype(out["convert_me"])


def test_data_check_drops_all_na_rows_and_columns():
    oced = OCEDStore()
    import numpy as np

    df = pd.DataFrame({"all_na_col": [np.nan, np.nan, np.nan], "mix": [1, np.nan, 3]})
    df.loc[1, "mix"] = np.nan
    out = oced.data_check(df)
    assert "all_na_col" not in out.columns
    assert len(out) == 2
    assert out["mix"].tolist() == [1, 3]


def test_data_check_coerces_zero_one_strings_to_boolean():
    oced = OCEDStore()
    df = pd.DataFrame({"flag": ["0", "1", "0", "1"]})
    out = oced.data_check(df)
    assert str(out["flag"].dtype) == "boolean"


def test_data_check_coerces_whole_number_strings_to_int64():
    oced = OCEDStore()
    df = pd.DataFrame({"count": ["1", "2", "3", "10"]})
    out = oced.data_check(df)
    assert str(out["count"].dtype) == "Int64"


# ---------------------------------------------------------------------------
# ocel_mode
# ---------------------------------------------------------------------------


def _make_object_changes_df():
    return pd.DataFrame(
        {
            "ocel:oid": ["o1", "o1"],
            "ocel:type": ["order", "order"],
            "ocel:timestamp": pd.to_datetime(["2022-01-01", "2022-01-02"], utc=True),
            "ocel:field": ["status", "status"],
            "status": ["pending", "shipped"],
        }
    )


def test_ocel_mode_default_is_false():
    oced = OCEDStore()
    assert oced.has_ocel_mode() is False


def test_ocel_mode_true_when_set():
    oced = OCEDStore(ocel_mode=True)
    assert oced.has_ocel_mode() is True


def test_add_object_changes_raises_when_ocel_mode_false():
    oced = OCEDStore()
    with pytest.raises(ValueError, match="ocel_mode"):
        oced.add_object_changes("order", _make_object_changes_df())


def test_get_object_changes_raises_when_ocel_mode_false():
    oced = OCEDStore()
    with pytest.raises(ValueError, match="ocel_mode"):
        oced.get_object_changes("order")


def test_get_object_change_types_raises_when_ocel_mode_false():
    oced = OCEDStore()
    with pytest.raises(ValueError, match="ocel_mode"):
        oced.get_object_change_types()


def test_add_object_changes_stores_data():
    oced = OCEDStore(ocel_mode=True)
    df = _make_object_changes_df()
    oced.add_object_changes("order", df)
    result = oced.get_object_changes("order")
    assert len(result) == 2


def test_add_object_changes_raises_for_missing_required_columns():
    oced = OCEDStore(ocel_mode=True)
    bad_df = pd.DataFrame({"ocel:oid": ["o1"], "ocel:type": ["order"]})
    with pytest.raises(ValueError):
        oced.add_object_changes("order", bad_df)


def test_get_object_change_types_returns_added_types():
    oced = OCEDStore(ocel_mode=True)
    oced.add_object_changes("order", _make_object_changes_df())
    oced.add_object_changes("item", _make_object_changes_df())
    assert set(oced.get_object_change_types()) == {"order", "item"}


def test_get_object_changes_raises_for_unknown_type():
    oced = OCEDStore(ocel_mode=True)
    with pytest.raises(ValueError):
        oced.get_object_changes("nonexistent")


def test_get_object_change_types_empty_when_nothing_added():
    oced = OCEDStore(ocel_mode=True)
    assert oced.get_object_change_types() == []
