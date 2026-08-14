"""The store: adding, reading, renaming and flattening."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from semanticlog.data.flatten import get_log, iter_logs
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind, RelationKind


def _make_oced_with_e2o():
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


def test_add_event_stores_metadata_and_data(event_df_1):
    oced = OCEDStore()
    oced.add_event("type1", "event", "timestamp", event_df_1)
    ev = oced.get_event_table("type1")
    assert ev.id == "event"
    assert ev.time == "timestamp"
    assert_frame_equal(ev.data, event_df_1)


def test_add_event_raises_for_missing_required_columns(event_df_1):
    oced = OCEDStore()
    bad = event_df_1.drop("timestamp", axis=1)
    with pytest.raises(ValueError):
        oced.add_event("type1", "event", "timestamp", bad)


def test_add_event_raises_on_a_second_table_for_one_activity(event_df_2):
    oced = OCEDStore()
    oced.add_event("type2", "event", "timestamp", event_df_2)
    with pytest.raises(ValueError):
        oced.add_event("type2", "event", "timestamp", event_df_2)


def test_add_object_stores_metadata_and_data(object_df_a):
    oced = OCEDStore()
    oced.add_object("A", "object", object_df_a)
    obj = oced.get_object_table("A")
    assert obj.id == "object"
    assert_frame_equal(obj.data, object_df_a)


def test_add_object_raises_for_missing_id_column(object_df_a):
    oced = OCEDStore()
    with pytest.raises(ValueError):
        oced.add_object("A", "nonexistent_col", object_df_a)


def test_add_object_raises_on_a_second_table_for_one_type(object_df_a):
    oced = OCEDStore()
    oced.add_object("A", "object", object_df_a)
    with pytest.raises(ValueError):
        oced.add_object("A", "object", object_df_a)


def test_get_object_types_returns_all_added_types(object_df_a, object_df_b):
    oced = OCEDStore()
    oced.add_object("A", "object", object_df_a)
    oced.add_object("B", "object", object_df_b)
    assert set(oced.get_object_types()) == {"A", "B"}


def test_get_most_common_objects_returns_sorted_by_size(object_df_a, object_df_b):
    oced = OCEDStore()
    oced.add_object("A", "object", object_df_a)
    oced.add_object("B", "object", object_df_b)
    assert oced.get_most_common_objects(n=1) == ["A"]
    assert oced.get_most_common_objects(n=2) == ["A", "B"]


def test_get_most_common_objects_refuses_a_negative_count(object_df_a, object_df_b):
    """A negative slice would read as "all but the smallest" instead."""
    oced = OCEDStore()
    oced.add_object("A", "object", object_df_a)
    oced.add_object("B", "object", object_df_b)

    with pytest.raises(ValueError, match="needs n >= 0"):
        oced.get_most_common_objects(n=-1)


def test_rename_activity_updates_key(minimal_oced):
    minimal_oced.rename_activity("type1", "type1_new")
    assert "type1_new" in minimal_oced.get_event_types()
    assert "type1" not in minimal_oced.get_event_types()


def test_rename_activity_raises_for_nonexistent(minimal_oced):
    with pytest.raises(ValueError):
        minimal_oced.rename_activity("no_such", "x")


def test_rename_object_updates_key(minimal_oced):
    minimal_oced.rename_object("A", "C")
    assert "C" in minimal_oced.get_object_types()
    assert "A" not in minimal_oced.get_object_types()


def test_rename_object_propagates_type_in_relations(minimal_oced):
    minimal_oced.rename_object("A", "C")
    for rel in minimal_oced.get_e2o().values():
        assert "A" not in rel.to_type
        assert "C" in rel.to_type
    for rel in minimal_oced.get_o2o().values():
        assert "A" not in rel.from_type
        assert "A" not in rel.to_type


def test_rename_object_raises_for_nonexistent(minimal_oced):
    with pytest.raises(ValueError):
        minimal_oced.rename_object("no_such", "x")


def test_rename_object_syncs_object_changes_key():
    oced = OCEDStore(ocel_mode=True)
    oced.add_object("order", "oid", pd.DataFrame({"oid": ["o1"]}))
    ev = pd.DataFrame({"eid": ["e1"], "ts": pd.to_datetime(["2022-01-01"], utc=True)})
    oced.add_event("act", "eid", "ts", ev)
    oced.add_event_to_object_relation(
        "rel", "eid", "oid", ["order"], pd.DataFrame({"eid": ["e1"], "oid": ["o1"]})
    )
    changes = pd.DataFrame(
        {
            "ocel:oid": ["o1"],
            "ocel:type": ["order"],
            "ocel:timestamp": pd.to_datetime(["2022-01-02"], utc=True),
            "ocel:field": ["status"],
            "status": ["pending"],
        }
    )
    oced.add_object_changes("order", changes)

    oced.rename_object("order", "purchase_order")

    assert "purchase_order" in oced.get_object_change_types()
    assert "order" not in oced.get_object_change_types()


def test_rename_object_leaves_object_changes_intact_when_no_entry():
    oced = OCEDStore(ocel_mode=True)
    oced.add_object("order", "oid", pd.DataFrame({"oid": ["o1"]}))
    ev = pd.DataFrame({"eid": ["e1"], "ts": pd.to_datetime(["2022-01-01"], utc=True)})
    oced.add_event("act", "eid", "ts", ev)
    oced.add_event_to_object_relation(
        "rel", "eid", "oid", ["order"], pd.DataFrame({"eid": ["e1"], "oid": ["o1"]})
    )
    oced.rename_object("order", "purchase_order")
    assert oced.get_object_change_types() == []


def test_rename_relation_updates_key(minimal_oced):
    minimal_oced.rename_relation("rel1", "rel1_new")
    assert "rel1_new" in minimal_oced.get_e2o()
    assert "rel1" not in minimal_oced.get_e2o()


def test_rename_relation_raises_for_nonexistent(minimal_oced):
    with pytest.raises(ValueError):
        minimal_oced.rename_relation("no_such", "x")


def test_rename_relation_raises_when_target_name_exists(minimal_oced):
    with pytest.raises(ValueError):
        minimal_oced.rename_relation("rel1", "rel2")


def test_add_e2o_relation_stores_metadata_and_data(object_df_a, e2o_data):
    oced = OCEDStore()
    oced.add_object("A", "object", object_df_a)
    oced.add_event_to_object_relation("rel1", "from_id", "to_id", ["A"], e2o_data)
    rel = oced.get_relation("rel1")
    assert rel.from_id == "from_id"
    assert rel.to_id == "to_id"
    assert rel.to_type == ["A"]
    assert_frame_equal(rel.data, e2o_data)


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
    assert rel.from_id == "from_id"
    assert rel.from_type == ["A"]
    assert rel.to_id == "to_id"
    assert rel.to_type == ["B"]
    assert_frame_equal(rel.data, o2o_data)


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
    with pytest.raises(ValueError):
        minimal_oced.add_object_to_object_relation(
            "rel1",
            "from_id",
            ["A"],
            "to_id",
            ["B"],
            pd.DataFrame({"from_id": [1], "to_id": [4]}),
        )


def test_get_e2o_returns_only_event_to_object(minimal_oced):
    result = minimal_oced.get_e2o()
    assert "rel1" in result
    assert "rel2" not in result


def test_get_o2o_returns_only_object_to_object(minimal_oced):
    result = minimal_oced.get_o2o()
    assert "rel2" in result
    assert "rel1" not in result


def test_the_two_families_are_separate_maps(minimal_oced):
    assert set(minimal_oced.get_e2o()) == {"rel1"}
    assert set(minimal_oced.get_o2o()) == {"rel2"}


def test_get_relation_types_all_returns_both(minimal_oced):
    types = minimal_oced.get_relation_types(RelationKind.ALL)
    assert "rel1" in types
    assert "rel2" in types


def test_get_relation_types_e2o_returns_only_e2o(minimal_oced):
    types = minimal_oced.get_relation_types(RelationKind.E2O)
    assert "rel1" in types
    assert "rel2" not in types


def test_get_relation_types_o2o_returns_only_o2o(minimal_oced):
    types = minimal_oced.get_relation_types(RelationKind.O2O)
    assert "rel2" in types
    assert "rel1" not in types


def test_get_ids_event_returns_correct_ids(event_df_1):
    oced = OCEDStore()
    oced.add_event("type1", "event", "timestamp", event_df_1)
    ids = set(oced.get_ids(EntityKind.EVENT, "type1"))
    assert ids == {"e1", "e2", "e5"}


def test_get_ids_object_returns_correct_ids(object_df_a):
    oced = OCEDStore()
    oced.add_object("A", "object", object_df_a)
    ids = set(oced.get_ids(EntityKind.OBJECT, "A"))
    assert ids == {1, 2, 3}


def test_get_ids_invalid_of_raises():
    oced = OCEDStore()
    with pytest.raises(ValueError):
        oced.get_ids("bad", "whatever")


def test_get_ids_unknown_activity_raises():
    oced = OCEDStore()
    with pytest.raises(ValueError):
        oced.get_ids(EntityKind.EVENT, "no_such_activity")


def test_get_ids_unknown_object_raises():
    oced = OCEDStore()
    with pytest.raises(ValueError):
        oced.get_ids(EntityKind.OBJECT, "no_such_type")


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


def test_get_log_raises_for_unknown_case_concept(minimal_oced):
    with pytest.raises(ValueError, match="Unknown object type"):
        get_log(minimal_oced, "no_such_type")


def test_get_log_raises_when_no_e2o_references_case_concept():
    oced = OCEDStore()
    oced.add_object("B", "bid", pd.DataFrame({"bid": ["b1"]}))
    oced.add_object("A", "oid", pd.DataFrame({"oid": [1]}))
    ev = pd.DataFrame({"eid": ["e1"], "ts": pd.to_datetime(["2022-01-01"], utc=True)})
    oced.add_event("E", "eid", "ts", ev)
    oced.add_event_to_object_relation(
        "rel", "eid", "oid", ["A"], pd.DataFrame({"eid": ["e1"], "oid": [1]})
    )
    with pytest.raises(ValueError, match="No event-to-object relations"):
        get_log(oced, "B")


def test_get_log_returns_dataframe_with_pm4py_columns(minimal_oced):
    log = get_log(minimal_oced, "A")
    assert isinstance(log, pd.DataFrame)
    assert "case:concept:name" in log.columns
    assert "concept:name" in log.columns
    assert "time:timestamp" in log.columns


def test_get_log_row_count_and_activity_column(minimal_oced):
    log = get_log(minimal_oced, "A")
    assert len(log) == 5
    activity_counts = log["concept:name"].value_counts()
    assert activity_counts["type1"] == 3
    assert activity_counts["type2"] == 2


def test_iter_logs_matches_get_log_per_case_concept():
    oced = OCEDStore()
    oced.add_object("A", "oid", pd.DataFrame({"oid": ["a1", "a2"]}))
    oced.add_object("B", "bid", pd.DataFrame({"bid": ["b1"]}))
    oced.add_event(
        "E",
        "eid",
        "ts",
        pd.DataFrame(
            {"eid": ["e1", "e2"], "ts": pd.to_datetime(["2022-01-01", "2022-01-02"])}
        ),
    )
    oced.add_event_to_object_relation(
        "rel_a",
        "eid",
        "oid",
        ["A"],
        pd.DataFrame({"eid": ["e1", "e2"], "oid": ["a1", "a2"]}),
    )
    oced.add_event_to_object_relation(
        "rel_b", "eid", "bid", ["B"], pd.DataFrame({"eid": ["e1"], "bid": ["b1"]})
    )

    logs = dict(iter_logs(oced, ["A", "B"]))

    assert set(logs) == {"A", "B"}
    assert_frame_equal(logs["A"], get_log(oced, "A"))
    assert_frame_equal(logs["B"], get_log(oced, "B"))


def test_iter_logs_propagates_unknown_case_concept(minimal_oced):
    with pytest.raises(ValueError, match="Unknown object type"):
        list(iter_logs(minimal_oced, ["A", "no_such_type"]))


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
    assert oced.ocel_mode is False


def test_ocel_mode_true_when_set():
    oced = OCEDStore(ocel_mode=True)
    assert oced.ocel_mode is True


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


def test_id_backmap_empty_by_default():
    assert OCEDStore().backmap == {}


def test_get_original_id_returns_none_for_unknown():
    assert OCEDStore().get_original("not_in_map") is None


def test_get_original_id_looks_up_local_name():
    oced = OCEDStore()
    oced.backmap = {"order__123": "Order #123"}
    assert oced.get_original("order__123") == "Order #123"


def test_get_original_id_strips_default_base_iri():
    oced = OCEDStore()
    oced.backmap = {"order__123": "Order #123"}
    assert oced.get_original("https://oced#order__123") == "Order #123"


def test_get_original_id_strips_custom_base_iri():
    oced = OCEDStore()
    oced.backmap = {"order__123": "Order #123"}
    result = oced.get_original(
        "https://example.com/onto#order__123",
        base_iri="https://example.com/onto#",
    )
    assert result == "Order #123"


def test_the_store_takes_enums_not_strings(minimal_oced):
    assert set(minimal_oced.get_relation_types(RelationKind.E2O)) <= set(
        minimal_oced.get_relation_types(RelationKind.ALL)
    )
    assert minimal_oced.get_e2o().keys() == minimal_oced.get_e2o().keys()
    assert minimal_oced.get_o2o().keys() == minimal_oced.get_o2o().keys()


# --------------------------------------------------------------------------
# A rejected add leaves the store as it was
# --------------------------------------------------------------------------


def test_add_event_with_duplicate_ids_leaves_the_store_unchanged():
    oced = OCEDStore()
    oced.add_object("A", "oid", pd.DataFrame({"oid": ["shared"]}))
    clashing = pd.DataFrame(
        {"eid": ["shared"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
    )

    with pytest.raises(ValueError):
        oced.add_event("E", "eid", "ts", clashing)

    assert oced.get_event_types() == []


def test_add_object_with_duplicate_ids_adds_no_table():
    oced = OCEDStore()
    oced.add_object("A", "oid", pd.DataFrame({"oid": ["o1"]}))
    oced.add_object("B", "oid", pd.DataFrame({"oid": ["o2"]}))

    with pytest.raises(ValueError):
        oced.add_object("C", "oid", pd.DataFrame({"oid": ["o2"]}))

    assert oced.get_object_types() == ["A", "B"]
