"""Making names safe to emit, and recording what changed."""

import pandas as pd
import pytest

from semanticlog.data.cleaning import (
    RenameKind,
    RenameLog,
    _collect_schema_names,
    clean_object_changes,
    clean_oced_data,
    clean_string,
    clean_value,
    fix_duplicate_names,
    is_timestamp,
)
from semanticlog.data.oced import OCEDStore


def _oced_with_event(activity: str, attr_col: str = "status", attr_val: str = "ok"):
    oced = OCEDStore()
    ev = pd.DataFrame(
        {
            "eid": ["e1", "e2"],
            "ts": pd.date_range("2022-01-01", periods=2, tz="UTC"),
            attr_col: [attr_val, "normal_value"],
        }
    )
    oced.add_event(activity, "eid", "ts", ev)
    oced.add_object("obj", "oid", pd.DataFrame({"oid": ["o1"]}))
    return oced


@pytest.mark.parametrize(
    "s",
    ["2023-03-15T12:45:00Z", "2022-01-01", "2023-03-15 12:45:00"],
)
def test_is_timestamp_true_for_date_strings(s):
    assert is_timestamp(s) is True


@pytest.mark.parametrize("s", ["hello", "not_a_date", "activity_name", "some_value"])
def test_is_timestamp_false_for_plain_words(s):
    assert is_timestamp(s) is False


def test_clean_string_replaces_special_chars_with_underscore():
    assert clean_string("Hello World!", casefold=False) == "Hello_World_"


def test_clean_string_casefold_lowercases_result():
    assert clean_string("Hello World!", casefold=True) == "hello_world_"


def test_clean_string_preserves_alphanumeric_and_underscore():
    assert clean_string("abc_123", casefold=False) == "abc_123"


def test_clean_value_cleans_plain_string():
    assert clean_value("Hello World!") == "hello_world_"


def test_clean_value_leaves_timestamp_string_unchanged():
    ts = "2023-03-15T12:45:00Z"
    assert clean_value(ts) == ts


def test_clean_value_recursively_cleans_dict():
    result = clean_value({"Hello World!": "Test Value!"})
    assert result == {"hello_world_": "test_value_"}


def test_clean_value_cleans_dataframe_columns_and_values():
    df = pd.DataFrame({"Name!": ["Alice?", "Bob$"], "Age*": [25, 30]})
    cleaned = clean_value(df)
    assert "name_" in cleaned.columns
    assert "age_" in cleaned.columns
    assert cleaned["name_"].iloc[0] == "alice_"
    assert cleaned["age_"].iloc[0] == 25


def test_clean_value_passes_through_int():
    assert clean_value(42) == 42


def test_clean_value_passes_through_float():
    assert clean_value(3.14) == 3.14


def test_clean_value_passes_through_bool():
    assert clean_value(True) is True


def test_clean_value_cleans_list_elements():
    result = clean_value(["Hello!", "World?"])
    assert result == ["hello_", "world_"]


def test_clean_value_passes_through_timestamp_object():
    ts = pd.Timestamp("2022-01-01", tz="UTC")
    assert clean_value(ts) == ts


def test_fix_duplicate_names_renames_reserved_relation():
    oced = OCEDStore()
    oced.add_object("A", "oid", pd.DataFrame({"oid": [1, 2]}))
    ev = pd.DataFrame(
        {"eid": ["e1", "e2"], "ts": pd.date_range("2022-01-01", periods=2, tz="UTC")}
    )
    oced.add_event("act", "eid", "ts", ev)
    oced.add_event_to_object_relation(
        "df", "eid", "oid", ["A"], pd.DataFrame({"eid": ["e1", "e2"], "oid": [1, 2]})
    )

    result = fix_duplicate_names(oced)

    assert "df_" in result.get_relation_types()
    assert "df" not in result.get_relation_types()


def test_fix_duplicate_names_renames_reserved_activity():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
    )
    oced.add_event("observes", "eid", "ts", ev)
    oced.add_object("obj", "oid", pd.DataFrame({"oid": ["o1"]}))

    result = fix_duplicate_names(oced)

    assert "observes_" in result.get_event_types()
    assert "observes" not in result.get_event_types()


def test_fix_duplicate_names_renames_activity_colliding_with_derived_property():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
    )
    oced.add_event("has_price", "eid", "ts", ev)
    oced.add_object("obj", "oid", pd.DataFrame({"oid": ["o1"], "price": [10.0]}))

    result = fix_duplicate_names(oced)

    assert "has_price_" in result.get_event_types()
    assert "has_price" not in result.get_event_types()


def test_fix_duplicate_names_renames_object_colliding_with_case_concept_property():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
    )
    oced.add_event("df_order", "eid", "ts", ev)
    oced.add_object("order", "oid", pd.DataFrame({"oid": ["o1"]}))

    result = fix_duplicate_names(oced)

    assert "df_order_" in result.get_event_types()


def test_fix_duplicate_names_leaves_unrelated_derived_lookalikes_alone():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
    )
    oced.add_event("has_weight", "eid", "ts", ev)
    oced.add_object("obj", "oid", pd.DataFrame({"oid": ["o1"], "price": [10.0]}))

    result = fix_duplicate_names(oced)

    assert "has_weight" in result.get_event_types()


def test_fix_duplicate_names_renames_object_colliding_with_activity():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
    )
    oced.add_event("my_activity", "eid", "ts", ev)
    oced.add_object("my_activity", "oid", pd.DataFrame({"oid": ["o1"]}))

    result = fix_duplicate_names(oced)

    assert "my_activity_" in result.get_object_types()
    assert "my_activity" not in result.get_object_types()


def test_fix_duplicate_names_renames_attribute_col_colliding_with_object_type():
    oced = OCEDStore()
    oced.add_object(
        "TypeA",
        "oid",
        pd.DataFrame({"oid": ["o1"], "TypeA": ["some_val"]}),
    )

    result = fix_duplicate_names(oced)

    obj_df = result.get_object_table("TypeA").data
    assert "TypeA_attr" in obj_df.columns
    assert "TypeA" not in obj_df.columns


def test_fix_duplicate_names_renames_attribute_value_colliding_with_reserved():
    oced = _oced_with_event("activity", attr_col="status", attr_val="is_a")

    result = fix_duplicate_names(oced)

    status_vals = list(result.get_event_table("activity").data["status"])
    assert "is_a_attr_val" in status_vals
    assert "normal_value" in status_vals


def test_clean_oced_data_raises_on_colliding_activity_names():
    oced = OCEDStore()
    ev1 = pd.DataFrame(
        {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
    )
    ev2 = pd.DataFrame(
        {"eid": ["e2"], "ts": pd.date_range("2022-01-02", periods=1, tz="UTC")}
    )
    oced.add_event("Place Order", "eid", "ts", ev1)
    oced.add_event("Place_Order", "eid", "ts", ev2)
    oced.add_object("obj", "oid", pd.DataFrame({"oid": ["o1"]}))
    with pytest.raises(ValueError, match="activities"):
        clean_oced_data(oced)


def test_clean_oced_data_raises_on_colliding_object_type_names():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
    )
    oced.add_event("act", "eid", "ts", ev)
    oced.add_object("Order Item", "oid", pd.DataFrame({"oid": ["o1"]}))
    oced.add_object("Order_Item", "oid2", pd.DataFrame({"oid2": ["o2"]}))
    with pytest.raises(ValueError, match="object types"):
        clean_oced_data(oced)


def test_clean_oced_data_raises_on_colliding_relation_names():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["e1", "e2"], "ts": pd.date_range("2022-01-01", periods=2, tz="UTC")}
    )
    oced.add_event("act", "eid", "ts", ev)
    oced.add_object("obj", "oid", pd.DataFrame({"oid": ["o1"]}))
    oced.add_event_to_object_relation(
        "places order",
        "eid",
        "oid",
        ["obj"],
        pd.DataFrame({"eid": ["e1"], "oid": ["o1"]}),
    )
    oced.add_event_to_object_relation(
        "places_order",
        "eid",
        "oid",
        ["obj"],
        pd.DataFrame({"eid": ["e2"], "oid": ["o1"]}),
    )
    with pytest.raises(ValueError, match="e2o relations"):
        clean_oced_data(oced)


def test_clean_oced_data_no_raise_when_names_are_distinct_after_cleaning():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
    )
    oced.add_event("Place Order", "eid", "ts", ev)
    oced.add_object("obj", "oid", pd.DataFrame({"oid": ["o1"]}))
    clean_oced_data(oced)


def test_clean_oced_data_sanitizes_special_chars_in_all_strings():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {
            "eid": ["e1"],
            "ts": pd.date_range("2022-01-01", periods=1, tz="UTC"),
            "some attr!": ["val?"],
        }
    )
    oced.add_event("activity type!", "eid", "ts", ev)
    oced.add_object(
        "obj type!", "oid", pd.DataFrame({"oid": ["o1"], "data!": ["val!"]})
    )
    oced.add_event_to_object_relation(
        "rel name!",
        "eid",
        "oid",
        ["obj type!"],
        pd.DataFrame({"eid": ["e1"], "oid": ["o1"]}),
    )

    result = clean_oced_data(oced)

    assert "activity_type_" in result.get_event_types()
    assert "obj_type_" in result.get_object_types()
    assert "rel_name_" in result.get_relation_types()
    ev_data = result.get_event_table("activity_type_").data
    assert "some_attr_" in ev_data.columns
    assert "val_" in ev_data["some_attr_"].values


def test_clean_oced_data_applies_fix_duplicate_names():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
    )
    oced.add_event("df", "eid", "ts", ev)
    oced.add_object("obj", "oid", pd.DataFrame({"oid": ["o1"]}))

    result = clean_oced_data(oced)

    assert "df_" in result.get_event_types()
    assert "df" not in result.get_event_types()


def test_clean_oced_data_backmap_maps_special_chars_to_original():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
    )
    oced.add_event("act", "eid", "ts", ev)
    oced.add_object("obj", "oid", pd.DataFrame({"oid": ["Order #123"]}))

    result = clean_oced_data(oced)

    assert result.backmap["Order__123"] == "Order #123"


def test_clean_oced_data_backmap_omits_already_clean_ids():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["e1", "e2"], "ts": pd.date_range("2022-01-01", periods=2, tz="UTC")}
    )
    oced.add_event("act", "eid", "ts", ev)
    oced.add_object("obj", "oid", pd.DataFrame({"oid": ["o1"]}))

    result = clean_oced_data(oced)

    assert "e1" not in result.backmap
    assert "e2" not in result.backmap


def test_clean_oced_data_backmap_respects_casefold():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["E1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
    )
    oced.add_event("act", "eid", "ts", ev)
    oced.add_object("obj", "oid", pd.DataFrame({"oid": ["o1"]}))

    result = clean_oced_data(oced, casefold=True)

    assert result.backmap["e1"] == "E1"


def test_clean_oced_data_backmap_maps_attr_value_to_original():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
    )
    oced.add_event("act", "eid", "ts", ev)
    oced.add_object(
        "obj",
        "oid",
        pd.DataFrame({"oid": ["o1"], "notes": ["Hello World!"]}),
    )

    result = clean_oced_data(oced)

    assert result.backmap["notes_Hello_World_"] == "Hello World!"


def test_clean_oced_data_backmap_omits_already_clean_attr_value():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
    )
    oced.add_event("act", "eid", "ts", ev)
    oced.add_object(
        "obj",
        "oid",
        pd.DataFrame({"oid": ["o1"], "status": ["open"]}),
    )

    result = clean_oced_data(oced)

    assert "status_open" not in result.backmap


def test_clean_oced_data_backmap_catches_reserved_name_collision():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["is_a"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
    )
    oced.add_event("act", "eid", "ts", ev)
    oced.add_object("obj", "oid", pd.DataFrame({"oid": ["o1"]}))

    result = clean_oced_data(oced)

    assert result.backmap.get("is_a_attr_val") == "is_a"


def test_collect_schema_names_returns_activity_and_object_type():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
    )
    oced.add_event("Order Execution", "eid", "ts", ev)
    oced.add_object("Purchase Item", "oid", pd.DataFrame({"oid": ["o1"]}))
    names = _collect_schema_names(oced)
    assert "Order Execution" in names
    assert "Purchase Item" in names


def test_collect_schema_names_includes_attribute_columns():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {
            "eid": ["e1"],
            "ts": pd.date_range("2022-01-01", periods=1, tz="UTC"),
            "total cost": ["100"],
        }
    )
    oced.add_event("act", "eid", "ts", ev)
    oced.add_object("obj", "oid", pd.DataFrame({"oid": ["o1"], "item type": ["A"]}))
    names = _collect_schema_names(oced)
    assert "total cost" in names
    assert "item type" in names
    assert "eid" not in names
    assert "ts" not in names
    assert "oid" not in names


def test_rename_log_maps_a_final_name_to_its_original():
    log = RenameLog()
    log.record(RenameKind.ACTIVITY, "Order Execution", "Order_Execution")

    assert log.flat()["Order_Execution"] == "Order Execution"


def test_rename_log_composes_successive_renames():
    log = RenameLog()
    log.record(RenameKind.ACTIVITY, "Has Price", "Has_Price")
    log.record(RenameKind.ACTIVITY, "Has_Price", "Has_Price_")

    assert log.flat() == {"Has_Price_": "Has Price"}


def test_rename_log_omits_names_it_did_not_change():
    log = RenameLog()
    log.record(RenameKind.ACTIVITY, "act", "act")

    assert log.flat() == {}


def test_rename_log_keeps_kinds_apart():
    log = RenameLog()
    log.record(RenameKind.ID, "o 1", "o_1")
    log.record(RenameKind.ACTIVITY, "o.1", "o_1")

    assert log.of_kind(RenameKind.ID) == {"o_1": "o 1"}
    assert log.of_kind(RenameKind.ACTIVITY) == {"o_1": "o.1"}


def test_rename_log_kinds_are_an_enum_rather_than_free_strings():
    assert {k.value for k in RenameKind} == {
        "activity",
        "object_type",
        "relation",
        "attribute",
        "id",
        "value",
    }


def test_rename_log_takes_an_explicit_original():
    log = RenameLog()
    log.record(RenameKind.VALUE, "status_df", "status_df_attr_val", original="df")

    assert log.flat()["status_df_attr_val"] == "df"


def test_backmap_records_a_reserved_activity_suffix():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
    )
    oced.add_event("observes", "eid", "ts", ev)
    oced.add_object("obj", "oid", pd.DataFrame({"oid": ["o1"]}))

    result = clean_oced_data(oced)

    assert result.backmap["observes_"] == "observes"


def test_backmap_records_an_attribute_column_suffix():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
    )
    oced.add_event("act", "eid", "ts", ev)
    oced.add_object("obj", "oid", pd.DataFrame({"oid": ["o1"], "observed_at": ["x"]}))

    result = clean_oced_data(oced)

    assert result.backmap["observed_at_attr"] == "observed_at"


def test_clean_oced_data_name_backmap_maps_activity_and_object_type():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
    )
    oced.add_event("Order Execution", "eid", "ts", ev)
    oced.add_object("Purchase Item", "oid", pd.DataFrame({"oid": ["o1"]}))
    result = clean_oced_data(oced)
    assert result.backmap["Order_Execution"] == "Order Execution"
    assert result.backmap["Purchase_Item"] == "Purchase Item"


def test_clean_oced_data_name_backmap_omits_already_clean_names():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
    )
    oced.add_event("act", "eid", "ts", ev)
    oced.add_object("obj", "oid", pd.DataFrame({"oid": ["o1"]}))
    result = clean_oced_data(oced)
    assert "act" not in result.backmap
    assert "obj" not in result.backmap


def test_clean_oced_data_name_backmap_maps_attribute_columns():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {
            "eid": ["e1"],
            "ts": pd.date_range("2022-01-01", periods=1, tz="UTC"),
            "total cost": ["100"],
        }
    )
    oced.add_event("act", "eid", "ts", ev)
    oced.add_object("obj", "oid", pd.DataFrame({"oid": ["o1"]}))
    result = clean_oced_data(oced)
    assert result.backmap["total_cost"] == "total cost"


def test_clean_object_changes_cleans_ocel_type():
    df = pd.DataFrame(
        {
            "ocel:oid": ["o1"],
            "ocel:type": ["order item"],
            "ocel:timestamp": pd.to_datetime(["2022-01-01"], utc=True),
            "ocel:field": ["status"],
            "status": ["pending"],
        }
    )
    result = clean_object_changes(df)
    assert result["ocel:type"].iloc[0] == "order_item"


def test_clean_object_changes_preserves_ocel_columns():
    df = pd.DataFrame(
        {
            "ocel:oid": ["o1"],
            "ocel:type": ["order"],
            "ocel:timestamp": pd.to_datetime(["2022-01-01"], utc=True),
            "ocel:field": ["status"],
            "status": ["pending"],
        }
    )
    result = clean_object_changes(df)
    assert "ocel:oid" in result.columns
    assert "ocel:type" in result.columns
    assert "ocel:field" in result.columns
    assert "ocel:timestamp" in result.columns


def _make_oced_with_changes(obj_type: str) -> OCEDStore:
    oced = OCEDStore(ocel_mode=True)
    oced.add_object(obj_type, "oid", pd.DataFrame({"oid": ["o1"]}))
    ev = pd.DataFrame({"eid": ["e1"], "ts": pd.to_datetime(["2022-01-01"], utc=True)})
    oced.add_event("act", "eid", "ts", ev)
    oced.add_event_to_object_relation(
        "rel", "eid", "oid", [obj_type], pd.DataFrame({"eid": ["e1"], "oid": ["o1"]})
    )
    changes = pd.DataFrame(
        {
            "ocel:oid": ["o1"],
            "ocel:type": [obj_type],
            "ocel:timestamp": pd.to_datetime(["2022-01-02"], utc=True),
            "ocel:field": ["status"],
            "status": ["pending"],
        }
    )
    oced.add_object_changes(obj_type, changes)
    return oced


def test_clean_oced_data_cleans_object_changes_dict_key():
    oced = _make_oced_with_changes("order item")
    result = clean_oced_data(oced)
    assert "order_item" in result.get_object_change_types()
    assert "order item" not in result.get_object_change_types()


def test_clean_oced_data_cleans_ocel_type_in_object_changes():
    oced = _make_oced_with_changes("order item")
    result = clean_oced_data(oced)
    changes = result.get_object_changes("order_item")
    assert (changes["ocel:type"] == "order_item").all()


def _store_with_colliding_object_id() -> OCEDStore:
    oced = OCEDStore()
    oced.add_event(
        "act",
        "eid",
        "ts",
        pd.DataFrame(
            {"eid": ["e1"], "ts": pd.to_datetime(["2024-01-01"], utc=True)},
        ),
    )
    oced.add_object(
        "order", "oid", pd.DataFrame({"oid": ["Event", "o2"], "attr": ["x", "y"]})
    )
    oced.add_event_to_object_relation(
        "modifies",
        "eid",
        "oid",
        ["order"],
        pd.DataFrame({"eid": ["e1"], "oid": ["Event"]}),
    )
    return oced


def test_a_renamed_object_id_is_rewritten_in_the_relation_table_too():
    oced = clean_oced_data(_store_with_colliding_object_id())

    object_ids = set(oced.get_object_table("order").data["oid"])
    relation = oced.get_relation("modifies")
    referenced = set(relation.data[relation.to_id])

    assert "Event" not in object_ids, "the colliding id should have been renamed"
    assert referenced <= object_ids, (
        f"relation references {referenced - object_ids} which no object table holds"
    )


def test_object_change_tables_are_rewritten_by_the_same_map():
    oced = OCEDStore(ocel_mode=True)
    oced.add_event(
        "act",
        "eid",
        "ts",
        pd.DataFrame({"eid": ["e1"], "ts": pd.to_datetime(["2024-01-01"], utc=True)}),
    )
    oced.add_object("order", "oid", pd.DataFrame({"oid": ["Event"], "status": ["new"]}))
    oced.add_event_to_object_relation(
        "modifies",
        "eid",
        "oid",
        ["order"],
        pd.DataFrame({"eid": ["e1"], "oid": ["Event"]}),
    )
    oced.replace_object_changes(
        {
            "order": pd.DataFrame(
                {
                    "ocel:oid": ["Event"],
                    "ocel:type": ["order"],
                    "ocel:field": ["status"],
                    "ocel:timestamp": pd.to_datetime(["2024-01-02"], utc=True),
                    "status": ["shipped"],
                }
            )
        }
    )

    cleaned = clean_oced_data(oced)

    object_ids = set(cleaned.get_object_table("order").data["oid"])
    changes = cleaned.get_object_changes("order")
    assert set(changes["ocel:oid"]) <= object_ids
    assert set(changes["ocel:type"]) == {"order"}
    assert set(changes["ocel:field"]) == {"status"}


def test_two_attribute_columns_cleaning_to_one_name_are_named():
    oced = OCEDStore()
    oced.add_event(
        "act",
        "eid",
        "ts",
        pd.DataFrame({"eid": ["e1"], "ts": pd.to_datetime(["2024-01-01"], utc=True)}),
    )
    oced.add_object(
        "order",
        "oid",
        pd.DataFrame({"oid": ["o1"], "unit price": [1.0], "unit-price": [2.0]}),
    )

    with pytest.raises(ValueError, match="unit_price") as excinfo:
        clean_oced_data(oced)

    message = str(excinfo.value)
    assert "unit price" in message
    assert "unit-price" in message
    assert "order" in message


def test_the_same_attribute_on_two_tables_is_not_a_collision():
    oced = OCEDStore()
    oced.add_event(
        "act",
        "eid",
        "ts",
        pd.DataFrame(
            {
                "eid": ["e1"],
                "ts": pd.to_datetime(["2024-01-01"], utc=True),
                "channel": ["web"],
            }
        ),
    )
    oced.add_object("order", "oid", pd.DataFrame({"oid": ["o1"], "channel": ["shop"]}))

    clean_oced_data(oced)


def test_a_rename_never_discards_an_existing_type():
    oced = OCEDStore()
    oced.add_event(
        "act",
        "eid",
        "ts",
        pd.DataFrame({"eid": ["e1"], "ts": pd.to_datetime(["2024-01-01"], utc=True)}),
    )
    oced.add_object("Event", "oid", pd.DataFrame({"oid": ["a1"]}))
    oced.add_object("Event_", "oid2", pd.DataFrame({"oid2": ["b1"]}))

    fix_duplicate_names(oced)

    kept = oced.get_object_tables()
    assert len(kept) == 2, f"an object type was discarded: {sorted(kept)}"
    assert all(len(table.data) == 1 for table in kept.values())


def test_renaming_onto_an_existing_type_is_refused_outright():
    oced = OCEDStore()
    oced.add_object("a", "oid", pd.DataFrame({"oid": ["a1"]}))
    oced.add_object("b", "oid", pd.DataFrame({"oid": ["b1"]}))

    with pytest.raises(ValueError, match="already exists"):
        oced.rename_object("a", "b")
