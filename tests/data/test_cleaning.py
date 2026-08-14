"""Making names safe to emit, and recording what changed."""

import pandas as pd
import pytest

from semanticlog.data.cleaning import (
    RenameKind,
    RenameLog,
    clean_object_changes,
    clean_oced_data,
    clean_string,
    clean_value,
    fix_duplicate_names,
    is_timestamp,
    object_type_renames,
)
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import RelationKind


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
    assert clean_string("Hello World!") == "Hello_World_"


def test_clean_string_preserves_alphanumeric_and_underscore():
    assert clean_string("abc_123") == "abc_123"


def test_clean_value_cleans_plain_string():
    assert clean_value("Hello World!") == "Hello_World_"


def test_clean_value_leaves_timestamp_string_unchanged():
    ts = "2023-03-15T12:45:00Z"
    assert clean_value(ts) == ts


def test_clean_value_recursively_cleans_dict():
    result = clean_value({"Hello World!": "Test Value!"})
    assert result == {"Hello_World_": "Test_Value_"}


def test_clean_value_cleans_dataframe_columns_and_values():
    df = pd.DataFrame({"Name!": ["Alice?", "Bob$"], "Age*": [25, 30]})
    cleaned = clean_value(df)
    assert "Name_" in cleaned.columns
    assert "Age_" in cleaned.columns
    assert cleaned["Name_"].iloc[0] == "Alice_"
    assert cleaned["Age_"].iloc[0] == 25


def test_clean_value_passes_through_int():
    assert clean_value(42) == 42


def test_clean_value_passes_through_float():
    assert clean_value(3.14) == 3.14


def test_clean_value_passes_through_bool():
    assert clean_value(True) is True


def test_clean_value_cleans_list_elements():
    result = clean_value(["Hello!", "World?"])
    assert result == ["Hello_", "World_"]


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


def test_two_activities_cleaning_to_one_name_are_kept_apart():
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

    cleaned = clean_oced_data(oced)

    assert sorted(cleaned.get_event_types()) == ["Place_Order", "Place_Order_2"]
    assert cleaned.backmap["Place_Order"] == "Place Order"
    assert cleaned.backmap["Place_Order_2"] == "Place_Order"


def test_two_object_types_cleaning_to_one_name_are_kept_apart():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
    )
    oced.add_event("act", "eid", "ts", ev)
    oced.add_object("Order Item", "oid", pd.DataFrame({"oid": ["o1"]}))
    oced.add_object("Order_Item", "oid2", pd.DataFrame({"oid2": ["o2"]}))

    cleaned = clean_oced_data(oced)

    assert sorted(cleaned.get_object_types()) == ["Order_Item", "Order_Item_2"]
    assert cleaned.backmap["Order_Item_2"] == "Order_Item"


def test_two_relations_cleaning_to_one_name_are_kept_apart():
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

    cleaned = clean_oced_data(oced)

    assert sorted(cleaned.get_relation_types()) == ["places_order", "places_order_2"]
    assert cleaned.backmap["places_order_2"] == "places_order"


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


def test_rename_log_groups_what_it_renamed_by_kind():
    log = RenameLog()
    log.record(RenameKind.ACTIVITY, "Order Execution", "Order_Execution")
    log.record(RenameKind.ID, "o 1", "o_1")
    log.record(RenameKind.OBJECT_TYPE, "item", "item")
    log.record_value("colour", "off_white", "off white")

    assert log.by_kind() == {
        "activity": ["Order Execution -> Order_Execution"],
        "id": ["o 1 -> o_1"],
        "value": ["off white -> colour_off_white"],
    }


def test_rename_log_keeps_kinds_apart():
    log = RenameLog()
    log.record(RenameKind.ID, "o 1", "o_1")
    log.record(RenameKind.ACTIVITY, "o.1", "o_1")

    assert log.by_kind() == {
        "id": ["o 1 -> o_1"],
        "activity": ["o.1 -> o_1"],
    }


def test_rename_log_kinds_are_an_enum_rather_than_free_strings():
    assert {k.value for k in RenameKind} == {
        "activity",
        "object_type",
        "relation",
        "attribute",
        "id",
    }


def test_rename_log_takes_an_explicit_original():
    log = RenameLog()
    log.record(RenameKind.ID, "o 1", "o_1", original="o~1")

    assert log.flat()["o_1"] == "o~1"


def test_rename_log_names_a_value_after_its_column():
    log = RenameLog()
    log.record_value("status", "in_progress", "in progress")

    assert log.flat() == {"status_in_progress": "in progress"}


def test_rename_log_carries_values_along_when_their_column_is_renamed():
    """The individual is named after the column, so a renamed column moves it."""
    log = RenameLog()
    log.record_value("event", "by_hand", "by hand")

    log.rename_value_column("event", "event_attr")

    assert log.flat() == {"event_attr_by_hand": "by hand"}


def test_rename_log_keeps_the_source_value_through_a_second_rewrite():
    log = RenameLog()
    log.record_value("status", "df", "d f")

    log.rename_value("status", "df", "df_attr_val")

    assert log.flat() == {"status_df_attr_val": "d f"}


def test_rename_log_records_a_suffixed_value_that_cleaning_left_alone():
    log = RenameLog()

    log.rename_value("status", "df", "df_attr_val")

    assert log.flat() == {"status_df_attr_val": "df"}


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


def _store_with_history(
    field: str,
    values: list[str],
    object_values: list[str] | None = None,
    object_ids: list[str] | None = None,
) -> OCEDStore:
    """One object type whose attribute both varies and has a recorded history."""
    oids = object_ids or ["o1"]
    oced = OCEDStore(ocel_mode=True)
    oced.add_event(
        "act",
        "eid",
        "ts",
        pd.DataFrame({"eid": ["e1"], "ts": pd.to_datetime(["2024-01-01"], utc=True)}),
    )
    columns: dict = {"oid": oids}
    if object_values is not None:
        columns[field] = object_values
    oced.add_object("order", "oid", pd.DataFrame(columns))
    oced.add_event_to_object_relation(
        "modifies",
        "eid",
        "oid",
        ["order"],
        pd.DataFrame({"eid": ["e1"], "oid": [oids[0]]}),
    )
    oced.add_object_changes(
        "order",
        pd.DataFrame(
            {
                "ocel:oid": oids[: len(values)],
                "ocel:type": ["order"] * len(values),
                "ocel:timestamp": pd.to_datetime(
                    ["2024-01-02"] * len(values), utc=True
                ),
                "ocel:field": [field] * len(values),
                field: values,
            }
        ),
    )
    return oced


def test_a_value_renamed_apart_is_renamed_in_the_history_too():
    """Otherwise the change records the value of the object it is not about."""
    oced = _store_with_history(
        "state",
        values=["in progress", "in/progress"],
        object_values=["in progress", "in/progress"],
        object_ids=["o1", "o2"],
    )

    cleaned = clean_oced_data(oced)

    assert list(cleaned.get_object_table("order").data["state"]) == list(
        cleaned.get_object_changes("order")["state"]
    )


def test_an_attribute_renamed_by_kind_stays_one_attribute_in_the_history():
    """The history would otherwise declare a second class under the old name."""
    oced = _store_with_history("act", values=["done"], object_values=["open"])

    cleaned = clean_oced_data(oced)

    changes = cleaned.get_object_changes("order")
    assert "attribute_act" in cleaned.get_object_table("order").data.columns
    assert "attribute_act" in changes.columns
    assert set(changes["ocel:field"]) == {"attribute_act"}


def test_an_attribute_renamed_off_a_reserved_name_takes_its_history_with_it():
    oced = _store_with_history("observes", values=["x"], object_values=["x"])

    cleaned = clean_oced_data(oced)

    changes = cleaned.get_object_changes("order")
    assert "observes_attr" in changes.columns
    assert set(changes["ocel:field"]) == {"observes_attr"}
    assert "observes" not in changes.columns


def test_an_attribute_only_the_history_carries_is_reserved_like_any_other():
    """A column empty for every object is dropped on load but still changes."""
    oced = _store_with_history("act", values=["done"])

    cleaned = clean_oced_data(oced)

    changes = cleaned.get_object_changes("order")
    assert "act" not in cleaned.get_event_types()
    assert set(changes["ocel:field"]) == {"attribute_act"}


def test_two_attribute_columns_cleaning_to_one_name_are_kept_apart():
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
        pd.DataFrame({"oid": ["o1"], "unit price": [1.0], "unit/price": [2.0]}),
    )

    cleaned = clean_oced_data(oced)

    columns = list(cleaned.get_object_table("order").data.columns)
    assert columns == ["oid", "unit_price", "unit_price_2"]
    assert cleaned.backmap["unit_price"] == "unit price"
    assert cleaned.backmap["unit_price_2"] == "unit/price"


def test_a_hyphen_and_a_dot_are_kept_so_the_names_stay_apart():
    """Both survive an XML element name, so folding them would merge for nothing."""
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
        pd.DataFrame({"oid": ["o1"], "unit-price": [1.0], "unit.price": [2.0]}),
    )

    cleaned = clean_oced_data(oced)

    columns = list(cleaned.get_object_table("order").data.columns)
    assert columns == ["oid", "unit-price", "unit.price"]


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


# --------------------------------------------------------------------------
# Names that would clean to one name are kept apart, not merged
# --------------------------------------------------------------------------


def _store_with_values(*values: str, column: str = "channel") -> OCEDStore:
    oced = OCEDStore()
    oced.add_object("order", "oid", pd.DataFrame({"oid": ["o1"]}))
    oced.add_event(
        "step",
        "eid",
        "ts",
        pd.DataFrame(
            {
                "eid": [f"e{i}" for i in range(len(values))],
                "ts": pd.date_range("2022-01-01", periods=len(values), tz="UTC"),
                column: list(values),
            }
        ),
    )
    return oced


def test_two_attribute_values_that_clean_alike_are_kept_apart():
    """Both become one individual otherwise, so one source fact disappears."""
    oced = _store_with_values("a b", "a_b")

    cleaned = clean_oced_data(oced)

    assert sorted(cleaned.get_event_table("step").data["channel"]) == ["a_b", "a_b_2"]
    assert cleaned.backmap["channel_a_b"] == "a b"
    assert cleaned.backmap["channel_a_b_2"] == "a_b"


def test_a_value_keeps_its_hyphen_and_stays_its_own_individual():
    oced = _store_with_values("a b", "a-b", column="channel")

    cleaned = clean_oced_data(oced)

    assert sorted(cleaned.get_event_table("step").data["channel"]) == ["a-b", "a_b"]


def test_one_value_repeated_across_rows_is_not_a_collision():
    assert clean_oced_data(_store_with_values("web", "web", "shop"))


def test_the_same_value_under_two_attributes_is_not_a_collision():
    oced = OCEDStore()
    oced.add_object("order", "oid", pd.DataFrame({"oid": ["o1"], "channel": ["a b"]}))
    oced.add_event(
        "step",
        "eid",
        "ts",
        pd.DataFrame(
            {
                "eid": ["e1"],
                "ts": pd.date_range("2022-01-01", periods=1, tz="UTC"),
                "clerk": ["a b"],
            }
        ),
    )

    assert clean_oced_data(oced)


def test_one_attribute_carried_by_two_tables_is_resolved_as_one_pool():
    """Two tables sharing an attribute name share its value individuals."""
    oced = OCEDStore()
    oced.add_object("order", "oid", pd.DataFrame({"oid": ["o1"], "channel": ["a b"]}))
    oced.add_object("item", "iid", pd.DataFrame({"iid": ["i1"], "channel": ["a_b"]}))

    cleaned = clean_oced_data(oced)

    order = list(cleaned.get_object_table("order").data["channel"])
    item = list(cleaned.get_object_table("item").data["channel"])
    assert sorted(order + item) == ["a_b", "a_b_2"]


def test_a_numeric_column_is_not_checked_for_value_collisions():
    oced = OCEDStore()
    oced.add_object(
        "order", "oid", pd.DataFrame({"oid": ["o1", "o2"], "price": [1.5, 1.5]})
    )

    assert clean_oced_data(oced)


def test_ids_are_resolved_across_the_whole_store_not_per_table():
    """The store holds one id namespace, so a cross-table clash is one too."""
    oced = OCEDStore()
    oced.add_object("order", "oid", pd.DataFrame({"oid": ["a_1"]}))
    oced.add_event(
        "step",
        "eid",
        "ts",
        pd.DataFrame(
            {"eid": ["a 1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
        ),
    )

    cleaned = clean_oced_data(oced)

    object_ids = list(cleaned.get_object_table("order").data["oid"])
    event_ids = list(cleaned.get_event_table("step").data["eid"])
    assert sorted(object_ids + event_ids) == ["a_1", "a_1_2"]


def test_one_name_claimed_by_three_kinds_is_prefixed_by_kind():
    """An activity, an object type and an attribute each become a class."""
    oced = OCEDStore()
    oced.add_event(
        "order",
        "eid",
        "ts",
        pd.DataFrame(
            {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
        ),
    )
    oced.add_object("order", "oid", pd.DataFrame({"oid": ["o1"], "order": ["x"]}))

    cleaned = clean_oced_data(oced)

    assert cleaned.get_event_types() == ["event_order"]
    assert cleaned.get_object_types() == ["object_order"]
    columns = list(cleaned.get_object_table("object_order").data.columns)
    assert columns == ["oid", "attribute_order"]
    assert cleaned.backmap["event_order"] == "order"


def test_a_value_follows_the_column_that_a_kind_prefix_renamed():
    oced = OCEDStore()
    oced.add_event(
        "order",
        "eid",
        "ts",
        pd.DataFrame(
            {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
        ),
    )
    oced.add_object(
        "item",
        "oid",
        pd.DataFrame({"oid": ["o1", "o2"], "order": ["a b", "a_b"]}),
    )

    cleaned = clean_oced_data(oced)

    assert cleaned.backmap["attribute_order_a_b"] == "a b"
    assert cleaned.backmap["attribute_order_a_b_2"] == "a_b"


def test_relations_are_resolved_across_both_families_not_per_family():
    """Both families become properties of one ontology, so they share a namespace."""
    oced = OCEDStore()
    oced.add_event(
        "act",
        "eid",
        "ts",
        pd.DataFrame(
            {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
        ),
    )
    oced.add_object("order", "oid", pd.DataFrame({"oid": ["o1", "o2"]}))
    oced.add_event_to_object_relation(
        "contains x",
        "eid",
        "oid",
        ["order"],
        pd.DataFrame({"eid": ["e1"], "oid": ["o1"]}),
    )
    oced.add_object_to_object_relation(
        "contains_x",
        "from_oid",
        ["order"],
        "to_oid",
        ["order"],
        pd.DataFrame({"from_oid": ["o1"], "to_oid": ["o2"]}),
    )

    cleaned = clean_oced_data(oced)

    assert cleaned.get_relation_types(RelationKind.E2O) == ["contains_x"]
    assert cleaned.get_relation_types(RelationKind.O2O) == ["contains_x_2"]
    assert cleaned.backmap["contains_x"] == "contains x"
    assert cleaned.backmap["contains_x_2"] == "contains_x"


def test_a_relation_cannot_open_a_property_name_with_a_digit():
    """RDF/XML writes a property as an element, which cannot start with one."""
    oced = OCEDStore()
    oced.add_event(
        "act",
        "eid",
        "ts",
        pd.DataFrame(
            {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
        ),
    )
    oced.add_object("order", "oid", pd.DataFrame({"oid": ["o1"]}))
    oced.add_event_to_object_relation(
        "2modifies",
        "eid",
        "oid",
        ["order"],
        pd.DataFrame({"eid": ["e1"], "oid": ["o1"]}),
    )

    cleaned = clean_oced_data(oced)

    assert cleaned.get_relation_types() == ["_2modifies"]
    assert cleaned.backmap["_2modifies"] == "2modifies"


def test_an_object_type_gives_way_to_a_relation_of_the_same_name():
    """A class and an object property cannot share a name in OWL 2 DL."""
    oced = OCEDStore()
    oced.add_event(
        "place order",
        "eid",
        "ts",
        pd.DataFrame(
            {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
        ),
    )
    oced.add_object("order", "oid", pd.DataFrame({"oid": ["o1"]}))
    oced.add_event_to_object_relation(
        "order",
        "eid",
        "oid",
        ["order"],
        pd.DataFrame({"eid": ["e1"], "oid": ["o1"]}),
    )

    cleaned = clean_oced_data(oced)

    assert cleaned.get_relation_types() == ["order"]
    assert cleaned.get_object_types() == ["order_"]
    assert cleaned.get_relation("order").to_type == ["order_"]
    assert object_type_renames(cleaned) == {"order": "order_"}


def test_object_type_renames_leave_out_the_other_kinds_of_name():
    oced = _oced_with_event("place order")

    cleaned = clean_oced_data(oced)

    assert cleaned.backmap == {"place_order": "place order"}
    assert object_type_renames(cleaned) == {}


def test_an_id_that_carries_a_relation_is_renamed_on_both_sides():
    oced = OCEDStore()
    oced.add_object("order", "oid", pd.DataFrame({"oid": ["a 1", "a_1"]}))
    oced.add_event(
        "step",
        "eid",
        "ts",
        pd.DataFrame(
            {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
        ),
    )
    oced.add_event_to_object_relation(
        "touches",
        "eid",
        "oid",
        ["order"],
        pd.DataFrame({"eid": ["e1"], "oid": ["a_1"]}),
    )

    cleaned = clean_oced_data(oced)

    object_ids = set(cleaned.get_object_table("order").data["oid"])
    related = set(cleaned.get_relation("touches").data["oid"])
    assert related <= object_ids
    assert related == {"a_1_2"}


def test_a_bin_class_name_in_the_data_is_renamed_out_of_the_way():
    oced = OCEDStore()
    oced.add_object(
        "order",
        "oid",
        pd.DataFrame({"oid": ["price_bin_low", "o2"], "price": [1.0, 2.0]}),
    )

    cleaned = clean_oced_data(oced, bin_labels=["low", "high"])

    assert "price_bin_low" not in set(cleaned.get_object_table("order").data["oid"])


def test_bin_labels_are_only_reserved_when_they_are_given():
    oced = OCEDStore()
    oced.add_object(
        "order",
        "oid",
        pd.DataFrame({"oid": ["price_bin_low", "o2"], "price": [1.0, 2.0]}),
    )

    cleaned = clean_oced_data(oced)

    assert "price_bin_low" in set(cleaned.get_object_table("order").data["oid"])


def test_a_value_of_a_renamed_attribute_keeps_its_source_label():
    """`event` is reserved, so the column becomes `event_attr` and its values
    have to follow, or the backmap names an individual that does not exist."""
    oced = OCEDStore()
    oced.add_object("order", "oid", pd.DataFrame({"oid": ["o1"]}))
    oced.add_event(
        "step",
        "eid",
        "ts",
        pd.DataFrame(
            {
                "eid": ["e1"],
                "ts": pd.date_range("2022-01-01", periods=1, tz="UTC"),
                "event": ["by hand"],
            }
        ),
    )

    backmap = clean_oced_data(oced).backmap

    assert backmap["event_attr_by_hand"] == "by hand"
    assert "event_by_hand" not in backmap
