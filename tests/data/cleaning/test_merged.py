"""Keeping apart the source names that cleaning would merge into one."""

import pandas as pd
import pytest

from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.cleaning.merged import merged_names
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind, RelationKind


@pytest.mark.parametrize(
    ("names", "fixed", "expected"),
    [
        (["a_b", "a b", "c"], (), {"a_b": "x_a_b"}),
        (["a_b", "a b", "c"], ("a_b",), {"a b": "x_a_b"}),
        (["a_b", "a_b"], (), {}),
    ],
    ids=["sorted_first_keeps", "fixed_keeps", "repeat_is_no_collision"],
)
def test_of_names_cleaning_alike_one_keeps_its_name_and_the_rest_move(
    names, fixed, expected
):
    assert merged_names(names, "x_", fixed) == expected


def test_two_activities_cleaning_alike_are_kept_apart(make_store, events):
    oced = make_store(activity="Place Order")
    oced.add(EntityKind.EVENT, "Place_Order", events("e2"), id_col="eid", time_col="ts")

    cleaned = clean_oced_data(oced)

    assert cleaned.types(EntityKind.EVENT) == ["Place_Order", "event_Place_Order"]
    assert cleaned.backmap["Place_Order"] == "Place Order"
    assert cleaned.backmap["event_Place_Order"] == "Place_Order"


def test_two_object_types_cleaning_alike_are_kept_apart(make_store):
    oced = make_store(obj_type="Order Item")
    oced.add(EntityKind.OBJECT, "Order_Item", pd.DataFrame({"x": ["o2"]}), id_col="x")

    cleaned = clean_oced_data(oced)

    assert sorted(cleaned.types(EntityKind.OBJECT)) == [
        "Order_Item",
        "object_Order_Item",
    ]


def test_two_relations_cleaning_alike_are_kept_apart_across_both_families(
    make_store,
):
    oced = make_store(objects=pd.DataFrame({"oid": ["o1", "o2"]}))
    oced.add_relation(
        RelationKind.E2O,
        "contains x",
        pd.DataFrame({"eid": ["e1"], "oid": ["o1"]}),
        from_id="eid",
        to_id="oid",
        to_type=["obj"],
    )
    oced.add_relation(
        RelationKind.O2O,
        "contains_x",
        pd.DataFrame({"src": ["o1"], "dst": ["o2"]}),
        from_id="src",
        from_type=["obj"],
        to_id="dst",
        to_type=["obj"],
    )

    cleaned = clean_oced_data(oced)

    assert cleaned.get_relation_types() == ["contains_x", "relation_contains_x"]
    assert cleaned.backmap["relation_contains_x"] == "contains_x"


def test_an_attribute_cleaning_to_the_id_column_name_moves_and_the_id_keeps_it():
    oced = OCEDStore()
    oced.add(
        EntityKind.OBJECT,
        "order",
        pd.DataFrame({"o id": ["o1"], "o_id": ["x"], "a b": [1], "a_b": [2]}),
        id_col="o id",
    )

    cleaned = clean_oced_data(oced)

    table = cleaned.table(EntityKind.OBJECT, "order")
    assert table.id == "o_id"
    assert list(table.data.columns) == [
        "o_id",
        "order.attribute_o_id",
        "order.a_b",
        "order.attribute_a_b",
    ]
    assert cleaned.backmap["order.attribute_a_b"] == "a_b"


def test_two_ids_cleaning_alike_are_kept_apart_across_the_store(make_store, events):
    oced = make_store(objects=pd.DataFrame({"oid": ["a_1"]}), events=events("a 1"))

    cleaned = clean_oced_data(oced)

    assert list(cleaned.ids(EntityKind.EVENT, "act")) == ["a_1"]
    assert list(cleaned.ids(EntityKind.OBJECT, "obj")) == ["id_a_1"]


def test_values_of_one_attribute_cleaning_alike_are_kept_apart_across_tables():
    oced = OCEDStore(ocel_mode=True)
    oced.add(
        EntityKind.OBJECT,
        "order",
        pd.DataFrame({"oid": ["o1"], "status": ["a b"]}),
        id_col="oid",
    )
    oced.add_object_changes(
        "order",
        pd.DataFrame(
            {
                "ocel:oid": ["o1"],
                "ocel:type": ["order"],
                "ocel:timestamp": pd.to_datetime(["2024-01-02"], utc=True),
                "ocel:field": ["status"],
                "status": ["a_b"],
            }
        ),
    )

    cleaned = clean_oced_data(oced)

    assert list(cleaned.table(EntityKind.OBJECT, "order").data["order.status"]) == [
        "a_b"
    ]
    assert list(cleaned.get_object_changes("order")["order.status"]) == ["value_a_b"]
    assert cleaned.backmap["order.status.a_b"] == "a b"
    assert cleaned.backmap["order.status.value_a_b"] == "a_b"


def test_one_value_under_two_attributes_is_not_a_collision(make_store, events):
    oced = make_store(
        objects=pd.DataFrame({"oid": ["o1"], "channel": ["a b"]}),
        events=events("e1", clerk=["a b"]),
    )

    cleaned = clean_oced_data(oced)

    assert cleaned.backmap == {
        "act.clerk": "clerk",
        "obj.channel": "channel",
        "act.clerk.a_b": "a b",
        "obj.channel.a_b": "a b",
    }


def test_values_of_one_attribute_cleaning_alike_on_two_types_keep_their_names(
    make_store, events
):
    oced = make_store(
        objects=pd.DataFrame({"oid": ["o1"], "status": ["a_b"]}),
        events=events("e1", status=["a b"]),
    )

    cleaned = clean_oced_data(oced)

    assert list(cleaned.table(EntityKind.EVENT, "act").data["act.status"]) == ["a_b"]
    assert list(cleaned.table(EntityKind.OBJECT, "obj").data["obj.status"]) == ["a_b"]
    assert cleaned.backmap == {
        "act.status": "status",
        "obj.status": "status",
        "act.status.a_b": "a b",
    }


def test_a_cleaned_value_has_a_source_name_only_on_the_type_that_holds_it(
    make_store, events
):
    oced = make_store(
        objects=pd.DataFrame({"oid": ["o1"], "status": ["a b"]}),
        events=events("e1", status=["c"]),
    )

    cleaned = clean_oced_data(oced)

    assert "act.status.a_b" not in cleaned.backmap
    assert cleaned.backmap["obj.status.a_b"] == "a b"
