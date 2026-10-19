"""Qualifying an attribute name that several types carry with each type."""

import pandas as pd

from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind


def test_an_attribute_of_two_types_is_renamed_per_type_keeping_its_source_name(
    make_store, events
):
    oced = make_store(
        objects=pd.DataFrame({"oid": ["o1"], "unit weight": [2.5]}),
        obj_type="item",
        activity="ship order",
        events=events("e1", **{"unit weight": [3.0]}),
    )

    cleaned = clean_oced_data(oced)

    assert "item_unit_weight" in cleaned.table(EntityKind.OBJECT, "item").data
    assert (
        "ship_order_unit_weight" in cleaned.table(EntityKind.EVENT, "ship_order").data
    )
    assert cleaned.backmap["item_unit_weight"] == "unit weight"
    assert cleaned.backmap["ship_order_unit_weight"] == "unit weight"


def test_an_attribute_of_one_type_keeps_its_name(make_store, events):
    oced = make_store(
        objects=pd.DataFrame({"oid": ["o1"], "status": ["open"]}),
        events=events("e1", clerk=["ann"]),
    )

    cleaned = clean_oced_data(oced)

    assert "status" in cleaned.table(EntityKind.OBJECT, "obj").data
    assert "clerk" in cleaned.table(EntityKind.EVENT, "act").data


def test_a_changed_attribute_is_renamed_in_its_column_and_its_change_rows():
    oced = OCEDStore(ocel_mode=True)
    for obj_type in ("order", "item"):
        oced.add(
            EntityKind.OBJECT,
            obj_type,
            pd.DataFrame({"oid": [f"{obj_type}1"], "status": ["new"]}),
            id_col="oid",
        )
    oced.add_object_changes(
        "order",
        pd.DataFrame(
            {
                "ocel:oid": ["order1"],
                "ocel:type": ["order"],
                "ocel:timestamp": pd.to_datetime(["2022-01-02"], utc=True),
                "ocel:field": ["status"],
                "status": ["done"],
            }
        ),
    )

    changes = clean_oced_data(oced).get_object_changes("order")

    assert list(changes["ocel:field"]) == ["order_status"]
    assert list(changes["order_status"]) == ["done"]


def test_an_empty_column_of_a_change_table_does_not_make_an_attribute_shared():
    oced = OCEDStore(ocel_mode=True)
    oced.add(
        EntityKind.OBJECT,
        "order",
        pd.DataFrame({"oid": ["o1"], "status": ["new"]}),
        id_col="oid",
    )
    oced.add(EntityKind.OBJECT, "item", pd.DataFrame({"oid": ["i1"]}), id_col="oid")
    oced.add_object_changes(
        "item",
        pd.DataFrame(
            {
                "ocel:oid": ["i1"],
                "ocel:type": ["item"],
                "ocel:timestamp": pd.to_datetime(["2022-01-02"], utc=True),
                "ocel:field": ["size"],
                "size": ["L"],
                "status": [None],
            }
        ),
    )

    cleaned = clean_oced_data(oced)

    assert "status" in cleaned.table(EntityKind.OBJECT, "order").data
