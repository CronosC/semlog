"""Scoping every attribute name to the type that carries it."""

import pandas as pd
import pytest

from semanticlog.core.errors import InputError
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

    assert "item.unit_weight" in cleaned.table(EntityKind.OBJECT, "item").data
    assert (
        "ship_order.unit_weight" in cleaned.table(EntityKind.EVENT, "ship_order").data
    )
    assert cleaned.backmap["item.unit_weight"] == "unit weight"
    assert cleaned.backmap["ship_order.unit_weight"] == "unit weight"


def test_an_attribute_of_one_type_is_renamed_under_its_type(make_store, events):
    oced = make_store(
        objects=pd.DataFrame({"oid": ["o1"], "status": ["open"]}),
        events=events("e1", clerk=["ann"]),
    )

    cleaned = clean_oced_data(oced)

    assert list(cleaned.table(EntityKind.OBJECT, "obj").data.columns) == [
        "oid",
        "obj.status",
    ]
    assert list(cleaned.table(EntityKind.EVENT, "act").data.columns) == [
        "eid",
        "ts",
        "act.clerk",
    ]
    assert cleaned.backmap["obj.status"] == "status"
    assert cleaned.backmap["act.clerk"] == "clerk"


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

    assert list(changes["ocel:field"]) == ["order.status"]
    assert list(changes["order.status"]) == ["done"]


def test_an_empty_column_of_a_change_table_is_not_renamed():
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

    changes = clean_oced_data(oced).get_object_changes("item")

    assert "status" in changes
    assert "item.status" not in changes
    assert "item.size" in changes


def test_two_pairs_of_type_and_attribute_giving_one_name_are_rejected(
    make_store, events
):
    oced = make_store(
        objects=pd.DataFrame({"oid": ["o1"], "c": [1]}),
        obj_type="a.b",
        activity="a",
        events=events("e1", **{"b.c": [3]}),
    )

    with pytest.raises(InputError, match="both become 'a.b.c'"):
        clean_oced_data(oced)
