"""Cleaning a whole store, and the source names it records."""

import logging

import pandas as pd
import pytest

from semanticlog.core.errors import InputError
from semanticlog.core.vocabulary import is_safe_local_name
from semanticlog.data.cleaning import clean_oced_data, object_type_renames
from semanticlog.data.integrity import check_integrity
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind, RelationKind


def _changes(obj_type, oid="o1", field="status", **attributes):
    return pd.DataFrame(
        {
            "ocel:oid": [oid],
            "ocel:type": [obj_type],
            "ocel:timestamp": pd.to_datetime(["2022-01-02"], utc=True),
            "ocel:field": [field],
            **attributes,
        }
    )


def test_every_name_in_the_store_is_made_safe_and_mapped_to_its_source():
    oced = OCEDStore()
    oced.add(
        EntityKind.EVENT,
        "act type!",
        pd.DataFrame(
            {
                "eid": ["e 1"],
                "ts": pd.to_datetime(["2022-01-01"], utc=True),
                "total cost": ["2023-03-15 12:45"],
            }
        ),
        id_col="eid",
        time_col="ts",
    )
    oced.add(EntityKind.OBJECT, "obj type", pd.DataFrame({"oid": ["o1"]}), id_col="oid")
    oced.add_relation(
        RelationKind.E2O,
        "rel name",
        pd.DataFrame({"eid": ["e 1"], "oid": ["o1"]}),
        from_id="eid",
        to_id="oid",
        to_type=["obj type"],
    )

    cleaned = clean_oced_data(oced)

    assert cleaned.backmap == {
        "act_type_": "act type!",
        "obj_type": "obj type",
        "rel_name": "rel name",
        "total_cost": "total cost",
        "e_1": "e 1",
        "total_cost_2023-03-15_12_45": "2023-03-15 12:45",
    }
    assert cleaned.get_relation("rel_name").data.values.tolist() == [["e_1", "o1"]]


def test_a_cleaned_store_holds_only_safe_names_and_passes_the_integrity_check(
    pipeline_raw_store,
):
    cleaned = clean_oced_data(pipeline_raw_store)
    names = [
        *cleaned.types(EntityKind.EVENT),
        *cleaned.types(EntityKind.OBJECT),
        *cleaned.get_relation_types(),
        *(column for view in cleaned.tables() for column in view.attributes),
    ]

    assert [name for name in names if not is_safe_local_name(name)] == []
    assert check_integrity(cleaned) is None


def test_two_spellings_of_one_attribute_across_tables_are_read_as_one_with_a_warning(
    make_store, events, caplog
):
    oced = make_store(
        objects=pd.DataFrame({"oid": ["o1"], "Total_Cost": [1.0]}),
        obj_type="order",
        events=events("e1", **{"Total Cost": [2.0]}),
    )

    with caplog.at_level(logging.WARNING, logger="semanticlog.data.cleaning"):
        cleaned = clean_oced_data(oced)

    assert (
        "Attribute 'Total_Cost' is spelled 'Total Cost' in activity 'act', "
        "'Total_Cost' in object type 'order'. The spellings are read as one attribute."
    ) in caplog.text
    assert "Total_Cost" in cleaned.table(EntityKind.EVENT, "act").data.columns
    assert "Total_Cost" not in cleaned.backmap


def test_a_change_table_is_keyed_by_the_cleaned_type_and_keeps_its_ocel_column_names():
    oced = OCEDStore(ocel_mode=True)
    oced.add(
        EntityKind.OBJECT, "order item", pd.DataFrame({"oid": ["o1"]}), id_col="oid"
    )
    oced.add_object_changes("order item", _changes("order item", field="sta tus"))

    cleaned = clean_oced_data(oced)

    changes = cleaned.get_object_changes("order_item")
    assert cleaned.get_object_change_types() == ["order_item"]
    assert list(changes.columns) == [
        "ocel:oid",
        "ocel:type",
        "ocel:timestamp",
        "ocel:field",
    ]
    assert changes[["ocel:type", "ocel:field"]].values.tolist() == [
        ["order_item", "sta_tus"]
    ]


def test_a_renamed_id_is_rewritten_in_relations_and_change_rows(make_store):
    oced = make_store(
        objects=pd.DataFrame({"oid": ["Event"], "status": ["new"]}),
        obj_type="order",
        ocel_mode=True,
    )
    oced.add_relation(
        RelationKind.E2O,
        "modifies",
        pd.DataFrame({"eid": ["e1"], "oid": ["Event"]}),
        from_id="eid",
        to_id="oid",
        to_type=["order"],
    )
    oced.add_object_changes("order", _changes("order", "Event", status=["shipped"]))

    cleaned = clean_oced_data(oced)

    assert list(cleaned.ids(EntityKind.OBJECT, "order")) == ["id_Event"]
    assert list(cleaned.get_relation("modifies").data["oid"]) == ["id_Event"]
    assert list(cleaned.get_object_changes("order")["ocel:oid"]) == ["id_Event"]


def test_a_blank_id_is_rejected_before_cleaning_rewrites_it(make_store):
    oced = make_store(objects=pd.DataFrame({"oid": [" ", "x1"], "kind": ["a", "b"]}))

    with pytest.raises(
        InputError,
        match=r"no id:\n  object type 'obj': 1 object\(s\) have no 'oid'\n\nGive every event and object an id",
    ):
        clean_oced_data(oced)


def test_only_a_moved_object_type_is_listed_with_its_new_name(make_store):
    cleaned = clean_oced_data(make_store(obj_type="Event", activity="place order"))

    assert object_type_renames(cleaned) == {"Event": "object_Event"}
