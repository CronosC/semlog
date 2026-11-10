"""Cleaning a whole store, and the source names it records."""

import pandas as pd
import pytest

from semanticlog.core import vocabulary as vocab
from semanticlog.core.errors import InputError
from semanticlog.core.vocabulary import is_safe_local_name
from semanticlog.data.cleaning import (
    clean_oced_data,
    object_type_renames,
    verify_unique_names,
)
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
        "activity": {"act_type_": "act type!"},
        "object_type": {"obj_type": "obj type"},
        "relation": {"rel_name": "rel name"},
        "attribute": {"total_cost": "total cost"},
        "id": {"e_1": "e 1"},
        "value": {"total_cost.2023-03-15_12_45": "2023-03-15 12:45"},
    }
    assert cleaned.relation("rel_name").data.values.tolist() == [["e_1", "o1"]]


def test_a_cleaned_store_holds_only_safe_names_and_passes_the_integrity_check(
    pipeline_raw_store,
):
    cleaned = clean_oced_data(pipeline_raw_store)
    names = [
        *cleaned.types(EntityKind.EVENT),
        *cleaned.types(EntityKind.OBJECT),
        *cleaned.relation_types(),
        *(column for view in cleaned.tables() for column in view.attributes),
    ]

    assert [name for name in names if not is_safe_local_name(name)] == []
    assert check_integrity(cleaned) is None


def test_a_change_table_is_keyed_by_the_cleaned_type_and_keeps_its_ocel_column_names():
    oced = OCEDStore(ocel_mode=True)
    oced.add(
        EntityKind.OBJECT, "order item", pd.DataFrame({"oid": ["o1"]}), id_col="oid"
    )
    oced.add_object_changes("order item", _changes("order item", field="sta tus"))

    cleaned = clean_oced_data(oced)

    changes = cleaned.object_changes("order_item")
    assert cleaned.object_change_types() == ["order_item"]
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
    """`a b` cleans to `a_b`, so the id `a_b` moves to `id_a_b`."""
    oced = make_store(
        objects=pd.DataFrame({"oid": ["a b", "a_b"], "status": ["new", "new"]}),
        obj_type="order",
        ocel_mode=True,
    )
    oced.add_relation(
        RelationKind.E2O,
        "modifies",
        pd.DataFrame({"eid": ["e1"], "oid": ["a_b"]}),
        from_id="eid",
        to_id="oid",
        to_type=["order"],
    )
    oced.add_object_changes("order", _changes("order", "a_b", status=["shipped"]))

    cleaned = clean_oced_data(oced)

    assert list(cleaned.ids(EntityKind.OBJECT, "order")) == ["a_b", "id_a_b"]
    assert list(cleaned.relation("modifies").data["oid"]) == ["id_a_b"]
    assert list(cleaned.object_changes("order")["ocel:oid"]) == ["id_a_b"]


def test_an_id_equal_to_an_activity_name_is_kept(make_store, events):
    cleaned = clean_oced_data(make_store(events=events("act", "e2")))

    assert list(cleaned.ids(EntityKind.EVENT, "act")) == ["act", "e2"]
    assert cleaned.backmap == {}


def test_a_value_equal_to_an_object_type_name_is_kept(make_store, events):
    cleaned = clean_oced_data(make_store(events=events("e1", channel=["obj"])))

    assert list(cleaned.table(EntityKind.EVENT, "act").data["channel"]) == ["obj"]
    assert "channel.obj" not in cleaned.backmap.get("value", {})


def test_an_attribute_of_two_types_keeps_one_name(make_store, events):
    oced = make_store(
        objects=pd.DataFrame({"oid": ["o1"], "status": ["new"]}),
        events=events("e1", status=["open"]),
    )

    cleaned = clean_oced_data(oced)

    assert list(cleaned.table(EntityKind.EVENT, "act").data.columns) == [
        "eid",
        "ts",
        "status",
    ]
    assert list(cleaned.table(EntityKind.OBJECT, "obj").data.columns) == [
        "oid",
        "status",
    ]
    assert cleaned.backmap == {}


def test_a_blank_id_is_rejected_before_cleaning_rewrites_it(make_store):
    oced = make_store(objects=pd.DataFrame({"oid": [" ", "x1"], "kind": ["a", "b"]}))

    with pytest.raises(
        InputError,
        match=r"no id: object type 'obj': 1 object\(s\) have no 'oid'\. Give every event and object an id",
    ):
        clean_oced_data(oced)


def test_an_empty_text_value_becomes_an_underscore_mapped_to_the_empty_string(
    make_store,
):
    oced = make_store(objects=pd.DataFrame({"oid": ["o1", "o2"], "flag": ["", "X"]}))

    cleaned = clean_oced_data(oced)

    assert list(cleaned.table(EntityKind.OBJECT, "obj").data["flag"]) == ["_", "X"]
    assert cleaned.backmap["value"]["flag._"] == ""


def test_only_a_moved_object_type_is_listed_with_its_new_name(make_store):
    """`order` derives `df_order`, which an object type claims first."""
    oced = make_store(obj_type="order", activity="place order")
    oced.add(EntityKind.OBJECT, "df_order", pd.DataFrame({"x": ["d1"]}), id_col="x")

    cleaned = clean_oced_data(oced)

    assert object_type_renames(cleaned) == {"order": "object_order"}


def test_a_dot_in_an_attribute_name_becomes_an_underscore_so_values_stay_apart(
    make_store, events
):
    """`a.b = c` and `a = b.c` named one value individual `a.b.c`."""
    cleaned = clean_oced_data(
        make_store(events=events("e1", **{"a.b": ["c"], "a": ["b.c"]}))
    )

    data = cleaned.table(EntityKind.EVENT, "act").data
    assert list(data.columns) == ["eid", "ts", "a_b", "a"]
    assert vocab.attribute_value("a_b", data["a_b"][0]) != vocab.attribute_value(
        "a", data["a"][0]
    )
    assert cleaned.backmap["attribute"]["a_b"] == "a.b"


def test_a_dotted_attribute_is_renamed_in_the_change_rows(make_store):
    oced = make_store(
        objects=pd.DataFrame({"oid": ["o1"], "unit.price": [1]}), ocel_mode=True
    )
    oced.add_object_changes(
        "obj", _changes("obj", field="unit.price", **{"unit.price": [2]})
    )

    changes = clean_oced_data(oced).object_changes("obj")

    assert list(changes["ocel:field"]) == ["unit_price"]
    assert "unit_price" in changes.columns


def test_the_source_names_are_kept_per_kind(make_store, events):
    oced = make_store(activity="pay order", events=events("pay_order"))

    cleaned = clean_oced_data(oced)

    assert cleaned.backmap == {"activity": {"pay_order": "pay order"}}


def test_a_name_claimed_twice_after_cleaning_is_a_programming_error():
    oced = OCEDStore()
    oced.add(EntityKind.OBJECT, "order", pd.DataFrame({"oid": ["o1"]}), id_col="oid")
    oced.add(
        EntityKind.EVENT,
        "order",
        pd.DataFrame({"eid": ["e1"], "t": pd.to_datetime(["2024-01-01"], utc=True)}),
        id_col="eid",
        time_col="t",
    )

    with pytest.raises(ValueError, match="'order' is claimed 2 times") as excinfo:
        verify_unique_names(oced)

    assert excinfo.type is ValueError


def test_names_claimed_twice_are_listed_on_one_line_up_to_five():
    oced = OCEDStore()
    for name in ("a", "b", "c", "d", "e", "f"):
        oced.add(
            EntityKind.OBJECT, name, pd.DataFrame({"oid": [f"o_{name}"]}), id_col="oid"
        )
        oced.add(
            EntityKind.EVENT,
            name,
            pd.DataFrame(
                {"eid": [f"e_{name}"], "t": pd.to_datetime(["2024-01-01"], utc=True)}
            ),
            id_col="eid",
            time_col="t",
        )

    with pytest.raises(ValueError, match="and 1 more") as excinfo:
        verify_unique_names(oced)

    assert "\n" not in str(excinfo.value)
    assert "'f' is claimed" not in str(excinfo.value)
