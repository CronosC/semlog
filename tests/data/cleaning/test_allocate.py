"""Giving every schema name of a store one owner."""

import pandas as pd
import pytest

from semlog.data.cleaning import clean_oced_data
from semlog.data.cleaning.allocate import allocate_schema_names
from semlog.data.cleaning.renames import RenameLog
from semlog.data.oced import OCEDStore
from semlog.data.tables import EntityKind, RelationKind


def _relate(oced, name, to_type="obj"):
    oced.add_relation(
        RelationKind.E2O,
        name,
        pd.DataFrame({"eid": ["e1"], "oid": ["o1"]}),
        from_id="eid",
        to_id="oid",
        to_type=[to_type],
    )


def _allocate(oced):
    allocate_schema_names(oced, RenameLog(), ())
    return oced


def _columns(oced, obj_type="obj"):
    return list(oced.table(EntityKind.OBJECT, obj_type).data.columns)


@pytest.mark.parametrize("activity", ["observes", "Event", "df"])
def test_a_log_name_equal_to_a_schema_name_keeps_the_name(make_store, activity):
    cleaned = clean_oced_data(make_store(activity=activity))

    assert cleaned.types(EntityKind.EVENT) == [activity]
    assert activity not in cleaned.backmap.get("activity", {})


@pytest.mark.parametrize(
    ("activity", "objects"),
    [
        ("has_price", pd.DataFrame({"oid": ["o1"], "price": [10.0]})),
        ("df_obj", None),
    ],
    ids=["from_an_attribute", "from_an_object_type"],
)
def test_an_activity_named_after_a_derived_name_is_prefixed(
    make_store, activity, objects
):
    oced = make_store(activity=activity, objects=objects)

    assert _allocate(oced).types(EntityKind.EVENT) == [f"event_{activity}"]


@pytest.mark.parametrize("attribute", ["has_weight", "Observes"])
def test_a_name_that_only_resembles_a_schema_name_is_left_alone(make_store, attribute):
    """`has_weight` derives from no attribute of the log, and IRIs are case sensitive."""
    oced = make_store(objects=pd.DataFrame({"oid": ["o1"], attribute: ["x"]}))

    assert _columns(_allocate(oced)) == ["oid", attribute]


def test_an_object_type_keeps_a_name_an_activity_and_an_attribute_also_claim(
    make_store,
):
    """Attribute `x` is also the name of an object type and an activity."""
    oced = make_store(
        objects=pd.DataFrame({"oid": ["o1"], "x": ["v"]}),
        obj_type="order",
        activity="x",
    )
    _with_object_type(oced, "x")

    cleaned = clean_oced_data(oced)

    assert cleaned.types(EntityKind.OBJECT) == ["order", "x"]
    assert cleaned.types(EntityKind.EVENT) == ["event_x"]
    assert _columns(cleaned, "order") == ["oid", "attribute_x"]


def test_a_relation_named_after_an_object_type_is_prefixed(make_store):
    """A class and an object property cannot share a name in OWL 2 DL."""
    oced = make_store(obj_type="order")
    _relate(oced, "order", to_type="order")

    cleaned = clean_oced_data(oced)

    assert cleaned.relation_types() == ["relation_order"]
    assert cleaned.relation("relation_order").to_type == ["order"]


def test_a_relation_cannot_open_a_property_name_with_a_digit(make_store):
    """RDF/XML writes a property as an element, which cannot start with one."""
    oced = make_store()
    _relate(oced, "2modifies")

    assert clean_oced_data(oced).relation_types() == ["relation_2modifies"]


def _with_object_type(oced, name):
    oced.add(EntityKind.OBJECT, name, pd.DataFrame({"x": [f"{name}_1"]}), id_col="x")
    return oced


def test_a_prefixed_name_the_log_holds_is_numbered_past(make_store):
    """`order` derives `df_order`, which an object type claims first."""
    oced = _with_object_type(make_store(obj_type="order"), "df_order")
    _with_object_type(oced, "object_order")

    cleaned = clean_oced_data(oced)

    assert sorted(cleaned.types(EntityKind.OBJECT)) == [
        "df_order",
        "object_order",
        "object_order_2",
    ]
    assert cleaned.backmap["object_type"]["object_order_2"] == "order"


def test_a_rename_never_discards_a_type_already_in_the_store(make_store):
    oced = make_store(obj_type="order", activity="order")
    oced.add(
        EntityKind.EVENT,
        "event_order",
        pd.DataFrame({"eid": ["e9"], "ts": pd.to_datetime(["2022-01-01"], utc=True)}),
        id_col="eid",
        time_col="ts",
    )

    _allocate(oced)

    assert sorted(oced.types(EntityKind.EVENT)) == ["event_order", "event_order_2"]


def test_the_names_derived_from_a_renamed_type_are_claimed_under_its_new_name(
    make_store,
):
    """`object_o_x` derives `df_object_o_x`, which the log holds as an attribute.

    Its prefixed name derives `has_attribute_df_object_o_x`, which another attribute holds.
    """
    oced = _with_object_type(make_store(obj_type="o_x"), "df_o_x")
    oced.add(
        EntityKind.OBJECT,
        "t",
        pd.DataFrame(
            {
                "id": ["t1"],
                "df_object_o_x": ["v"],
                "has_df_object_o_x": ["v"],
                "has_attribute_df_object_o_x": ["v"],
            }
        ),
        id_col="id",
    )

    cleaned = clean_oced_data(oced)

    assert "object_o_x" in cleaned.types(EntityKind.OBJECT)
    assert _columns(cleaned, "t") == [
        "id",
        "attribute_df_object_o_x_2",
        "has_df_object_o_x",
        "has_attribute_df_object_o_x",
    ]


def _shuffled_store(order):
    oced = OCEDStore()
    for kind, name in order:
        frame = pd.DataFrame({"id": [f"{kind.value}_{name}"], "order": ["x"]})
        if kind is EntityKind.EVENT:
            frame["ts"] = pd.to_datetime(["2022-01-01"], utc=True)
        oced.add(
            kind,
            name,
            frame,
            id_col="id",
            time_col="ts" if kind is EntityKind.EVENT else None,
        )
    return oced


def test_the_names_do_not_depend_on_the_order_the_store_holds_things_in():
    parts = [
        (EntityKind.OBJECT, "order"),
        (EntityKind.EVENT, "order"),
        (EntityKind.OBJECT, "item"),
        (EntityKind.EVENT, "ship"),
    ]

    one = clean_oced_data(_shuffled_store(parts))
    other = clean_oced_data(_shuffled_store(parts[::-1]))

    assert one.backmap == other.backmap
    assert sorted(one.types(EntityKind.EVENT)) == sorted(other.types(EntityKind.EVENT))


def test_a_value_of_a_renamed_attribute_is_recorded_under_the_new_column(
    make_store, events
):
    """Attribute `obj` is the name of an object type, and is renamed."""
    oced = make_store(events=events("e1", obj=["by hand"]))

    backmap = clean_oced_data(oced).backmap["value"]

    assert backmap["attribute_obj.by_hand"] == "by hand"
    assert "obj.by_hand" not in backmap


def test_an_attribute_only_the_history_carries_is_renamed_in_the_change_rows(
    make_store,
):
    oced = _with_object_type(make_store(obj_type="order", ocel_mode=True), "status")
    oced.add_object_changes(
        "order",
        pd.DataFrame(
            {
                "ocel:oid": ["o1"],
                "ocel:type": ["order"],
                "ocel:timestamp": pd.to_datetime(["2024-01-02"], utc=True),
                "ocel:field": ["status"],
                "status": ["done"],
            }
        ),
    )

    changes = clean_oced_data(oced).object_changes("order")

    assert list(changes["ocel:field"]) == ["attribute_status"]
    assert list(changes["attribute_status"]) == ["done"]
