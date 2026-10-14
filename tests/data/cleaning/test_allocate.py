"""Giving every schema name of a store one owner."""

import pandas as pd
import pytest

from semanticlog.data.cleaning import clean_oced_data, fix_duplicate_names
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind, RelationKind


def _relate(oced, name, to_type="obj"):
    oced.add_relation(
        RelationKind.E2O,
        name,
        pd.DataFrame({"eid": ["e1"], "oid": ["o1"]}),
        from_id="eid",
        to_id="oid",
        to_type=[to_type],
    )


def _columns(oced, obj_type="obj"):
    return list(oced.table(EntityKind.OBJECT, obj_type).data.columns)


def test_an_activity_named_after_a_schema_name_takes_the_event_prefix(make_store):
    cleaned = clean_oced_data(make_store(activity="observes"))

    assert cleaned.types(EntityKind.EVENT) == ["event_observes"]
    assert cleaned.backmap["event_observes"] == "observes"


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

    assert fix_duplicate_names(oced).types(EntityKind.EVENT) == [f"event_{activity}"]


@pytest.mark.parametrize("attribute", ["has_weight", "Observes"])
def test_a_name_that_only_resembles_a_schema_name_is_left_alone(make_store, attribute):
    """`has_weight` derives from no attribute of the log, and IRIs are case sensitive."""
    oced = make_store(objects=pd.DataFrame({"oid": ["o1"], attribute: ["x"]}))

    assert _columns(fix_duplicate_names(oced)) == ["oid", attribute]


def test_an_object_type_keeps_a_name_an_activity_and_an_attribute_also_claim(
    make_store,
):
    oced = make_store(
        objects=pd.DataFrame({"oid": ["o1"], "order": ["x"]}),
        obj_type="order",
        activity="order",
    )

    cleaned = clean_oced_data(oced)

    assert cleaned.types(EntityKind.OBJECT) == ["order"]
    assert cleaned.types(EntityKind.EVENT) == ["event_order"]
    assert _columns(cleaned, "order") == ["oid", "attribute_order"]


def test_a_relation_named_after_an_object_type_is_prefixed(make_store):
    """A class and an object property cannot share a name in OWL 2 DL."""
    oced = make_store(obj_type="order")
    _relate(oced, "order", to_type="order")

    cleaned = clean_oced_data(oced)

    assert cleaned.get_relation_types() == ["relation_order"]
    assert cleaned.get_relation("relation_order").to_type == ["order"]


def test_a_relation_cannot_open_a_property_name_with_a_digit(make_store):
    """RDF/XML writes a property as an element, which cannot start with one."""
    oced = make_store()
    _relate(oced, "2modifies")

    assert clean_oced_data(oced).get_relation_types() == ["relation_2modifies"]


def test_a_prefixed_name_the_log_holds_is_numbered_past(make_store):
    oced = make_store(obj_type="Event")
    oced.add(EntityKind.OBJECT, "object_Event", pd.DataFrame({"x": ["o2"]}), id_col="x")

    cleaned = clean_oced_data(oced)

    assert sorted(cleaned.types(EntityKind.OBJECT)) == [
        "object_Event",
        "object_Event_2",
    ]
    assert cleaned.backmap["object_Event_2"] == "Event"


def test_a_rename_never_discards_a_type_already_in_the_store(make_store):
    oced = make_store(obj_type="Event")
    oced.add(EntityKind.OBJECT, "Event_", pd.DataFrame({"x": ["b1"]}), id_col="x")

    fix_duplicate_names(oced)

    assert sorted(oced.types(EntityKind.OBJECT)) == ["Event_", "object_Event"]


def test_the_names_derived_from_a_renamed_type_are_claimed_under_its_new_name(
    make_store,
):
    """`object_Event` derives `df_object_Event`, which the log holds as an attribute.

    Its prefixed name derives `has_attribute_df_object_Event`, which another attribute holds.
    """
    oced = make_store(
        objects=pd.DataFrame(
            {
                "oid": ["o1"],
                "df_object_Event": ["a"],
                "has_df_object_Event": ["b"],
                "has_attribute_df_object_Event": ["c"],
            }
        ),
        obj_type="Event",
    )

    cleaned = clean_oced_data(oced)

    assert _columns(cleaned, "object_Event") == [
        "oid",
        "attribute_df_object_Event_2",
        "has_df_object_Event",
        "has_attribute_df_object_Event",
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
    oced = make_store(events=events("e1", observed_at=["by hand"]))

    backmap = clean_oced_data(oced).backmap

    assert backmap["attribute_observed_at_by_hand"] == "by hand"
    assert "observed_at_by_hand" not in backmap


def test_an_attribute_only_the_history_carries_is_renamed_in_the_change_rows(
    make_store,
):
    oced = make_store(obj_type="order", ocel_mode=True)
    oced.add_object_changes(
        "order",
        pd.DataFrame(
            {
                "ocel:oid": ["o1"],
                "ocel:type": ["order"],
                "ocel:timestamp": pd.to_datetime(["2024-01-02"], utc=True),
                "ocel:field": ["order"],
                "order": ["done"],
            }
        ),
    )

    changes = clean_oced_data(oced).get_object_changes("order")

    assert list(changes["ocel:field"]) == ["attribute_order"]
    assert list(changes["attribute_order"]) == ["done"]
