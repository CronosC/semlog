"""Writing one rename into every table that holds the name."""

import pandas as pd
import pytest

from semanticlog.data.cleaning.rewrite import (
    apply_attribute_renames,
    apply_id_renames,
    apply_value_renames,
)
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind, RelationKind


@pytest.fixture
def store():
    """An event and order o1 carrying a status, box 7 with a numeric id, o1 in box 7, and the status history of o1."""
    oced = OCEDStore(ocel_mode=True)
    oced.add(
        EntityKind.EVENT,
        "act",
        pd.DataFrame(
            {
                "eid": ["e1"],
                "ts": pd.to_datetime(["2024-01-01"], utc=True),
                "status": ["new"],
            }
        ),
        id_col="eid",
        time_col="ts",
    )
    oced.add(
        EntityKind.OBJECT,
        "order",
        pd.DataFrame({"oid": ["o1"], "status": ["new"]}),
        id_col="oid",
    )
    oced.add(EntityKind.OBJECT, "box", pd.DataFrame({"bid": [7]}), id_col="bid")
    oced.add_relation(
        RelationKind.O2O,
        "in",
        pd.DataFrame({"src": ["o1"], "dst": [7]}),
        from_id="src",
        from_type=["order"],
        to_id="dst",
        to_type=["box"],
    )
    oced.add_object_changes(
        "order",
        pd.DataFrame(
            {
                "ocel:oid": ["o1"],
                "ocel:type": ["order"],
                "ocel:timestamp": pd.to_datetime(["2024-01-02"], utc=True),
                "ocel:field": ["status"],
                "status": ["new"],
            }
        ),
    )
    return oced


def test_an_id_rename_reaches_id_columns_relation_endpoints_and_change_rows(store):
    apply_id_renames(store, {"o1": "id_o1"})

    assert list(store.ids(EntityKind.OBJECT, "order")) == ["id_o1"]
    assert list(store.relation("in").data["src"]) == ["id_o1"]
    assert list(store.object_changes("order")["ocel:oid"]) == ["id_o1"]


def test_an_id_rename_turns_an_id_column_that_is_not_text_into_text(store):
    """A relation endpoint then compares equal to the table it names."""
    apply_id_renames(store, {"o1": "id_o1"})

    assert list(store.ids(EntityKind.OBJECT, "box")) == ["7"]
    assert list(store.relation("in").data["dst"]) == ["7"]


def test_a_value_rename_reaches_every_table_carrying_the_attribute(store):
    apply_value_renames(store, "status", {"new": "value_new"})

    assert [
        list(view.frame["status"]) for view in store.tables() if "status" in view.frame
    ] == [["value_new"]] * 3


def test_an_attribute_rename_reaches_every_table_and_the_field_of_change_rows(store):
    apply_attribute_renames(store, {"status": "attribute_status"})
    changes = store.object_changes("order")

    assert "attribute_status" in store.table(EntityKind.EVENT, "act").data.columns
    assert "attribute_status" in store.table(EntityKind.OBJECT, "order").data.columns
    assert list(changes["ocel:field"]) == ["attribute_status"]
    assert list(changes["attribute_status"]) == ["new"]
