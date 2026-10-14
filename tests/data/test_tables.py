from unittest.mock import Mock

import pandas as pd
import pytest

from semanticlog.data.tables import (
    EntityKind,
    EntityTable,
    RelationKind,
    Relations,
    RelationTable,
    TableView,
)


@pytest.mark.parametrize(("time", "expected"), [("ts", {"eid", "ts"}), (None, {"eid"})])
def test_a_table_keys_on_its_id_and_any_time_column(time, expected):
    table = EntityTable(id="eid", data=pd.DataFrame(), time=time)

    assert table.key_columns == frozenset(expected)


def _view(**overrides) -> TableView:
    fields = dict(
        frame=pd.DataFrame(columns=["eid", "ts", "price", "channel"]),
        name="step",
        kind=EntityKind.EVENT,
        key_columns=frozenset({"eid", "ts"}),
        origin="activity 'step'",
        id_column="eid",
        replace=Mock(),
    )
    return TableView(**{**fields, **overrides})


def test_every_column_outside_the_keys_of_a_view_is_an_attribute():
    assert _view().attributes == ["price", "channel"]


def test_the_structural_columns_of_a_view_leave_out_its_id():
    assert _view().structural == ["ts"]


def test_new_relations_hold_both_families_empty():
    assert Relations().families == {RelationKind.E2O: {}, RelationKind.O2O: {}}


def test_relations_are_listed_event_to_object_first():
    holds = RelationTable("a", "b", ["B"], pd.DataFrame(), ["A"])
    touches = RelationTable("eid", "oid", ["A"], pd.DataFrame())
    relations = Relations({RelationKind.O2O: {"holds": holds}})
    relations.of(RelationKind.E2O)["touches"] = touches

    assert list(relations.items()) == [("touches", touches), ("holds", holds)]
