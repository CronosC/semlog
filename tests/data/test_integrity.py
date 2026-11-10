"""Finding rows a build could not emit an individual for."""

import pandas as pd
import pytest

from semlog.core.checks import raise_reports
from semlog.core.errors import InputError
from semlog.data.integrity import check_integrity, check_missing_ids
from semlog.data.oced import OCEDStore
from semlog.data.tables import EntityKind, RelationKind


@pytest.fixture
def store():
    """One activity, object types A and B, and one relation of each family, all consistent."""
    oced = OCEDStore()
    oced.add(
        EntityKind.EVENT,
        "act",
        pd.DataFrame(
            {"eid": ["e1", "e2"], "ts": pd.date_range("2024-01-01", periods=2)}
        ),
        id_col="eid",
        time_col="ts",
    )
    oced.add(EntityKind.OBJECT, "A", pd.DataFrame({"oid": ["a1", "a2"]}), id_col="oid")
    oced.add(EntityKind.OBJECT, "B", pd.DataFrame({"oid": ["b1"]}), id_col="oid")
    oced.add_relation(
        RelationKind.E2O,
        "touches",
        pd.DataFrame({"eid": ["e1", "e2"], "oid": ["a1", "b1"]}),
        from_id="eid",
        to_id="oid",
        to_type=["A", "B"],
    )
    oced.add_relation(
        RelationKind.O2O,
        "holds",
        pd.DataFrame({"src": ["a1"], "dst": ["b1"]}),
        from_id="src",
        from_type=["A"],
        to_id="dst",
        to_type=["B"],
    )
    return oced


def _problems(store):
    report = check_integrity(store)
    return None if report is None else list(report.problems)


def test_a_consistent_store_passes_both_checks(store):
    assert (check_integrity(store), check_missing_ids(store)) == (None, None)


def test_rows_with_a_missing_or_blank_id_are_counted_per_table(store):
    store.add(
        EntityKind.OBJECT,
        "C",
        pd.DataFrame({"oid": [None, "", "  ", "c1"], "kind": ["a", "b", "c", "d"]}),
        id_col="oid",
    )

    assert list(check_missing_ids(store).problems) == [
        "object type 'C': 3 object(s) have no 'oid'"
    ]


def test_a_relation_row_with_an_empty_endpoint_is_reported(store):
    store.add_relation(
        RelationKind.E2O,
        "loose",
        pd.DataFrame({"eid": ["e1", None], "oid": [None, "a1"]}),
        from_id="eid",
        to_id="oid",
        to_type=["A"],
    )

    assert _problems(store) == [
        "e2o 'loose': 1 row(s) have no 'eid'",
        "e2o 'loose': 1 row(s) have no 'oid'",
    ]


def test_an_e2o_row_naming_an_unknown_event_or_an_object_of_another_type_is_reported(
    store,
):
    store.add_relation(
        RelationKind.E2O,
        "stray",
        pd.DataFrame({"eid": ["ghost", "e1"], "oid": ["a1", "b1"]}),
        from_id="eid",
        to_id="oid",
        to_type=["A"],
    )

    assert _problems(store) == [
        "e2o 'stray': 1 event id(s) are in no event table (ghost)",
        "e2o 'stray' -> ['A']: 1 object id(s) are in none of those types (b1)",
    ]


def test_an_o2o_row_naming_unknown_objects_is_reported_on_each_side(store):
    store.add_relation(
        RelationKind.O2O,
        "stray",
        pd.DataFrame({"src": ["b1"], "dst": ["x0"]}),
        from_id="src",
        from_type=["A"],
        to_id="dst",
        to_type=["B"],
    )

    assert _problems(store) == [
        "o2o 'stray' from -> ['A']: 1 id(s) are in none of those types (b1)",
        "o2o 'stray' to -> ['B']: 1 id(s) are in none of those types (x0)",
    ]


def test_a_long_list_of_unknown_ids_shows_five_and_counts_the_rest(store):
    store.add_relation(
        RelationKind.E2O,
        "stray",
        pd.DataFrame({"eid": ["e1"] * 7, "oid": [f"x{i}" for i in range(7)]}),
        from_id="eid",
        to_id="oid",
        to_type=["A"],
    )

    assert _problems(store) == [
        "e2o 'stray' -> ['A']: 7 object id(s) are in none of those types "
        "(x0, x1, x2, x3, x4 and 2 more)"
    ]


def test_a_store_with_violations_is_rejected_listing_them_with_advice(store):
    store.add_relation(
        RelationKind.E2O,
        "stray",
        pd.DataFrame({"eid": ["ghost"], "oid": ["a1"]}),
        from_id="eid",
        to_id="oid",
        to_type=["A"],
    )

    with pytest.raises(
        InputError,
        match=r"could not emit an individual for: e2o 'stray': 1 event id[^\n]*\. Fix the source data",
    ):
        raise_reports([check_integrity(store)])
