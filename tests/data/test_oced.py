"""The store: adding, reading and renaming tables and relations."""

import re

import pandas as pd
import pytest

from semanticlog.core.errors import InputError, SemlogError
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind, EntityTable, RelationKind, Relations

STAMPS = pd.date_range("2022-01-01", periods=3, tz="UTC")


def _events(*ids):
    return pd.DataFrame({"eid": list(ids), "ts": STAMPS[: len(ids)]})


def _changes(obj_type="A"):
    return pd.DataFrame(
        {
            "ocel:oid": ["a1"],
            "ocel:type": [obj_type],
            "ocel:timestamp": STAMPS[:1],
            "ocel:field": ["status"],
            "status": ["shipped"],
        }
    )


@pytest.fixture
def oced():
    """Two activities, object types A (three objects) and B (one), and one relation of each family."""
    store = OCEDStore(ocel_mode=True)
    store.add(
        EntityKind.EVENT, "ship", _events("e1", "e2"), id_col="eid", time_col="ts"
    )
    store.add(EntityKind.EVENT, "pay", _events("e3"), id_col="eid", time_col="ts")
    store.add(
        EntityKind.OBJECT, "A", pd.DataFrame({"oid": ["a1", "a2", "a3"]}), id_col="oid"
    )
    store.add(EntityKind.OBJECT, "B", pd.DataFrame({"oid": ["b1"]}), id_col="oid")
    store.add_relation(
        RelationKind.E2O,
        "touches",
        pd.DataFrame({"eid": ["e1", "e3"], "oid": ["a1", "b1"]}),
        from_id="eid",
        to_id="oid",
        to_type=["A", "B"],
    )
    store.add_relation(
        RelationKind.O2O,
        "holds",
        pd.DataFrame({"src": ["a1"], "dst": ["b1"]}),
        from_id="src",
        from_type=["A"],
        to_id="dst",
        to_type=["B"],
    )
    return store


def _add_relation(store, name, **kwargs):
    store.add_relation(
        kwargs.pop("kind", RelationKind.E2O),
        name,
        kwargs.pop("data", pd.DataFrame({"eid": ["e1"], "oid": ["a1"]})),
        from_id=kwargs.pop("from_id", "eid"),
        to_id="oid",
        to_type=kwargs.pop("to_type", ["A"]),
        **kwargs,
    )


# --------------------------------------------------------------------------
# Adding entities
# --------------------------------------------------------------------------


def test_the_store_holds_the_frames_it_is_given_without_copying():
    store = OCEDStore(ocel_mode=True)
    objects = pd.DataFrame({"oid": ["a1"]})
    relation = pd.DataFrame({"eid": ["e1"], "oid": ["a1"]})
    changes = _changes()
    store.add(EntityKind.OBJECT, "A", objects, id_col="oid")
    _add_relation(store, "touches", data=relation)
    store.add_object_changes("A", changes)

    assert store.table(EntityKind.OBJECT, "A").data is objects
    assert store.get_relation("touches").data is relation
    assert store.get_object_changes("A") is changes


@pytest.mark.parametrize(
    ("kind", "time_col", "message"),
    [
        (EntityKind.EVENT, None, "Activity 'x' needs a time_col"),
        (EntityKind.OBJECT, "ts", "Object type 'x' takes no time_col, got 'ts'"),
    ],
)
def test_only_an_event_table_takes_a_time_column(kind, time_col, message):
    store = OCEDStore()

    with pytest.raises(InputError, match=message):
        store.add(kind, "x", _events("e1"), id_col="eid", time_col=time_col)

    assert store.types(kind) == []


def test_a_table_without_the_columns_it_names_is_rejected_listing_its_columns():
    with pytest.raises(
        InputError,
        match=r"Activity 'x' names \['time'\], .* Its columns are \['eid', 'ts'\]",
    ):
        OCEDStore().add(
            EntityKind.EVENT, "x", _events("e1"), id_col="eid", time_col="time"
        )


def test_types_holding_ids_lists_each_type_once_sorted(oced):
    assert oced.types_holding(EntityKind.EVENT, ["e3", "e1", "e2", "x9"]) == [
        "pay",
        "ship",
    ]
    assert oced.types_holding(EntityKind.EVENT, ["x9"]) == []


def test_a_second_table_for_one_type_is_rejected(oced):
    with pytest.raises(InputError, match="Object type 'B' is already in this store"):
        oced.add(EntityKind.OBJECT, "B", pd.DataFrame({"oid": ["b9"]}), id_col="oid")


def test_an_id_another_table_holds_is_rejected_and_the_store_left_as_it_was(oced):
    with pytest.raises(InputError, match=r"Activity 'x' reuses 1 id\(s\) .* \['a1'\]"):
        oced.add(EntityKind.EVENT, "x", _events("a1"), id_col="eid", time_col="ts")

    assert oced.types(EntityKind.EVENT) == ["ship", "pay"]


def test_an_id_repeated_within_one_table_is_rejected():
    with pytest.raises(InputError, match=r"reuses 1 id\(s\) .* \['o1'\]"):
        OCEDStore().add(
            EntityKind.OBJECT, "A", pd.DataFrame({"oid": ["o1", "o1"]}), id_col="oid"
        )


def test_rows_with_no_id_are_accepted_for_the_integrity_checks_to_report():
    store = OCEDStore()

    store.add(EntityKind.OBJECT, "A", pd.DataFrame({"oid": [None, None]}), id_col="oid")

    assert store.types(EntityKind.OBJECT) == ["A"]


# --------------------------------------------------------------------------
# Reading entities
# --------------------------------------------------------------------------


def test_an_unknown_type_is_rejected_listing_what_the_store_holds(oced):
    kind, word = EntityKind.OBJECT, "object type"
    held = sorted(oced.types(kind))

    for call in (
        lambda: oced.table(kind, "nope"),
        lambda: oced.ids(kind, "nope"),
        lambda: oced.rename(kind, "nope", "x"),
    ):
        with pytest.raises(
            InputError,
            match=rf"{word} 'nope', which this store does not hold\. It holds {re.escape(str(held))}",
        ):
            call()


def test_a_kind_that_is_not_an_entity_kind_is_a_programming_error():
    with pytest.raises(ValueError, match="'bad' is not an EntityKind") as excinfo:
        OCEDStore().ids("bad", "x")

    assert not isinstance(excinfo.value, SemlogError)


def test_the_most_common_object_types_come_largest_first(oced):
    assert oced.get_most_common_objects(n=2) == ["A", "B"]


def test_a_negative_count_of_common_object_types_is_a_programming_error(oced):
    with pytest.raises(ValueError, match="needs n >= 0, got -1") as excinfo:
        oced.get_most_common_objects(n=-1)

    assert not isinstance(excinfo.value, SemlogError)


# --------------------------------------------------------------------------
# Renaming
# --------------------------------------------------------------------------


def test_renaming_an_object_type_renames_it_in_relations_and_its_change_table(oced):
    oced.add_object_changes("A", _changes())

    oced.rename(EntityKind.OBJECT, "A", "C")

    assert oced.types(EntityKind.OBJECT) == ["B", "C"]
    assert oced.get_relation("touches").to_type == ["C", "B"]
    assert oced.get_relation("holds").from_type == ["C"]
    assert oced.get_object_change_types() == ["C"]


def test_renaming_an_activity_changes_nothing_but_the_activity(oced):
    before = {
        name: (rel.from_type, list(rel.to_type)) for name, rel in oced.relations()
    }

    oced.rename(EntityKind.EVENT, "ship", "send")

    assert oced.types(EntityKind.EVENT) == ["pay", "send"]
    assert {
        name: (rel.from_type, list(rel.to_type)) for name, rel in oced.relations()
    } == before


def test_renaming_a_type_onto_a_taken_name_is_rejected(oced):
    with pytest.raises(InputError, match="'B' is already an object type"):
        oced.rename(EntityKind.OBJECT, "A", "B")


def test_renaming_a_relation_keeps_it_in_its_family(oced):
    oced.rename_relation("holds", "contains")

    assert oced.get_relation_types(RelationKind.O2O) == ["contains"]


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("touches", "holds", "'holds' already exists as a relation"),
        (
            "nope",
            "x",
            r"relation 'nope', which this store does not hold\. It holds \['holds', 'touches'\]",
        ),
    ],
)
def test_renaming_a_relation_onto_a_taken_name_or_from_an_unknown_one_is_rejected(
    oced, old, new, message
):
    with pytest.raises(InputError, match=message):
        oced.rename_relation(old, new)


# --------------------------------------------------------------------------
# Relations
# --------------------------------------------------------------------------


def test_relations_are_listed_by_family_event_to_object_first(oced):
    assert oced.get_relation_types() == ["touches", "holds"]
    assert list(oced.relations_of(RelationKind.E2O)) == ["touches"]
    assert oced.get_relation_types(RelationKind.O2O) == ["holds"]


def test_adding_a_relation_to_both_families_at_once_is_a_programming_error(oced):
    with pytest.raises(ValueError, match="is not a family to add to") as excinfo:
        _add_relation(oced, "x", kind=RelationKind.ALL)

    assert not isinstance(excinfo.value, SemlogError)


@pytest.mark.parametrize(
    ("kind", "from_type", "message"),
    [
        (
            RelationKind.E2O,
            ["A"],
            "Relation 'x' is event-to-object and takes no from_type",
        ),
        (
            RelationKind.O2O,
            None,
            "Relation 'x' is object-to-object and needs a from_type",
        ),
    ],
)
def test_only_an_object_to_object_relation_names_its_source_types(
    oced, kind, from_type, message
):
    with pytest.raises(InputError, match=message):
        _add_relation(oced, "x", kind=kind, from_type=from_type)


@pytest.mark.parametrize(
    ("name", "kwargs", "message"),
    [
        (
            "holds",
            {},
            "'holds' already exists in the object-to-object family. Give it another name",
        ),
        (
            "touches",
            {},
            "'touches' already exists in the event-to-object family. Give it another name",
        ),
        ("x", {"to_type": ["A", "Z"]}, r"names 1 unknown object type\(s\): \['Z'\]"),
        (
            "x",
            {"from_id": "event"},
            r"names from_id 'event', which the given data does not have",
        ),
    ],
    ids=["other_family", "same_family", "unknown_type", "missing_column"],
)
def test_a_relation_the_store_cannot_hold_is_rejected(oced, name, kwargs, message):
    with pytest.raises(InputError, match=message):
        _add_relation(oced, name, **kwargs)


def test_an_unknown_relation_is_rejected_listing_both_families(oced):
    with pytest.raises(
        InputError,
        match=r"Unknown relation 'nope'\. This store holds \['touches', 'holds'\]",
    ):
        oced.get_relation("nope")


# --------------------------------------------------------------------------
# Attribute change history
# --------------------------------------------------------------------------


def test_change_history_outside_ocel_mode_is_a_programming_error():
    with pytest.raises(
        ValueError, match=r"add_object_changes requires ocel_mode=True"
    ) as excinfo:
        OCEDStore().add_object_changes("A", _changes())

    assert not isinstance(excinfo.value, SemlogError)


def test_changes_of_a_type_the_store_does_not_hold_are_rejected(oced):
    with pytest.raises(
        InputError, match="object type 'Z', which this store does not hold"
    ):
        oced.add_object_changes("Z", _changes("Z"))


def test_changes_without_the_ocel_columns_are_rejected_naming_them(oced):
    with pytest.raises(
        InputError,
        match=r"missing the required column\(s\) \['ocel:field', 'ocel:timestamp'\]",
    ):
        oced.add_object_changes(
            "A", _changes().drop(columns=["ocel:field", "ocel:timestamp"])
        )


def test_reading_the_changes_of_a_type_with_none_recorded_is_rejected(oced):
    with pytest.raises(InputError, match="No object changes recorded for type 'B'"):
        oced.get_object_changes("B")


# --------------------------------------------------------------------------
# Replacing every table
# --------------------------------------------------------------------------


def test_replacing_the_tables_recounts_the_ids_in_use(oced):
    entities = {
        EntityKind.OBJECT: {
            "A": EntityTable(id="oid", data=pd.DataFrame({"oid": ["n1"]}))
        }
    }

    oced.replace_tables(entities, Relations())
    oced.add(EntityKind.OBJECT, "B", pd.DataFrame({"oid": ["a1"]}), id_col="oid")

    with pytest.raises(InputError, match="reuses 1 id"):
        oced.add(EntityKind.OBJECT, "C", pd.DataFrame({"oid": ["n1"]}), id_col="oid")


def test_replacing_the_tables_with_a_repeated_id_is_rejected():
    entities = {
        EntityKind.OBJECT: {
            "A": EntityTable(id="oid", data=pd.DataFrame({"oid": ["x"]})),
            "B": EntityTable(id="oid", data=pd.DataFrame({"oid": ["x"]})),
        }
    }

    with pytest.raises(
        InputError, match=r"1 id\(s\) are used more than once, such as \['x'\]"
    ):
        OCEDStore().replace_tables(entities, Relations())
