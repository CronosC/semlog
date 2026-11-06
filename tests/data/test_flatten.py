"""Projecting a store onto one case concept, and choosing the case concepts."""

import logging

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from semanticlog.core.errors import InputError
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.flatten import (
    case_concept_candidates,
    choose_case_concepts,
    get_log,
    iter_logs,
)
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind, RelationKind

LOGGER = "semanticlog.data.flatten"


def _store(*activities, relations=()):
    """Return a store with object types `order` (o1, o2) and `item` (i1), and the given activities and relations.

    Args:
        activities: Each as name, event ids and timestamps.
        relations: Each as name, event ids, object ids and target types.
    """
    store = OCEDStore()
    store.add(
        EntityKind.OBJECT, "order", pd.DataFrame({"oid": ["o1", "o2"]}), id_col="oid"
    )
    store.add(EntityKind.OBJECT, "item", pd.DataFrame({"oid": ["i1"]}), id_col="oid")
    for name, ids, stamps in activities:
        store.add(
            EntityKind.EVENT,
            name,
            pd.DataFrame({"eid": ids, "ts": stamps}),
            id_col="eid",
            time_col="ts",
        )
    for name, eids, oids, to_type in relations:
        store.add_relation(
            RelationKind.E2O,
            name,
            pd.DataFrame({"eid": eids, "oid": oids}),
            from_id="eid",
            to_id="oid",
            to_type=to_type,
        )
    return store


def _utc(*days):
    return pd.Series(pd.to_datetime([f"2024-01-{d:02d}" for d in days], utc=True))


@pytest.fixture
def store():
    """Three events on o1 and o2, one of them related twice, and one event on i1."""
    return _store(
        ("place", ["e1", "e2"], _utc(1, 2)),
        ("ship", ["e3", "e4"], _utc(3, 4)),
        relations=[
            ("places", ["e1", "e2", "e3"], ["o1", "o2", "o1"], ["order"]),
            ("touches", ["e1", "e4"], ["o1", "i1"], ["order", "item"]),
        ],
    )


def test_a_log_holds_each_event_of_a_case_once_in_the_pm4py_columns(store):
    log = get_log(store, "order")

    assert log.index.name == "event_id"
    assert log.reset_index().to_dict("list") == {
        "event_id": ["e1", "e3", "e2"],
        "case:concept:name": ["o1", "o1", "o2"],
        "concept:name": ["place", "ship", "place"],
        "time:timestamp": list(_utc(1, 3, 2)),
    }


def _positioned_store(positions):
    """Return a store whose events e1, e9 and e10 on o1 carry the given positions.

    e1 is a day later than e9 and e10, which share a timestamp and belong to
    two activities.
    """
    store = OCEDStore()
    store.add(EntityKind.OBJECT, "order", pd.DataFrame({"oid": ["o1"]}), id_col="oid")
    for name, ids, days in (("late", ["e1"], [2]), ("tied", ["e9"], [1])):
        store.add(
            EntityKind.EVENT,
            name,
            pd.DataFrame({"eid": ids, "ts": _utc(*days), "pos": [positions[ids[0]]]}),
            id_col="eid",
            time_col="ts",
            position_col="pos",
        )
    store.add(
        EntityKind.EVENT,
        "other",
        pd.DataFrame({"eid": ["e10"], "ts": _utc(1), "pos": [positions["e10"]]}),
        id_col="eid",
        time_col="ts",
        position_col="pos",
    )
    store.add_relation(
        RelationKind.E2O,
        "on",
        pd.DataFrame({"eid": ["e1", "e9", "e10"], "oid": ["o1"] * 3}),
        from_id="eid",
        to_id="oid",
        to_type=["order"],
    )
    return store


@pytest.mark.parametrize(
    ("positions", "expected"),
    [
        ({"e10": 0, "e9": 1, "e1": 2}, ["e10", "e9", "e1"]),
        ({"e9": 0, "e1": 1, "e10": 2}, ["e9", "e10", "e1"]),
    ],
)
def test_events_of_a_case_are_ordered_by_timestamp_then_by_position(
    positions, expected
):
    assert list(get_log(_positioned_store(positions), "order").index) == expected


def test_the_position_survives_cleaning():
    store = clean_oced_data(_positioned_store({"e10": 0, "e9": 1, "e1": 2}))

    assert list(get_log(store, "order").index) == ["e10", "e9", "e1"]


def test_events_without_positions_at_one_timestamp_keep_the_store_order():
    store = _store(
        ("late", ["e9"], _utc(2)),
        ("tied", ["e2", "e1"], _utc(1, 1)),
        relations=[("on", ["e9", "e2", "e1"], ["o1", "o1", "o1"], ["order"])],
    )

    assert list(get_log(store, "order").index) == ["e2", "e1", "e9"]


def test_an_event_with_no_timestamp_is_left_out_and_warned_about(caplog):
    store = _store(
        (
            "act",
            ["e1", "e2"],
            pd.Series([pd.Timestamp("2024-01-01", tz="UTC"), pd.NaT]),
        ),
        relations=[("on", ["e1", "e2"], ["o1", "o1"], ["order"])],
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        log = get_log(store, "order")

    assert list(log.index) == ["e1"]
    assert (
        "1 event(s) have no timestamp and are left out of the event order of order."
        in caplog.text
    )


def test_a_related_id_in_no_object_table_is_dropped_and_warned_about(caplog):
    """An id of another object type is dropped without a warning."""
    store = _store(
        ("act", ["e1", "e2", "e3"], _utc(1, 2, 3)),
        relations=[("on", ["e1", "e2", "e3"], ["o1", "i1", "ghost"], ["order"])],
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        log = get_log(store, "order")

    assert list(log["case:concept:name"]) == ["o1"]
    assert "1 referenced object id(s) are in no object table" in caplog.text
    assert "['ghost']" in caplog.text


def test_iterating_gives_each_case_concept_the_log_get_log_gives_it(store):
    logs = dict(iter_logs(store, ["order", "item"]))

    assert list(logs) == ["order", "item"]
    assert_frame_equal(logs["item"], get_log(store, "item"))


@pytest.mark.parametrize(
    ("case_concept", "message"),
    [
        ("nope", r"Unknown object type 'nope'\. This store holds \['item', 'order'\]"),
        ("item", "No event-to-object relation names object type 'item'"),
    ],
)
def test_a_case_concept_no_relation_reaches_is_rejected(case_concept, message):
    store = _store(
        ("act", ["e1"], _utc(1)), relations=[("on", ["e1"], ["o1"], ["order"])]
    )

    with pytest.raises(InputError, match=message):
        get_log(store, case_concept)


def test_a_store_with_no_events_cannot_be_flattened():
    store = _store(relations=[("on", ["e1"], ["o1"], ["order"])])

    with pytest.raises(InputError, match="No event data available"):
        get_log(store, "order")


@pytest.mark.parametrize(
    "unzoned",
    [
        pd.Series(pd.to_datetime(["2024-01-01"])),
        pd.Series([pd.Timestamp("2024-01-01")], dtype=object),
    ],
    ids=["datetime_column", "object_column"],
)
def test_activities_mixing_zoned_and_unzoned_timestamps_are_rejected_by_name(unzoned):
    store = _store(
        ("naive", ["e1"], unzoned),
        ("aware", ["e2"], _utc(2)),
        relations=[("on", ["e1", "e2"], ["o1", "o1"], ["order"])],
    )

    with pytest.raises(
        InputError,
        match=r"mix zoned and unzoned timestamps:\n  with a zone: aware\n  without one: naive",
    ):
        get_log(store, "order")


# --------------------------------------------------------------------------
# Choosing case concepts
# --------------------------------------------------------------------------


def test_only_a_type_an_event_relation_names_is_a_case_concept_candidate():
    store = _store(
        ("act", ["e1"], _utc(1)), relations=[("on", ["e1"], ["i1"], ["item"])]
    )

    assert case_concept_candidates(store) == ["item"]


def test_without_a_choice_the_most_common_reachable_types_are_chosen(store):
    store.add(
        EntityKind.OBJECT,
        "unreached",
        pd.DataFrame({"oid": ["u1", "u2", "u3"]}),
        id_col="oid",
    )

    assert choose_case_concepts(store, None, auto=2) == (["order", "item"], "auto(n=2)")


def test_without_a_reachable_type_none_is_chosen_and_a_warning_given(caplog):
    store = _store(("act", ["e1"], _utc(1)))

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        chosen = choose_case_concepts(store, None)

    assert chosen == ([], "auto(n=3)")
    assert "No case concept was chosen" in caplog.text


def test_a_chosen_case_concept_is_followed_to_its_name_after_cleaning():
    store = _store(("act", ["e1"], _utc(1)))
    store.add(
        EntityKind.OBJECT, "line item", pd.DataFrame({"oid": ["l1"]}), id_col="oid"
    )
    store = clean_oced_data(store)

    assert choose_case_concepts(store, ["line item", "order", "nope"]) == (
        ["line_item", "order", "nope"],
        "config",
    )
