"""Converting an OCEL 2.0 log into an OCED store."""

import logging

import pandas as pd
import pytest

from semanticlog.core.enums import OCELExtension
from semanticlog.data.loading.ocel_loader import OCELLoader
from semanticlog.data.tables import EntityKind
from tests.support.ocel_files import EPOCH, small_log, write_log, write_sqlite

HISTORY = [OCELExtension.ATTRIBUTE_HISTORY]


def _positions(oced) -> dict[str, int]:
    """Return the position of every event, keyed by event id."""
    positions = {}
    for activity in oced.types(EntityKind.EVENT):
        table = oced.table(EntityKind.EVENT, activity)
        positions.update(
            zip(table.data[table.id], table.data[table.position], strict=True)
        )
    return positions


def _in_order(oced) -> list[str]:
    positions = _positions(oced)
    return sorted(positions, key=positions.get)


def test_an_event_linked_twice_to_one_object_keeps_both_relations(tmp_path):
    log = small_log()
    log["events"][1]["relationships"].append({"objectId": "o1", "qualifier": "checks"})

    oced = OCELLoader(write_log(tmp_path, log)).get_oced()

    assert oced.get_relation("ships").data.values.tolist() == [["e2", "o1"]]
    assert oced.get_relation("checks").data.values.tolist() == [["e2", "o1"]]


def test_an_event_without_objects_and_an_object_without_events_are_kept(tmp_path):
    log = small_log()
    log["events"][1]["relationships"] = []
    log["objects"].append({"id": "i2", "type": "item"})

    oced = OCELLoader(write_log(tmp_path, log)).get_oced()

    assert list(oced.ids(EntityKind.EVENT, "ship")) == ["e2"]
    assert list(oced.ids(EntityKind.OBJECT, "item")) == ["i1", "i2"]


def test_an_event_time_with_an_offset_is_ordered_by_its_utc_time(tmp_path):
    log = small_log()
    log["events"][0]["time"] = "2024-01-02T10:00:00-07:00"
    log["events"][1]["time"] = "2024-01-02T12:00:00Z"

    oced = OCELLoader(write_log(tmp_path, log)).get_oced()

    assert _in_order(oced) == ["e2", "e1"]


def test_sqlite_events_at_one_time_are_ordered_by_the_event_table(tmp_path):
    log = small_log()
    log["events"][1]["time"] = log["events"][0]["time"]
    path = tmp_path / "tied.sqlite"
    write_sqlite(log, path, event_rows=["e2", "e1"])

    oced = OCELLoader(str(path)).get_oced()

    assert _in_order(oced) == ["e2", "e1"]


def _status_log(tmp_path) -> str:
    """Write a log whose order `o1` lists its statuses out of time order."""
    log = small_log()
    log["objects"][0]["attributes"] = [
        {"name": "status", "time": "2024-01-05T00:00:00Z", "value": "shipped"},
        {"name": "status", "time": "2024-01-01T00:00:00Z", "value": "new"},
        {"name": "status", "time": "2024-01-09T00:00:00Z", "value": "delivered"},
        {"name": "status", "time": "2024-01-09T00:00:00Z", "value": "closed"},
    ]
    return write_log(tmp_path, log)


def test_an_object_carries_the_latest_value_the_last_listed_among_equal_times(
    tmp_path,
):
    oced = OCELLoader(_status_log(tmp_path)).get_oced()
    orders = oced.table(EntityKind.OBJECT, "order").data

    assert orders.set_index("ocel:oid").loc["o1", "status"] == "closed"


def test_the_changes_hold_every_value_after_the_earliest_in_time_order(tmp_path):
    oced = OCELLoader(_status_log(tmp_path), HISTORY).get_oced()

    assert oced.get_object_changes("order")["status"].tolist() == [
        "shipped",
        "delivered",
        "closed",
    ]


@pytest.mark.parametrize("suffix", [".jsonocel", ".xmlocel", ".sqlite"])
def test_the_values_after_a_null_earliest_value_are_all_changes(tmp_path, suffix):
    log = small_log()
    log["objects"][0]["attributes"] = [
        {"name": "status", "time": EPOCH, "value": None},
        {"name": "status", "time": "2024-01-03T00:00:00Z", "value": "a"},
        {"name": "status", "time": "2024-01-05T00:00:00Z", "value": "b"},
    ]

    oced = OCELLoader(write_log(tmp_path, log, suffix), HISTORY).get_oced()
    changes = oced.get_object_changes("order")
    orders = oced.table(EntityKind.OBJECT, "order").data

    assert list(zip(changes["ocel:timestamp"], changes["status"], strict=True)) == [
        (pd.Timestamp("2024-01-03", tz="UTC"), "a"),
        (pd.Timestamp("2024-01-05", tz="UTC"), "b"),
    ]
    assert orders.set_index("ocel:oid").loc["o1", "status"] == "b"


def test_a_relation_to_an_object_the_log_lacks_is_left_out_with_a_warning(
    tmp_path, caplog
):
    log = small_log()
    log["events"][1]["relationships"].append({"objectId": "gone", "qualifier": "x"})

    with caplog.at_level(logging.WARNING):
        oced = OCELLoader(write_log(tmp_path, log)).get_oced()

    assert "x" not in oced.get_relation_types()
    assert "1 e2o relation(s) of" in caplog.text
