"""Converting an OCEL 2.0 log into an OCED store."""

import logging

import pytest

from semlog.data.loading.ocel_loader import OCELLoader
from semlog.data.tables import EntityKind
from tests.support.ocel_files import (
    events_in_order,
    small_log,
    write_log,
    write_sqlite,
)


def test_an_event_linked_twice_to_one_object_keeps_both_relations(tmp_path):
    log = small_log()
    log["events"][1]["relationships"].append({"objectId": "o1", "qualifier": "checks"})

    oced = OCELLoader(write_log(tmp_path, log)).get_oced()

    assert oced.relation("ships").data.values.tolist() == [["e2", "o1"]]
    assert oced.relation("checks").data.values.tolist() == [["e2", "o1"]]


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

    assert events_in_order(oced) == ["e2", "e1"]


@pytest.mark.parametrize("suffix", [".jsonocel", ".xmlocel"])
@pytest.mark.parametrize("listed", [["e1", "e2"], ["e2", "e1"]])
def test_events_at_one_time_keep_the_order_of_the_file(tmp_path, suffix, listed):
    log = small_log()
    log["events"][1]["time"] = log["events"][0]["time"]
    log["events"] = sorted(log["events"], key=lambda event: listed.index(event["id"]))

    oced = OCELLoader(write_log(tmp_path, log, suffix)).get_oced()

    assert events_in_order(oced) == listed


def test_sqlite_events_at_one_time_are_ordered_by_the_event_table(tmp_path):
    log = small_log()
    log["events"][1]["time"] = log["events"][0]["time"]
    path = tmp_path / "tied.sqlite"
    write_sqlite(log, path, event_rows=["e2", "e1"])

    oced = OCELLoader(str(path)).get_oced()

    assert events_in_order(oced) == ["e2", "e1"]


def test_a_relation_to_an_object_the_log_lacks_is_left_out_with_a_warning(
    tmp_path, caplog
):
    log = small_log()
    log["events"][1]["relationships"].append({"objectId": "gone", "qualifier": "x"})

    with caplog.at_level(logging.WARNING):
        oced = OCELLoader(write_log(tmp_path, log)).get_oced()

    assert "x" not in oced.relation_types()
    assert "1 e2o relation(s) of" in caplog.text


def test_a_log_without_objects_keeps_its_events(tmp_path):
    log = small_log()
    log["objectTypes"] = []
    log["objects"] = []
    for event in log["events"]:
        event["relationships"] = []

    oced = OCELLoader(write_log(tmp_path, log)).get_oced()

    assert sorted(events_in_order(oced)) == ["e1", "e2"]
    assert oced.types(EntityKind.OBJECT) == []
