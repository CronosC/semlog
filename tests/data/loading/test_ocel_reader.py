"""Reading the three forms of an OCEL 2.0 log into an OCEL store."""

import logging
import sqlite3
from dataclasses import fields

import pandas as pd
import pytest

from semanticlog.core.errors import InputError
from semanticlog.data.loading.ocel_reader import read_ocel
from semanticlog.data.loading.ocel_store import OCELStore
from tests.support.ocel_files import EPOCH, small_log, write_log, write_sqlite


def test_the_json_xml_and_sqlite_forms_of_one_log_give_equal_stores(tmp_path):
    stores = [
        read_ocel(write_log(tmp_path, small_log(), suffix))
        for suffix in (".jsonocel", ".xmlocel", ".sqlite")
    ]

    for other in stores[1:]:
        for table in fields(OCELStore):
            pd.testing.assert_frame_equal(
                getattr(other, table.name), getattr(stores[0], table.name)
            )


def test_a_declared_attribute_value_holds_its_type(tmp_path):
    store = read_ocel(write_log(tmp_path, small_log(), ".xmlocel"))
    values = store.event_values.set_index("name")["value"]

    assert values["count"] == 3
    assert values["due"] == pd.Timestamp("2024-02-01", tz="UTC")
    assert store.object_values.set_index("name")["value"]["rush"] is True


def test_an_event_time_with_an_offset_is_converted_to_utc(tmp_path):
    log = small_log()
    log["events"][0]["time"] = "2024-01-01T10:00:00-07:00"

    store = read_ocel(write_log(tmp_path, log))

    assert store.events["time"][0] == pd.Timestamp("2024-01-01T17:00:00", tz="UTC")


@pytest.mark.parametrize("suffix", [".jsonocel", ".xmlocel", ".sqlite"])
def test_an_event_time_that_does_not_parse_fails_the_log_naming_the_event(
    tmp_path, suffix
):
    log = small_log()
    log["events"][1]["time"] = "yesterday"

    with pytest.raises(
        InputError,
        match=r"has 1 event\(s\) whose time is not an ISO 8601 timestamp, "
        r"such as e2 \('yesterday'\)\. Write every event time as ISO 8601\.",
    ):
        read_ocel(write_log(tmp_path, log, suffix))


def test_the_attribute_values_of_an_object_are_ordered_by_time(tmp_path):
    log = small_log()
    log["objects"][0]["attributes"].reverse()

    store = read_ocel(write_log(tmp_path, log))
    status = store.object_values[store.object_values["name"] == "status"]

    assert list(status["value"]) == ["new", "done"]


def test_both_qualifiers_of_an_event_linked_twice_to_one_object_are_kept(tmp_path):
    log = small_log()
    log["events"][1]["relationships"].append({"objectId": "o1", "qualifier": "checks"})

    store = read_ocel(write_log(tmp_path, log))

    assert store.e2o[store.e2o["event_id"] == "e2"]["qualifier"].tolist() == [
        "ships",
        "checks",
    ]


def test_sqlite_events_are_positioned_in_the_row_order_of_the_event_table(tmp_path):
    path = tmp_path / "log.sqlite"
    write_sqlite(small_log(), path, event_rows=["e2", "e1"])

    events = read_ocel(str(path)).events

    assert dict(zip(events["id"], events["position"], strict=True)) == {
        "e2": 0,
        "e1": 1,
    }


@pytest.mark.parametrize("suffix", [".jsonocel", ".xmlocel"])
def test_one_id_for_two_events_is_rejected(tmp_path, suffix):
    log = small_log()
    log["events"][1]["id"] = "e1"

    with pytest.raises(
        InputError, match="uses one event id for more than one event: e1"
    ):
        read_ocel(write_log(tmp_path, log, suffix))


def test_a_suffix_with_no_ocel_form_is_rejected(tmp_path):
    with pytest.raises(InputError, match=r"Cannot infer OCEL format from '.*log\.csv'"):
        read_ocel(str(tmp_path / "log.csv"))


def test_an_object_without_an_id_is_left_out_with_a_warning(tmp_path, caplog):
    log = small_log()
    log["objects"].append({"type": "item"})

    with caplog.at_level(logging.WARNING):
        store = read_ocel(write_log(tmp_path, log))

    assert store.objects["id"].tolist() == ["o1", "i1"]
    assert "1 object(s) of" in caplog.text


def test_an_undeclared_attribute_value_is_kept_as_read(tmp_path):
    log = small_log()
    log["events"][1]["attributes"] = [{"name": "note", "value": "late"}]

    store = read_ocel(write_log(tmp_path, log, ".xmlocel"))

    assert store.event_values.set_index("name")["value"]["note"] == "late"


@pytest.mark.parametrize(
    ("suffix", "text", "message"),
    [
        (".jsonocel", "{ not json", r"log\.jsonocel is not well-formed JSON: "),
        (".xmlocel", "<log><events>", r"log\.xmlocel is not well-formed XML: "),
    ],
)
def test_a_file_that_does_not_parse_is_rejected_naming_it(
    tmp_path, suffix, text, message
):
    path = tmp_path / f"log{suffix}"
    path.write_text(text)

    with pytest.raises(InputError, match=message):
        read_ocel(str(path))


def test_a_type_entry_without_a_name_is_rejected_naming_the_field(tmp_path):
    log = small_log()
    del log["objectTypes"][0]["name"]

    with pytest.raises(
        InputError, match=r"is not an OCEL 2\.0 JSON log: an entry has no 'name' field"
    ):
        read_ocel(write_log(tmp_path, log))


def test_a_sqlite_database_without_the_ocel_tables_is_rejected_naming_one(tmp_path):
    path = tmp_path / "log.sqlite"
    sqlite3.connect(path).execute("CREATE TABLE other (x TEXT)").connection.close()

    with pytest.raises(
        InputError,
        match=r"log\.sqlite is not an OCEL 2\.0 SQLite database: it has no table "
        r"event_map_type\.",
    ):
        read_ocel(str(path))


def test_a_sqlite_file_that_is_not_a_database_is_rejected(tmp_path):
    path = tmp_path / "log.sqlite"
    path.write_text("hello, not a database")

    with pytest.raises(
        InputError, match=r"log\.sqlite is not an OCEL 2\.0 SQLite database: "
    ):
        read_ocel(str(path))


@pytest.mark.parametrize("suffix", [".jsonocel", ".xmlocel", ".sqlite"])
def test_a_log_without_events_is_rejected(tmp_path, suffix):
    log = small_log()
    log["events"] = []

    with pytest.raises(InputError, match=r"log\.\w+ has no events\. "):
        read_ocel(write_log(tmp_path, log, suffix))


def test_a_list_or_object_value_is_read_as_json_text_with_one_warning(tmp_path, caplog):
    log = small_log()
    log["events"][1]["attributes"] = [{"name": "tags", "value": ["a", "b"]}]
    log["objects"][1]["attributes"].append(
        {"name": "size", "time": EPOCH, "value": {"w": "é"}}
    )

    with caplog.at_level(logging.WARNING):
        store = read_ocel(write_log(tmp_path, log))

    assert store.event_values.set_index("name")["value"]["tags"] == '["a", "b"]'
    assert store.object_values.set_index("name")["value"]["size"] == '{"w": "é"}'
    assert "Attribute 'tags' has 1 list or object value(s)" in caplog.text
    assert "Attribute 'size' has 1 list or object value(s)" in caplog.text
