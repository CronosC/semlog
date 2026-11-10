"""OCEL 2.0 logs written in the JSON, XML and SQLite forms.

`small_log` describes a log in the JSON form, and `write_log` writes any such
description in the form a suffix names. `events_in_order` reads the event order
back from a loaded store.
"""

# S608: the SQLite logs are built from table names the tests chose.
# ruff: noqa: S608

import json
import sqlite3

from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind

EPOCH = "1970-01-01T00:00:00Z"

_SQL_TYPES = {
    "string": "TEXT",
    "integer": "INTEGER",
    "float": "REAL",
    "boolean": "BOOLEAN",
    "time": "TIMESTAMP",
}


def small_log() -> dict:
    """Return a log in the JSON form with one attribute of each declared type."""
    return {
        "objectTypes": [
            {
                "name": "order",
                "attributes": [
                    {"name": "status", "type": "string"},
                    {"name": "rush", "type": "boolean"},
                ],
            },
            {"name": "item", "attributes": [{"name": "weight", "type": "float"}]},
        ],
        "eventTypes": [
            {
                "name": "place",
                "attributes": [
                    {"name": "count", "type": "integer"},
                    {"name": "due", "type": "time"},
                ],
            },
            {"name": "ship", "attributes": []},
        ],
        "objects": [
            {
                "id": "o1",
                "type": "order",
                "attributes": [
                    {"name": "status", "time": EPOCH, "value": "new"},
                    {"name": "rush", "time": EPOCH, "value": True},
                    {"name": "status", "time": "2024-01-03T00:00:00Z", "value": "done"},
                ],
                "relationships": [{"objectId": "i1", "qualifier": "contains"}],
            },
            {
                "id": "i1",
                "type": "item",
                "attributes": [{"name": "weight", "time": EPOCH, "value": 2.5}],
            },
        ],
        "events": [
            {
                "id": "e1",
                "type": "place",
                "time": "2024-01-01T00:00:00Z",
                "attributes": [
                    {"name": "count", "value": 3},
                    {"name": "due", "value": "2024-02-01T00:00:00Z"},
                ],
                "relationships": [
                    {"objectId": "o1", "qualifier": "places"},
                    {"objectId": "i1", "qualifier": "adds"},
                ],
            },
            {
                "id": "e2",
                "type": "ship",
                "time": "2024-01-02T00:00:00Z",
                "relationships": [{"objectId": "o1", "qualifier": "ships"}],
            },
        ],
    }


def _xml_value(value: object) -> str:
    """Return a value as XML text, empty for a null value."""
    if value is None:
        return ""
    return str(value).lower() if isinstance(value, bool) else str(value)


def _xml(log: dict) -> str:
    """Return the log in the XML form."""

    def types(section: str, tag: str) -> str:
        return "".join(
            f'<{tag} name="{t["name"]}"><attributes>'
            + "".join(
                f'<attribute name="{a["name"]}" type="{a["type"]}"/>'
                for a in t["attributes"]
            )
            + f"</attributes></{tag}>"
            for t in log[section]
        )

    def relationships(entry: dict) -> str:
        return (
            "<objects>"
            + "".join(
                f'<relationship object-id="{r["objectId"]}" qualifier="{r["qualifier"]}"/>'
                for r in entry.get("relationships", [])
            )
            + "</objects>"
        )

    objects = "".join(
        f'<object id="{o["id"]}" type="{o["type"]}"><attributes>'
        + "".join(
            f'<attribute name="{a["name"]}" time="{a["time"]}">{_xml_value(a["value"])}</attribute>'
            for a in o.get("attributes", [])
        )
        + "</attributes>"
        + relationships(o)
        + "</object>"
        for o in log["objects"]
    )
    events = "".join(
        f'<event id="{e["id"]}" type="{e["type"]}" time="{e["time"]}"><attributes>'
        + "".join(
            f'<attribute name="{a["name"]}">{_xml_value(a["value"])}</attribute>'
            for a in e.get("attributes", [])
        )
        + "</attributes>"
        + relationships(e)
        + "</event>"
        for e in log["events"]
    )
    return (
        "<?xml version='1.0' encoding='UTF-8'?><log>"
        f"<object-types>{types('objectTypes', 'object-type')}</object-types>"
        f"<event-types>{types('eventTypes', 'event-type')}</event-types>"
        f"<objects>{objects}</objects><events>{events}</events></log>"
    )


def write_sqlite(log: dict, path, event_rows: list[str] | None = None) -> None:
    """Write the log in the SQLite form.

    Args:
        event_rows: The order of the rows of the `event` table, by event id.
            Defaults to the order of the log.
    """
    connection = sqlite3.connect(path)
    run = connection.execute
    run("CREATE TABLE event (ocel_id TEXT, ocel_type TEXT)")
    run("CREATE TABLE object (ocel_id TEXT, ocel_type TEXT)")
    run("CREATE TABLE event_map_type (ocel_type TEXT, ocel_type_map TEXT)")
    run("CREATE TABLE object_map_type (ocel_type TEXT, ocel_type_map TEXT)")
    run(
        "CREATE TABLE event_object "
        "(ocel_event_id TEXT, ocel_object_id TEXT, ocel_qualifier TEXT)"
    )
    run(
        "CREATE TABLE object_object "
        "(ocel_source_id TEXT, ocel_target_id TEXT, ocel_qualifier TEXT)"
    )
    for kind, section, extra in (
        ("event", "eventTypes", ""),
        ("object", "objectTypes", ", ocel_changed_field TEXT"),
    ):
        for t in log[section]:
            run(f"INSERT INTO {kind}_map_type VALUES (?, ?)", (t["name"], t["name"]))
            columns = "".join(
                f', "{a["name"]}" {_SQL_TYPES[a["type"]]}' for a in t["attributes"]
            )
            run(
                f'CREATE TABLE "{kind}_{t["name"]}" '
                f"(ocel_id TEXT, ocel_time TEXT{extra}{columns})"
            )
    events = {e["id"]: e for e in log["events"]}
    for event_id in event_rows or list(events):
        run("INSERT INTO event VALUES (?, ?)", (event_id, events[event_id]["type"]))
    for e in log["events"]:
        names = [a["name"] for a in e.get("attributes", [])]
        values = [a["value"] for a in e.get("attributes", [])]
        run(
            f'INSERT INTO "event_{e["type"]}" (ocel_id, ocel_time'
            + "".join(f', "{n}"' for n in names)
            + ") VALUES (?, ?"
            + ", ?" * len(names)
            + ")",
            (e["id"], e["time"], *values),
        )
        for r in e.get("relationships", []):
            run(
                "INSERT INTO event_object VALUES (?, ?, ?)",
                (e["id"], r["objectId"], r["qualifier"]),
            )
    for o in log["objects"]:
        run("INSERT INTO object VALUES (?, ?)", (o["id"], o["type"]))
        initial = [a for a in o.get("attributes", []) if a["time"] == EPOCH]
        changes = [a for a in o.get("attributes", []) if a["time"] != EPOCH]
        run(
            f'INSERT INTO "object_{o["type"]}" (ocel_id, ocel_time'
            + "".join(f', "{a["name"]}"' for a in initial)
            + ") VALUES (?, ?"
            + ", ?" * len(initial)
            + ")",
            (o["id"], EPOCH, *[a["value"] for a in initial]),
        )
        for a in changes:
            run(
                f'INSERT INTO "object_{o["type"]}" '
                f'(ocel_id, ocel_time, ocel_changed_field, "{a["name"]}") '
                "VALUES (?, ?, ?, ?)",
                (o["id"], a["time"], a["name"], a["value"]),
            )
        for r in o.get("relationships", []):
            run(
                "INSERT INTO object_object VALUES (?, ?, ?)",
                (o["id"], r["objectId"], r["qualifier"]),
            )
    connection.commit()
    connection.close()


def write_log(tmp_path, log: dict, suffix: str = ".jsonocel") -> str:
    """Write the log in the form the suffix names and return its path."""
    path = tmp_path / f"log{suffix}"
    if suffix == ".jsonocel":
        path.write_text(json.dumps(log))
    elif suffix == ".xmlocel":
        path.write_text(_xml(log))
    else:
        write_sqlite(log, path)
    return str(path)


def events_in_order(store: OCEDStore) -> list[str]:
    """Return the event ids of a store, sorted by the position the loader gave them."""
    positions: dict[str, int] = {}
    for activity in store.types(EntityKind.EVENT):
        table = store.table(EntityKind.EVENT, activity)
        positions.update(
            zip(table.data[table.id], table.data[table.position], strict=True)
        )
    return sorted(positions, key=positions.__getitem__)
