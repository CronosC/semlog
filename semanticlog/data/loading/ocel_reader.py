"""Read the JSON, XML and SQLite forms of an OCEL 2.0 log into an `OCELTables`.

`read_ocel` picks the form by suffix. The forms are defined in the OCEL 2.0
specification, sections 6 and 7.
"""

import json
import logging
import sqlite3
import xml.etree.ElementTree as ElementTree
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ijson
import pandas as pd

from semanticlog.core.errors import InputError
from semanticlog.data.dtypes import cast_declared
from semanticlog.data.loading.ocel_frames import (
    OCEL_FORMATS,
    reject_repeated_ids,
    repeated,
)
from semanticlog.data.loading.ocel_schema import (
    DeclaredTypes,
    normalize_declared,
    sqlite_types,
)
from semanticlog.data.loading.ocel_tables import OCELTables
from semanticlog.data.tables import EntityKind

logger = logging.getLogger(__name__)

_SAMPLE = 3

_FORM_NAMES = {"json": "JSON", "xml": "XML", "sqlite": "SQLite"}

RawTypes = dict[tuple[EntityKind, str], dict[str, str]]


def _text(value: object) -> str | None:
    """Return the value as text, or None if it is missing."""
    return None if value is None else str(value)


@dataclass
class _Rows:
    """The entries of one log as read, one list per column of an `OCELTables` table."""

    event_ids: list[str | None] = field(default_factory=list)
    event_types: list[str | None] = field(default_factory=list)
    event_times: list[object] = field(default_factory=list)
    object_ids: list[str | None] = field(default_factory=list)
    object_types: list[str | None] = field(default_factory=list)
    e2o: list[tuple[str | None, str | None, object]] = field(default_factory=list)
    o2o: list[tuple[str | None, str | None, object]] = field(default_factory=list)
    event_values: list[tuple[str | None, str | None, object]] = field(
        default_factory=list
    )
    object_values: list[tuple[str | None, str | None, object, object]] = field(
        default_factory=list
    )
    types: RawTypes = field(default_factory=dict)

    def add_event(self, event_id: object, event_type: object, time: object) -> None:
        """Record one event in file order."""
        self.event_ids.append(_text(event_id))
        self.event_types.append(_text(event_type))
        self.event_times.append(time)

    def add_object(self, object_id: object, object_type: object) -> None:
        """Record one object in file order."""
        self.object_ids.append(_text(object_id))
        self.object_types.append(_text(object_type))


def read_ocel(path: str) -> OCELTables:
    """Read an OCEL 2.0 log in the form its suffix names.

    Raises:
        InputError: If the suffix names no OCEL form, the file does not parse
            in that form or lacks a part the standard requires, the log has
            no events, the file uses one id for two objects or two events, or
            an event time is not an ISO 8601 timestamp.
    """
    file_format = OCEL_FORMATS.get(Path(path).suffix.lower())
    if file_format is None:
        raise InputError(
            f"Cannot infer OCEL format from '{path}'. Supported extensions: "
            f"{', '.join(sorted(OCEL_FORMATS))}."
        )
    readers = {"json": _read_json, "xml": _read_xml, "sqlite": _read_sqlite}
    try:
        rows = readers[file_format](path)
    except (ijson.JSONError, ElementTree.ParseError, UnicodeDecodeError) as exc:
        detail = (
            str(exc).splitlines()[0].rstrip(".") if str(exc) else type(exc).__name__
        )
        raise InputError(
            f"{path} is not well-formed {file_format.upper()}: {detail}. Fix "
            "the file or export the log again."
        ) from exc
    except sqlite3.DatabaseError as exc:
        raise InputError(
            f"{path} is not an OCEL 2.0 SQLite database: {_sqlite_problem(exc)}. "
            "Export the log again as OCEL 2.0 SQLite, JSON or XML."
        ) from exc
    except KeyError as exc:
        part = "column" if file_format == "sqlite" else "field"
        kind = "a table" if file_format == "sqlite" else "an entry"
        raise InputError(
            f"{path} is not an OCEL 2.0 {_FORM_NAMES[file_format]} log: {kind} "
            f"has no {exc.args[0]!r} {part}. Add it as the OCEL 2.0 standard "
            "describes, or export the log again."
        ) from exc
    except (TypeError, AttributeError) as exc:
        if file_format != "json":
            raise
        raise InputError(
            f"{path} is not an OCEL 2.0 JSON log: an entry is not shaped as the "
            f"OCEL 2.0 standard describes ({exc}). Export the log again."
        ) from exc
    return _store(path, rows)


def _sqlite_problem(exc: sqlite3.DatabaseError) -> str:
    """Return what a SQLite error says about the file, in words a user reads."""
    text = str(exc)
    if text.startswith("no such table: "):
        return f"it has no table {text.removeprefix('no such table: ')}"
    if text == "file is not a database":
        return "it is not a SQLite database"
    return text


def _read_json(path: str) -> _Rows:
    """Return the entries of a JSON log, streaming one top-level array at a time.

    A list or object attribute value is kept as JSON text, with one warning
    per attribute.
    """
    rows = _Rows()
    nested: Counter[str] = Counter()

    def items(prefix: str) -> Iterable[dict[str, Any]]:
        with open(path, "rb") as handle:
            yield from ijson.items(handle, f"{prefix}.item", use_float=True)

    def scalar(attribute: dict[str, Any]) -> object:
        value = attribute.get("value")
        if isinstance(value, list | dict):
            nested[str(attribute.get("name"))] += 1
            return json.dumps(value, ensure_ascii=False)
        return value

    for kind, section in (
        (EntityKind.OBJECT, "objectTypes"),
        (EntityKind.EVENT, "eventTypes"),
    ):
        for entry in items(section):
            rows.types[(kind, str(entry["name"]))] = {
                str(a["name"]): str(a.get("type", ""))
                for a in entry.get("attributes") or []
            }
    for obj in items("objects"):
        object_id = _text(obj.get("id"))
        rows.add_object(object_id, obj.get("type"))
        for a in obj.get("attributes") or []:
            rows.object_values.append(
                (object_id, _text(a.get("name")), a.get("time"), scalar(a))
            )
        for r in obj.get("relationships") or []:
            rows.o2o.append((object_id, _text(r.get("objectId")), r.get("qualifier")))
    for event in items("events"):
        event_id = _text(event.get("id"))
        rows.add_event(event_id, event.get("type"), event.get("time"))
        for a in event.get("attributes") or []:
            rows.event_values.append((event_id, _text(a.get("name")), scalar(a)))
        for r in event.get("relationships") or []:
            rows.e2o.append((event_id, _text(r.get("objectId")), r.get("qualifier")))
    for name, count in nested.items():
        logger.warning(
            "Attribute '%s' has %d list or object value(s) in %s. They are read "
            "as JSON text.",
            name,
            count,
            path,
        )
    return rows


def _local(tag: str) -> str:
    """Return an XML tag without its namespace."""
    return tag.rsplit("}", 1)[-1]


def _read_xml(path: str) -> _Rows:
    """Return the entries of an XML log, clearing each entry once it is read."""
    rows = _Rows()
    kinds = {"object-type": EntityKind.OBJECT, "event-type": EntityKind.EVENT}
    depth = 0
    section: ElementTree.Element | None = None
    # The file is the user's own log.
    for action, element in ElementTree.iterparse(  # noqa: S314
        path, events=("start", "end")
    ):
        if action == "start":
            depth += 1
            if depth == 2:
                section = element
            continue
        depth -= 1
        if depth != 2:
            continue
        tag = _local(element.tag)
        if tag in kinds:
            rows.types[(kinds[tag], str(element.get("name")))] = {
                str(a.get("name")): str(a.get("type", ""))
                for a in element.iter()
                if _local(a.tag) == "attribute"
            }
        elif tag == "object":
            _xml_object(element, rows)
        elif tag == "event":
            _xml_event(element, rows)
        if section is not None:
            section.clear()
    return rows


def _xml_object(element: ElementTree.Element, rows: _Rows) -> None:
    """Record one `object` element with its attribute values and relationships."""
    object_id = element.get("id")
    rows.add_object(object_id, element.get("type"))
    for child in element:
        if _local(child.tag) == "attributes":
            for a in child:
                rows.object_values.append(
                    (object_id, a.get("name"), a.get("time"), a.text)
                )
        elif _local(child.tag) == "objects":
            for r in child:
                rows.o2o.append((object_id, r.get("object-id"), r.get("qualifier")))


def _xml_event(element: ElementTree.Element, rows: _Rows) -> None:
    """Record one `event` element with its attribute values and relationships."""
    event_id = element.get("id")
    rows.add_event(event_id, element.get("type"), element.get("time"))
    for child in element:
        if _local(child.tag) == "attributes":
            for a in child:
                rows.event_values.append((event_id, a.get("name"), a.text))
        elif _local(child.tag) == "objects":
            for r in child:
                rows.e2o.append((event_id, r.get("object-id"), r.get("qualifier")))


def _quoted(name: str) -> str:
    """Return a SQLite identifier in double quotes."""
    return '"' + name.replace('"', '""') + '"'


def _read_sqlite(path: str) -> _Rows:
    """Return the entries of a SQLite log, with events in the row order of the `event` table."""
    rows = _Rows()
    # sqlite3.connect creates a missing file instead of failing.
    Path(path).stat()
    connection = sqlite3.connect(path)
    try:
        times: dict[str | None, object] = {}
        for _, suffix in connection.execute(
            "SELECT ocel_type, ocel_type_map FROM event_map_type"
        ):
            cursor = connection.execute(
                f"SELECT * FROM {_quoted('event_' + str(suffix))}"  # noqa: S608
            )
            columns = [d[0] for d in cursor.description]
            for row in cursor:
                record = dict(zip(columns, row, strict=True))
                event_id = _text(record.pop("ocel_id"))
                times[event_id] = record.pop("ocel_time", None)
                for name, value in record.items():
                    rows.event_values.append((event_id, name, value))
        for event_id, event_type in connection.execute(
            "SELECT ocel_id, ocel_type FROM event ORDER BY rowid"
        ):
            rows.add_event(event_id, event_type, times.get(_text(event_id)))

        for object_id, object_type in connection.execute(
            "SELECT ocel_id, ocel_type FROM object ORDER BY rowid"
        ):
            rows.add_object(object_id, object_type)
        for _, suffix in connection.execute(
            "SELECT ocel_type, ocel_type_map FROM object_map_type"
        ):
            cursor = connection.execute(
                f"SELECT * FROM {_quoted('object_' + str(suffix))} "  # noqa: S608
                "ORDER BY rowid"
            )
            columns = [d[0] for d in cursor.description]
            for row in cursor:
                record = dict(zip(columns, row, strict=True))
                object_id = _text(record.pop("ocel_id"))
                time = record.pop("ocel_time", None)
                changed = record.pop("ocel_changed_field", None)
                for name, value in record.items():
                    if changed is None or name == changed:
                        rows.object_values.append((object_id, name, time, value))

        rows.e2o.extend(
            (_text(e), _text(o), q)
            for e, o, q in connection.execute(
                "SELECT ocel_event_id, ocel_object_id, ocel_qualifier "
                "FROM event_object ORDER BY rowid"
            )
        )
        rows.o2o.extend(
            (_text(s), _text(t), q)
            for s, t, q in connection.execute(
                "SELECT ocel_source_id, ocel_target_id, ocel_qualifier "
                "FROM object_object ORDER BY rowid"
            )
        )
    finally:
        connection.close()
    rows.types = sqlite_types(path)
    return rows


def _blank(values: pd.Series) -> pd.Series:
    """Return where a value is missing or empty text."""
    return values.isna() | values.eq("")


def _entities(
    path: str, what: str, ids: list[str | None], types: list[str | None]
) -> pd.DataFrame:
    """Return the ids and types of the events or objects with both given.

    Rows without an id or type are dropped with a warning.

    Raises:
        InputError: If an id repeats.
    """
    frame = pd.DataFrame({"id": ids, "type": types}, dtype=object)
    missing = _blank(frame["id"]) | _blank(frame["type"])
    if missing.any():
        logger.warning(
            "%d %s(s) of %s have no id or no type and were left out.",
            int(missing.sum()),
            what,
            path,
        )
    return frame[~missing]


def _event_times(path: str, events: pd.DataFrame) -> pd.Series:
    """Return the time of each event as a UTC timestamp.

    Raises:
        InputError: If a time is missing or is not an ISO 8601 timestamp.
    """
    parsed = pd.to_datetime(events["time"], errors="coerce", utc=True, format="ISO8601")
    failed = parsed.isna()
    if failed.any():
        sample = [
            f"{event_id} ({time!r})"
            for event_id, time in zip(
                events["id"][failed][:_SAMPLE],
                events["time"][failed][:_SAMPLE],
                strict=True,
            )
        ]
        raise InputError(
            f"{path} has {int(failed.sum())} event(s) whose time is not an "
            f"ISO 8601 timestamp, such as {', '.join(sample)}. Write every "
            "event time as ISO 8601."
        )
    return parsed


def _number_as_boolean(value: object) -> object:
    """Return a number as a boolean, true unless it is zero, and any other value unchanged."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return bool(value)
    return value


def _typed(
    values: pd.DataFrame,
    owner_types: pd.Series,
    kind: EntityKind,
    declared: DeclaredTypes,
) -> pd.Series:
    """Return the `value` column with each value of a declared attribute cast to its type.

    Args:
        owner_types: The type of the event or object each row belongs to.
    """
    typed = values["value"].astype(object)
    keys = pd.DataFrame({"type": owner_types, "name": values["name"]})
    for (type_name, name), group in keys.groupby(["type", "name"], sort=False):
        declared_type = declared.get((kind, str(type_name)), {}).get(str(name))
        if declared_type is None:
            continue
        column = str(name)
        given = typed.loc[group.index]
        if declared_type == "boolean":
            # SQLite has no boolean type and stores one as 0 or 1.
            given = given.map(_number_as_boolean)
        cast = cast_declared(pd.DataFrame({column: given}), {column: declared_type})[
            column
        ]
        typed.loc[group.index] = cast.astype(object)
    return typed


def _values(
    rows: list[tuple[Any, ...]],
    columns: list[str],
    positions: pd.Series,
    *,
    keep_null: bool = False,
) -> pd.DataFrame:
    """Return the named attribute values of known owners.

    Args:
        positions: The place of each owner in the file, keyed by id. The first
            column of a row is the owner id.
        keep_null: Keep the rows whose value is missing.
    """
    frame = pd.DataFrame(rows, columns=columns)
    owner = frame[columns[0]].map(positions)
    keep = owner.notna() & frame["name"].notna()
    if not keep_null:
        keep &= frame["value"].notna()
    return frame[keep].assign(_owner=owner[keep])


def _store(path: str, rows: _Rows) -> OCELTables:
    """Return the entries of a log as an `OCELTables`.

    Raises:
        InputError: If no event has an id and a type, an id repeats, or an
            event time is not an ISO 8601 timestamp.
    """
    events = _entities(path, "event", rows.event_ids, rows.event_types).assign(
        time=pd.Series(rows.event_times, dtype=object)
    )
    if events.empty:
        raise InputError(
            f"{path} has no events. Check that the file is the log to build "
            "and that every event has an id and a type."
        )
    objects = _entities(path, "object", rows.object_ids, rows.object_types)
    reject_repeated_ids(
        path, {"object": repeated(objects["id"]), "event": repeated(events["id"])}
    )
    events = events.assign(
        time=_event_times(path, events), position=events.index
    ).reset_index(drop=True)
    objects = objects.reset_index(drop=True)
    event_position = pd.Series(events.index, index=events["id"])
    object_position = pd.Series(objects.index, index=objects["id"])
    event_type = events.set_index("id")["type"]
    object_type = objects.set_index("id")["type"]
    declared = normalize_declared(rows.types)

    event_values = _values(
        rows.event_values, ["event_id", "name", "value"], event_position
    ).sort_values("_owner", kind="stable")
    event_values["value"] = _typed(
        event_values,
        event_values["event_id"].map(event_type),
        EntityKind.EVENT,
        declared,
    )

    # A null value is kept, because it is the first value of its attribute
    # when it is the earliest.
    object_values = _values(
        rows.object_values,
        ["object_id", "name", "time", "value"],
        object_position,
        keep_null=True,
    )
    object_values["time"] = pd.to_datetime(
        object_values["time"], errors="coerce", utc=True, format="ISO8601"
    )
    object_values = object_values.sort_values(
        ["_owner", "time"], kind="stable", na_position="first"
    )
    object_values["value"] = _typed(
        object_values,
        object_values["object_id"].map(object_type),
        EntityKind.OBJECT,
        declared,
    )

    e2o = pd.DataFrame(rows.e2o, columns=["event_id", "object_id", "qualifier"])
    o2o = pd.DataFrame(rows.o2o, columns=["source_id", "target_id", "qualifier"])
    return OCELTables(
        events=events,
        objects=objects,
        e2o=_in_order(e2o, e2o["event_id"].map(event_position)),
        o2o=_in_order(o2o, o2o["source_id"].map(object_position)),
        event_values=_in_order(event_values.drop(columns="_owner")),
        object_values=_in_order(object_values.drop(columns="_owner")),
        declared=declared,
    )


def _in_order(frame: pd.DataFrame, by: pd.Series | None = None) -> pd.DataFrame:
    """Return the rows sorted stably by `by`, unknown last, with a fresh index."""
    if by is not None:
        frame = (
            frame.assign(_by=by)
            .sort_values("_by", kind="stable", na_position="last")
            .drop(columns="_by")
        )
    return frame.reset_index(drop=True)
