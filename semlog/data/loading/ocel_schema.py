"""The attribute types an OCEL 2.0 file declares for its event and object types.

`declared_types` reads them from the JSON, XML or SQLite form. A declared type
decides the dtype of its column instead of inference.
"""

import json
import logging
import sqlite3
import xml.etree.ElementTree as ElementTree

from semlog.data.dtypes import DECLARED_TYPES
from semlog.data.tables import EntityKind

logger = logging.getLogger(__name__)

DeclaredTypes = dict[tuple[EntityKind, str], dict[str, str]]

# Matched in order against the lowercased type name. Exporters write
# variants such as `date`, `double` or SQL column types.
_TYPE_WORDS: tuple[tuple[str, str], ...] = (
    ("bool", "boolean"),
    ("int", "integer"),
    ("float", "float"),
    ("double", "float"),
    ("real", "float"),
    ("decimal", "float"),
    ("numeric", "float"),
    ("date", "time"),
    ("time", "time"),
    ("str", "string"),
    ("text", "string"),
    ("char", "string"),
)

_SQLITE_RESERVED = frozenset({"ocel_id", "ocel_time", "ocel_changed_field"})


def normalize_type(raw: str) -> str | None:
    """Return the OCEL type a declared type name stands for, or None if unknown."""
    lowered = raw.strip().lower()
    for word, declared in _TYPE_WORDS:
        if word in lowered:
            return declared
    return None


def declared_types(path: str, file_format: str) -> DeclaredTypes:
    """Return the declared type of each attribute, per event or object type.

    Args:
        path: The OCEL file.
        file_format: `json`, `xml` or `sqlite`.

    Returns:
        For each (kind, type name as the log spells it), the attribute names
        mapped to one of `DECLARED_TYPES`. An attribute whose declared type is
        not recognized is left out and its column is inferred.
    """
    readers = {"json": _from_json, "xml": _from_xml, "sqlite": sqlite_types}
    return normalize_declared(readers[file_format](path))


def normalize_declared(
    raw: dict[tuple[EntityKind, str], dict[str, str]],
) -> DeclaredTypes:
    """Return the declared types with each type name mapped to one of `DECLARED_TYPES`.

    An attribute whose type name is not recognized is left out, with a warning
    naming the type.
    """
    declared: DeclaredTypes = {}
    unknown: set[str] = set()
    for key, attributes in raw.items():
        types: dict[str, str] = {}
        for name, type_name in attributes.items():
            normalized = normalize_type(type_name)
            if normalized is None:
                unknown.add(type_name)
            else:
                types[name] = normalized
        declared[key] = types
    if unknown:
        logger.warning(
            "The log declares attribute type(s) %s, which are not among %s. "
            "Their columns are read by content.",
            ", ".join(sorted(map(repr, unknown))),
            ", ".join(DECLARED_TYPES),
        )
    return declared


def _from_json(path: str) -> dict[tuple[EntityKind, str], dict[str, str]]:
    """Return the raw declared types of a JSON log."""
    with open(path, encoding="utf-8") as handle:
        doc = json.load(handle)
    raw: dict[tuple[EntityKind, str], dict[str, str]] = {}
    for kind, section in (
        (EntityKind.OBJECT, "objectTypes"),
        (EntityKind.EVENT, "eventTypes"),
    ):
        for entry in doc.get(section, []):
            raw[(kind, str(entry["name"]))] = {
                str(a["name"]): str(a.get("type", ""))
                for a in entry.get("attributes", [])
            }
    return raw


def _from_xml(path: str) -> dict[tuple[EntityKind, str], dict[str, str]]:
    """Return the raw declared types of an XML log, reading only its type sections."""
    raw: dict[tuple[EntityKind, str], dict[str, str]] = {}
    kinds = {"object-type": EntityKind.OBJECT, "event-type": EntityKind.EVENT}
    # The file is the user's own log.
    for event, element in ElementTree.iterparse(  # noqa: S314
        path, events=("start", "end")
    ):
        if event == "start":
            if element.tag in ("objects", "events"):
                break
        elif element.tag in kinds:
            raw[(kinds[element.tag], element.attrib["name"])] = {
                a.attrib["name"]: a.attrib.get("type", "")
                for a in element.iter("attribute")
            }
    return raw


def sqlite_types(path: str) -> dict[tuple[EntityKind, str], dict[str, str]]:
    """Return the raw declared types of a SQLite log, from its column types."""
    raw: dict[tuple[EntityKind, str], dict[str, str]] = {}
    connection = sqlite3.connect(path)
    try:
        for kind, prefix in (
            (EntityKind.OBJECT, "object"),
            (EntityKind.EVENT, "event"),
        ):
            mapping = connection.execute(
                f"SELECT ocel_type, ocel_type_map FROM {prefix}_map_type"  # noqa: S608
            ).fetchall()
            for type_name, suffix in mapping:
                columns = connection.execute(
                    f'PRAGMA table_info("{prefix}_{suffix}")'
                ).fetchall()
                raw[(kind, str(type_name))] = {
                    str(column[1]): str(column[2])
                    for column in columns
                    if column[1] not in _SQLITE_RESERVED
                }
    finally:
        connection.close()
    return raw
