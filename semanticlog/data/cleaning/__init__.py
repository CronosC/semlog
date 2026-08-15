"""Making the names in a log safe to emit.

Every activity, object type, relation, attribute, id and text value in a log
becomes part of an IRI, so anything outside `[A-Za-z0-9_.-]` has to go first.
A relation type also becomes a property, which RDF/XML writes as an element
name, so its first character is narrower still.

Cleaning does that, and then resolves the collisions it can cause: two source
names that clean to one name, one name claimed by two kinds of thing, and
names that clash with the ones the build reserves for its own schema. Each
rename is recorded, so the ontology can still be annotated with what the
source called things, and what is left is checked for duplicates before the
store is handed back.

`clean_oced_data` is the entry point, and it has to run before any build. The
work is split across three modules: `names` rewrites one name, `renames`
records what it was called, and `collisions` decides what has to move.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from typing import cast

import pandas as pd

from semanticlog.core.summary import count_phrase, name_lines
from semanticlog.data.cleaning.collisions import (
    apply_attribute_renames as apply_attribute_renames,
)
from semanticlog.data.cleaning.collisions import (
    collect_attribute_names as collect_attribute_names,
)
from semanticlog.data.cleaning.collisions import (
    fix_duplicate_names,
    record_id_renames,
    record_value_renames,
    resolve_kind_collisions,
    resolve_merged_names,
    resolve_property_names,
    verify_unique_names,
)
from semanticlog.data.cleaning.names import (
    clean_dataframe,
    clean_object_changes,
    clean_string,
    clean_value,
    is_timestamp,
)
from semanticlog.data.cleaning.names import cleaned as _cleaned
from semanticlog.data.cleaning.renames import RenameKind, RenameLog
from semanticlog.data.oced import OCEDStore
from semanticlog.data.readable import ReadableStore
from semanticlog.data.tables import (
    EventTable,
    ObjectTable,
    RelationKind,
    Relations,
    RelationTable,
)

__all__ = [
    "RenameKind",
    "RenameLog",
    "clean_dataframe",
    "clean_object_changes",
    "clean_oced_data",
    "clean_string",
    "clean_value",
    "fix_duplicate_names",
    "is_timestamp",
    "object_type_renames",
]

logger = logging.getLogger(__name__)

_RENAME_WORDS = {
    "activity": ("activity", "activities"),
    "object_type": ("object type", "object types"),
    "relation": ("relation", "relations"),
    "attribute": ("attribute", "attributes"),
    "id": ("id", "ids"),
    "value": ("value", "values"),
}


def clean_oced_data(data: OCEDStore, bin_labels: Collection[str] = ()) -> OCEDStore:
    """Make every name in a store safe to emit, and record what changed.

    This runs before any build. It renames the source names that would clean to
    one name, rewrites the unsafe characters out of schema names, ids and text
    values, resolves collisions with the names the build reserves, and leaves
    the resulting map of new name to source name on the store as its backmap.

    Args:
        data: Store to clean, modified in place.
        bin_labels: Bin labels the build will use, so that a bin class name is
            reserved against the log as well.

    Returns:
        The same store.

    Raises:
        ValueError: If two things still share a name once every rename has run,
            which no rename could resolve without choosing one of them.
    """
    logger.info("Cleaning data..")
    log = RenameLog()
    resolve_merged_names(data, log)
    resolve_kind_collisions(data, log)
    resolve_property_names(data, log)

    record_id_renames(data, log)
    record_value_renames(data, log)

    data.replace_tables(
        events=_clean_events(data, log),
        objects=_clean_objects(data, log),
        relations=_clean_relations(data, log),
    )
    if data.ocel_mode:
        data.replace_object_changes(
            {
                clean_string(obj_type): clean_object_changes(
                    data.get_object_changes(obj_type)
                )
                for obj_type in data.get_object_change_types()
            }
        )
    data = fix_duplicate_names(data, log, bin_labels)
    verify_unique_names(data)
    data.update_used_ids()
    data.backmap = log.flat()
    _report_renames(log)
    return data


def _report_renames(log: RenameLog) -> None:
    """Log what cleaning renamed, as counts and then the names themselves.

    The names are cut to a few lines each, unless the log is at debug level,
    where every one of them is printed.
    """
    grouped = log.by_kind()
    phrase = count_phrase(
        (len(grouped.get(kind, ())), singular, plural)
        for kind, (singular, plural) in _RENAME_WORDS.items()
    )
    if not phrase:
        return

    logger.info(f"  renamed {phrase}")
    complete = logger.isEnabledFor(logging.DEBUG)
    for kind, (_, plural) in _RENAME_WORDS.items():
        for line in name_lines(plural, grouped.get(kind, []), complete=complete):
            logger.info(line)


def object_type_renames(data: ReadableStore) -> dict[str, str]:
    """Return where cleaning moved the object types it renamed.

    A config or a command line names object types as the source log spells
    them, while the store holds them under their cleaned names. This reads the
    backmap the other way round, so a source name can be followed to the type
    it now names.

    Args:
        data: Store that has been cleaned.

    Returns:
        Source name to current name, holding only the object types whose name
        changed.
    """
    current = set(data.get_object_types())
    return {
        original: name for name, original in data.backmap.items() if name in current
    }


def _clean_columns(df: pd.DataFrame, skip: set[str], log: RenameLog) -> None:
    """Record the attribute columns that cleaning is about to rename."""
    for column in df.columns:
        if column in skip:
            continue
        cleaned = _cleaned(column)
        if cleaned != str(column):
            log.record(RenameKind.ATTRIBUTE, str(column), cleaned)


def _clean_events(data: OCEDStore, log: RenameLog) -> dict[str, EventTable]:
    """Return the event tables with activity, column and cell names cleaned."""
    events = {}
    for activity in data.get_event_types():
        table = data.get_event_table(activity)
        cleaned = clean_string(activity)
        if cleaned != activity:
            log.record(RenameKind.ACTIVITY, activity, cleaned)
        _clean_columns(table.data, {table.id, table.time}, log)
        events[cleaned] = EventTable(
            id=clean_string(table.id),
            time=clean_string(table.time),
            data=clean_dataframe(table.data),
        )
    return events


def _clean_objects(data: OCEDStore, log: RenameLog) -> dict[str, ObjectTable]:
    """Return the object tables with type, column and cell names cleaned."""
    objects = {}
    for obj_type in data.get_object_types():
        obj = data.get_object_table(obj_type)
        cleaned = clean_string(obj_type)
        if cleaned != obj_type:
            log.record(RenameKind.OBJECT_TYPE, obj_type, cleaned)
        _clean_columns(obj.data, {obj.id}, log)
        objects[cleaned] = ObjectTable(
            id=clean_string(obj.id),
            data=clean_dataframe(obj.data),
        )
    return objects


def _clean_relations(data: OCEDStore, log: RenameLog) -> Relations:
    """Return both relation families with their names and endpoints cleaned."""

    def clean_types(types: list[str] | None) -> list[str] | None:
        if types is None:
            return None
        return [_cleaned(t) for t in types]

    def clean_family(kind: RelationKind) -> dict[str, RelationTable]:
        family = data.get_e2o() if kind is RelationKind.E2O else data.get_o2o()
        cleaned_family = {}
        for name, rel in family.items():
            cleaned = clean_string(name)
            if cleaned != name:
                log.record(RenameKind.RELATION, name, cleaned)
            cleaned_family[cleaned] = RelationTable(
                from_id=clean_string(rel.from_id),
                from_type=clean_types(rel.from_type),
                to_id=clean_string(rel.to_id),
                to_type=cast(list, clean_types(rel.to_type)),
                data=clean_dataframe(rel.data),
            )
        return cleaned_family

    return Relations(
        event_to_object=clean_family(RelationKind.E2O),
        object_to_object=clean_family(RelationKind.O2O),
    )
