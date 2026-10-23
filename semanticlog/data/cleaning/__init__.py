"""Making the names in a log safe to emit.

`clean_oced_data` replaces every character outside `[A-Za-z0-9_.-]`, renames
the names that collide, and records the source name of each rename as the
backmap of the store. It has to run before any build.
"""

from __future__ import annotations

import logging
from collections.abc import Collection

import pandas as pd

from semanticlog.core.checks import raise_reports
from semanticlog.core.errors import InputError
from semanticlog.core.summary import report_by_kind
from semanticlog.data.cleaning.allocate import allocate_schema_names
from semanticlog.data.cleaning.merged import resolve_merged_names
from semanticlog.data.cleaning.names import (
    clean_dataframe,
    clean_series,
    clean_string,
    cleaned,
)
from semanticlog.data.cleaning.renames import WORDS, RenameKind, RenameLog
from semanticlog.data.cleaning.rewrite import attribute_names, entity_ids, text_values
from semanticlog.data.integrity import check_missing_ids
from semanticlog.data.oced import OCEDStore
from semanticlog.data.readable import ReadableStore
from semanticlog.data.tables import (
    OCEL_PREFIX,
    EntityKind,
    EntityTable,
    RelationKind,
    Relations,
    RelationTable,
)

__all__ = [
    "clean_oced_data",
    "fix_duplicate_names",
    "object_type_renames",
]

logger = logging.getLogger(__name__)

_RENAME_KIND = {
    EntityKind.EVENT: RenameKind.ACTIVITY,
    EntityKind.OBJECT: RenameKind.OBJECT_TYPE,
}


def clean_oced_data(
    data: OCEDStore,
    bin_labels: Collection[str] = (),
) -> OCEDStore:
    """Make every name in a store safe to emit, and record what changed.

    Sets `data.backmap` to the map of new name to source name.

    Args:
        data: Store to clean, modified in place.
        bin_labels: Bin labels the build will use. Each bin class name is
            reserved.

    Returns:
        The same store.

    Raises:
        InputError: If an event or object has no id, or if two schema names
            are the same after every rename has run.
    """
    logger.info("Cleaning data")
    raise_reports([check_missing_ids(data)])
    log = RenameLog()
    resolve_merged_names(data, log)

    _record_id_renames(data, log)
    _record_value_renames(data, log)
    _record_attribute_renames(data, log)

    data.replace_tables(
        entities={kind: _clean_entities(data, kind, log) for kind in EntityKind},
        relations=_clean_relations(data, log),
    )
    if data.ocel_mode:
        data.replace_object_changes(
            {
                clean_string(obj_type): _clean_changes(
                    data.get_object_changes(obj_type)
                )
                for obj_type in data.get_object_change_types()
            }
        )
    fix_duplicate_names(data, log, bin_labels)
    verify_unique_names(data)
    data.backmap = log.flat()
    _report_renames(log)
    return data


def fix_duplicate_names(
    data: OCEDStore,
    log: RenameLog | None = None,
    bin_labels: Collection[str] = (),
) -> OCEDStore:
    """Give every schema name in a clean store one owner, and rename the rest.

    Ids and values are not touched. Each kind of individual has a path of its
    own in the ontology, so they collide with nothing but each other.

    Args:
        data: Store to fix, modified in place. Its names must be safe.
        log: Rename log to append to. A new one is used when none is given.
        bin_labels: Bin labels the build will use. Each bin class name is
            reserved.

    Returns:
        The same store.
    """
    log = log if log is not None else RenameLog()
    allocate_schema_names(data, log, bin_labels)
    return data


def verify_unique_names(data: OCEDStore) -> None:
    """Raise if two schema names in the store are the same after every rename has run.

    Raises:
        InputError: Listing every name claimed more than once.
    """
    claims: dict[str, list[str]] = {}

    def claim(name: str, description: str) -> None:
        claims.setdefault(name, []).append(description)

    for activity in data.types(EntityKind.EVENT):
        claim(activity, f"activity {activity!r}")
    for obj_type in data.types(EntityKind.OBJECT):
        claim(obj_type, f"object type {obj_type!r}")
    for attribute in sorted(attribute_names(data)):
        claim(attribute, f"attribute {attribute!r}")
    for relation in data.get_relation_types():
        claim(relation, f"relation {relation!r}")

    errors = [
        f"  {name!r} is claimed {len(owners)} times: {', '.join(sorted(owners))}"
        for name, owners in claims.items()
        if len(owners) > 1
    ]
    if not errors:
        return
    raise InputError(
        "These names are claimed more than once:\n"
        + "\n".join(errors)
        + "\n\nRename the duplicate in the input data."
    )


def _record_id_renames(data: OCEDStore, log: RenameLog) -> None:
    """Record the event and object ids that cleaning is about to rewrite."""
    ids = entity_ids(data)
    if ids.empty:
        return
    clean = clean_series(ids)
    for value, cleaned_value in zip(
        ids[clean != ids], clean[clean != ids], strict=True
    ):
        log.record(RenameKind.ID, value, cleaned_value)


def _record_value_renames(data: OCEDStore, log: RenameLog) -> None:
    """Record the attribute values that cleaning is about to rewrite."""
    for view in data.tables():
        for column in view.attributes:
            cleaned_column = cleaned(column)
            for value in text_values(view, column):
                cleaned_value = clean_string(value)
                if cleaned_value != value:
                    log.record_value(cleaned_column, cleaned_value, value)


def _record_attribute_renames(data: OCEDStore, log: RenameLog) -> None:
    """Record the attribute columns that cleaning is about to rename.

    An attribute spelled two ways gets no source name.
    """
    spellings: dict[str, dict[str, list[str]]] = {}
    for view in data.tables(changes=False):
        for column in view.attributes:
            by_source = spellings.setdefault(cleaned(column), {})
            by_source.setdefault(column, []).append(view.origin)
    for name, by_source in spellings.items():
        if len(by_source) > 1:
            continue
        (source,) = by_source
        if source != name:
            log.record(RenameKind.ATTRIBUTE, source, name)


def _report_renames(log: RenameLog) -> None:
    """Log what cleaning renamed, as counts and then the names themselves."""
    report_by_kind(logger, WORDS, log.by_kind(), lead="renamed")


def object_type_renames(data: ReadableStore) -> dict[str, str]:
    """Return source name to current name for every renamed object type."""
    current = set(data.types(EntityKind.OBJECT))
    return {
        original: name for name, original in data.backmap.items() if name in current
    }


def _clean_changes(changes: pd.DataFrame) -> pd.DataFrame:
    """Return a change table with its `ocel:` column names kept and everything else cleaned."""
    kept = [str(c) for c in changes.columns if str(c).startswith(OCEL_PREFIX)]
    return clean_dataframe(changes, keep=kept)


def _clean_entities(
    data: OCEDStore, kind: EntityKind, log: RenameLog
) -> dict[str, EntityTable]:
    """Return the tables of one entity kind with type, column and cell names cleaned."""
    tables = {}
    for name in data.types(kind):
        table = data.table(kind, name)
        clean_name = clean_string(name)
        if clean_name != name:
            log.record(_RENAME_KIND[kind], name, clean_name)
        tables[clean_name] = EntityTable(
            id=clean_string(table.id),
            data=clean_dataframe(table.data),
            time=None if table.time is None else clean_string(table.time),
        )
    return tables


def _clean_relations(data: OCEDStore, log: RenameLog) -> Relations:
    """Return both relation families with their names and endpoints cleaned."""

    def clean_types(types: list[str] | None) -> list[str] | None:
        if types is None:
            return None
        return [cleaned(t) for t in types]

    def clean_family(kind: RelationKind) -> dict[str, RelationTable]:
        family = data.relations_of(kind)
        cleaned_family = {}
        for name, rel in family.items():
            clean_name = clean_string(name)
            if clean_name != name:
                log.record(RenameKind.RELATION, name, clean_name)
            to_type = clean_types(rel.to_type)
            cleaned_family[clean_name] = RelationTable(
                from_id=clean_string(rel.from_id),
                from_type=clean_types(rel.from_type),
                to_id=clean_string(rel.to_id),
                to_type=to_type if to_type is not None else [],
                data=clean_dataframe(rel.data),
            )
        return cleaned_family

    return Relations({kind: clean_family(kind) for kind in RelationKind.families()})
