"""Making the names in a log safe to emit.

Every activity, object type, relation, attribute, id and text value in a log
becomes part of an IRI, so anything outside `[A-Za-z0-9_]` has to go first.
Cleaning does that, and then resolves the collisions it can cause: two source
names that clean to one name, and names that clash with the ones the build
reserves for its own schema. Each rename is recorded, so the ontology can
still be annotated with what the source called things.

`clean_oced_data` is the entry point, and it has to run before any build.
"""

import logging
import re
from collections.abc import Callable, Collection
from enum import Enum
from typing import Any, cast

import numpy as np
import pandas as pd

from semanticlog.core import vocabulary as vocab
from semanticlog.data.oced import (
    EventTable,
    ObjectTable,
    OCEDStore,
    RelationKind,
    Relations,
    RelationTable,
)

logger = logging.getLogger(__name__)

_ISO8601_RE = re.compile(
    r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2})?"
    r"(?:\.\d+)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?)?$"
)

_RESERVED_OWL_NAMES: frozenset[str] = vocab.RESERVED_NAMES | {"is_a"}

_MAX_REPORTED_COLLISIONS = 20


def is_timestamp(string: str) -> bool:
    """Report whether text looks like an ISO 8601 date or timestamp.

    Timestamps are exempt from cleaning, because rewriting their punctuation
    would destroy the format that later parsing depends on.
    """
    return bool(_ISO8601_RE.match(string))


def clean_string(string: str, casefold: bool = True) -> str:
    """Replace every character that cannot appear in an IRI local name.

    Args:
        string: Text to clean.
        casefold: Also lower-case the result, which merges names that differ
            only in case.

    Returns:
        The cleaned text, with each unsafe character replaced by an underscore.
    """
    string = re.sub(r"[^A-Za-z0-9_]", "_", string)
    if casefold:
        string = string.casefold()
    return string


def _apply_to_tree(
    value: Any,
    fn: Callable[[str], Any],
    df_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> Any:
    """Apply a cleaner to every string inside a nested value."""
    if isinstance(value, dict):
        return {
            _apply_to_tree(k, fn, df_fn): _apply_to_tree(v, fn, df_fn)
            for k, v in value.items()
        }
    if isinstance(value, str):
        return value if is_timestamp(value) else fn(value)
    if isinstance(value, pd.DataFrame):
        return df_fn(value) if df_fn is not None else value
    if isinstance(value, (list, pd.arrays.StringArray, np.ndarray)):
        return [_apply_to_tree(x, fn, df_fn) for x in value]
    return value


def clean_value(
    value: dict | str | int | float | bool | pd.DataFrame, casefold: bool = True
) -> dict | str | int | float | bool | pd.DataFrame:
    """Clean every string in a value, whatever shape it has.

    Cells in an OCEL log can hold nested structures as well as plain values,
    so dicts, lists, arrays and frames are all walked. Numbers, booleans and
    timestamps are returned unchanged.

    Args:
        value: Value to clean.
        casefold: Also lower-case the cleaned strings.

    Returns:
        A cleaned value of the same shape.
    """
    return _apply_to_tree(
        value,
        lambda s: clean_string(s, casefold=casefold),
        df_fn=lambda df: clean_dataframe(df, casefold=casefold),
    )


def clean_dataframe(
    df: pd.DataFrame, clean_columns: bool = True, casefold: bool = True
) -> pd.DataFrame:
    """Clean the text cells of a table, and optionally its column names.

    Numeric, boolean and datetime columns are left alone, because their values
    never reach an IRI as text.

    Args:
        df: Table to clean.
        clean_columns: Also clean the column names, which become attribute
            names in the ontology.
        casefold: Also lower-case the cleaned strings.

    Returns:
        A new table. The input is not modified.
    """
    clean_df = df.copy()
    text_columns = [
        column
        for column in clean_df.columns
        if not pd.api.types.is_numeric_dtype(clean_df[column])
        and not pd.api.types.is_datetime64_any_dtype(clean_df[column])
        and not pd.api.types.is_bool_dtype(clean_df[column])
    ]
    for column in text_columns:
        clean_df[column] = clean_df[column].map(
            lambda v: clean_value(v, casefold=casefold)
        )
    if clean_columns:
        col_map = {
            col_str: clean_string(col_str, casefold=casefold)
            for col_str in clean_df.columns
        }
        clean_df = clean_df.rename(columns=col_map)
    return clean_df


def clean_object_changes(df: pd.DataFrame, casefold: bool = False) -> pd.DataFrame:
    """Clean an attribute change table, leaving its OCEL columns intact.

    The `ocel:` columns carry structure, so their names are kept. Their
    contents are still cleaned where they name an object, a type or a field,
    since those become part of an IRI.

    Args:
        df: Change table to clean.
        casefold: Also lower-case the cleaned strings.

    Returns:
        A new table. The input is not modified.
    """
    df = df.copy()

    if "ocel:oid" in df.columns:
        df["ocel:oid"] = df["ocel:oid"].apply(lambda v: _clean_str_value(v, casefold))

    if "ocel:type" in df.columns:
        df["ocel:type"] = df["ocel:type"].apply(lambda v: _clean_str_value(v, casefold))

    system_cols = {c for c in df.columns if c.startswith("ocel:")}
    attr_cols = [c for c in df.columns if c not in system_cols]

    if not attr_cols:
        return df

    rename_map = {c: clean_string(c, casefold=casefold) for c in attr_cols}
    df = df.rename(columns=rename_map)

    if "ocel:field" in df.columns:
        df["ocel:field"] = df["ocel:field"].apply(
            lambda v: _clean_str_value(v, casefold)
        )

    for new_col in rename_map.values():
        df[new_col] = df[new_col].apply(
            lambda v: (
                clean_string(str(v), casefold=casefold)
                if isinstance(v, str) and not is_timestamp(str(v))
                else v
            )
        )

    return df


def _clean_str_value(v: Any, casefold: bool) -> Any:
    """Clean a cell if it holds text, and pass anything else through."""
    return clean_string(str(v), casefold=casefold) if isinstance(v, str) else v


def _rename_attr_val(value: Any, reserved: Collection[str]) -> Any:
    """Suffix a value that would otherwise collide with a schema name."""
    return value + "_attr_val" if value in reserved else value


def _collect_entity_ids(data: OCEDStore) -> set[str]:
    """Collect every event and object id in the store."""
    ids: set[str] = set()
    for activity in data.get_event_types():
        table = data.get_event_table(activity)
        ids.update(str(v) for v in table.data[table.id].unique())
    for obj_type in data.get_object_types():
        obj = data.get_object_table(obj_type)
        ids.update(str(v) for v in obj.data[obj.id].unique())
    return ids


def _collect_attribute_names(data: OCEDStore) -> set[str]:
    """Collect every attribute column of events and objects."""
    names: set[str] = set()
    for activity in data.get_event_types():
        table = data.get_event_table(activity)
        names |= set(table.data.columns) - {table.id, table.time}
    for obj_type in data.get_object_types():
        obj = data.get_object_table(obj_type)
        names |= set(obj.data.columns) - {obj.id}
    return names


def _collect_schema_names(data: OCEDStore) -> set[str]:
    """Collect every name in the store that becomes a class or a property."""
    names: set[str] = set()
    names.update(data.get_event_types())
    names.update(data.get_object_types())
    names.update(data.get_relation_types())
    for activity in data.get_event_types():
        table = data.get_event_table(activity)
        names.update(set(table.data.columns) - {table.id, table.time})
    for obj_type in data.get_object_types():
        obj = data.get_object_table(obj_type)
        names.update(set(obj.data.columns) - {obj.id})
    return names


def _check_name_collisions(data: OCEDStore, casefold: bool) -> None:
    """Raise if two names in one group would clean to the same name.

    Names are only compared within a group, such as the ids of one activity,
    because that is the scope in which they have to stay distinct.
    """
    groups: dict[str, list[str]] = {
        "activities": data.get_event_types(),
        "object types": data.get_object_types(),
        "e2o relations": data.get_relation_types(RelationKind.E2O),
        "o2o relations": data.get_relation_types(RelationKind.O2O),
    }
    for activity in data.get_event_types():
        table = data.get_event_table(activity)
        groups[f"event ids of activity {activity!r}"] = [
            str(v) for v in table.data[table.id].dropna().unique()
        ]
    for obj_type in data.get_object_types():
        obj = data.get_object_table(obj_type)
        groups[f"object ids of type {obj_type!r}"] = [
            str(v) for v in obj.data[obj.id].dropna().unique()
        ]
    for activity in data.get_event_types():
        table = data.get_event_table(activity)
        groups[f"attributes of activity {activity!r}"] = [
            str(c) for c in table.data.columns if c not in (table.id, table.time)
        ]
    for obj_type in data.get_object_types():
        obj = data.get_object_table(obj_type)
        groups[f"attributes of object type {obj_type!r}"] = [
            str(c) for c in obj.data.columns if c != obj.id
        ]

    errors: list[str] = []
    for group_name, names in groups.items():
        seen: dict[str, str] = {}
        for name in names:
            cleaned = clean_string(str(name), casefold=casefold)
            if cleaned in seen:
                errors.append(
                    f"  {group_name}: {name!r} and {seen[cleaned]!r} "
                    f"both clean to {cleaned!r}"
                )
            else:
                seen[cleaned] = str(name)

    if not errors:
        return
    shown = errors[:_MAX_REPORTED_COLLISIONS]
    more = len(errors) - len(shown)
    raise ValueError(
        "clean_oced_data: the following names collide after cleaning "
        "(cannot preserve both):\n"
        + "\n".join(shown)
        + (f"\n  ... and {more} more" if more else "")
        + "\n\nRename the colliding source in the input data. Two ids differing "
        "only by punctuation, such as 'a-1' and 'a_1', are the usual cause."
    )


class RenameKind(str, Enum):
    """What sort of name a rename applies to."""

    ACTIVITY = "activity"
    OBJECT_TYPE = "object_type"
    RELATION = "relation"
    ATTRIBUTE = "attribute"
    ID = "id"
    VALUE = "value"


class RenameLog:
    """Records what each cleaned name was called in the source.

    A name can be rewritten more than once, first by cleaning and then again
    to resolve a collision. The log always keeps the source name, not the
    intermediate one.
    """

    def __init__(self) -> None:
        self._original: dict[RenameKind, dict[str, str]] = {k: {} for k in RenameKind}
        self._current: dict[RenameKind, dict[str, str]] = {k: {} for k in RenameKind}

    def record(
        self, kind: RenameKind, old: str, new: str, original: str | None = None
    ) -> None:
        """Record that a name changed.

        Args:
            kind: What sort of name changed.
            old: The name before this change.
            new: The name after it.
            original: The source name to record, when it is neither `old` nor
                already known from an earlier rename of `old`.
        """
        root = original if original is not None else self._current[kind].get(old, old)
        self._current[kind].pop(old, None)
        self._original[kind].pop(old, None)
        if new != root:
            self._original[kind][new] = root
        self._current[kind][new] = root

    def of_kind(self, kind: RenameKind) -> dict[str, str]:
        """Return the renames of one kind, as current name to source name."""
        return dict(self._original[kind])

    def flat(self) -> dict[str, str]:
        """Return every rename in one map, as current name to source name."""
        merged: dict[str, str] = {}
        for kind in RenameKind:
            merged.update(self._original[kind])
        return merged


def _record_value_renames(
    data: OCEDStore, log: RenameLog, casefold: bool = False
) -> None:
    """Record the attribute values that cleaning is about to rewrite.

    Values are recorded under the individual name they will have, since that
    is the name an annotation will later have to look up.
    """

    def _collect(df: pd.DataFrame, skip: set[str]) -> None:
        for col in df.columns:
            if col in skip:
                continue
            cleaned_col = clean_string(str(col), casefold=casefold)
            for value in df[col].dropna().unique():
                if not isinstance(value, str) or is_timestamp(value):
                    continue
                cleaned_val = clean_string(value, casefold=casefold)
                if cleaned_val == value:
                    continue
                log.record(
                    RenameKind.VALUE,
                    value,
                    vocab.attribute_value(cleaned_col, cleaned_val),
                )

    for activity in data.get_event_types():
        table = data.get_event_table(activity)
        _collect(table.data, {table.id, table.time})
    for obj_type in data.get_object_types():
        obj = data.get_object_table(obj_type)
        _collect(obj.data, {obj.id})


def _record_id_renames(data: OCEDStore, log: RenameLog, casefold: bool = False) -> None:
    """Record the event and object ids that cleaning is about to rewrite."""
    for value in _collect_entity_ids(data):
        cleaned = clean_string(value, casefold=casefold)
        if cleaned != value:
            log.record(RenameKind.ID, value, cleaned)


def _free_name(name: str, taken: Collection[str]) -> str:
    """Append underscores until the name is unused."""
    candidate = name + "_"
    while candidate in taken:
        candidate += "_"
    return candidate


def fix_duplicate_names(data: OCEDStore, log: RenameLog | None = None) -> OCEDStore:
    """Rename anything that would collide with a name the build reserves.

    The reserved set covers the upper schema and every name the build will
    derive from this log, such as `has_price` for an attribute or `df_order`
    for a case concept. Relations, activities and object types are renamed
    with a trailing underscore, attributes with an `_attr` suffix, and
    colliding text values with an `_attr_val` suffix.

    Args:
        data: Store to fix, modified in place.
        log: Rename log to append to. A new one is used when none is given.

    Returns:
        The same store.
    """
    log = log if log is not None else RenameLog()
    attribute_names = _collect_attribute_names(data)
    names: set[str] = set(_RESERVED_OWL_NAMES) | vocab.derived_schema_names(
        data.get_object_types(), attribute_names
    )

    for relation_type in data.get_relation_types():
        if relation_type in names:
            free = _free_name(relation_type, names | set(data.get_relation_types()))
            logger.info(f"  renaming relation {relation_type} to {free}..")
            data.rename_relation(relation_type, free)
            log.record(RenameKind.RELATION, relation_type, free)
            names.add(free)
        else:
            names.add(relation_type)

    for activity in data.get_event_types():
        if activity in names:
            free = _free_name(activity, names | set(data.get_event_types()))
            logger.info(f"  renaming activity {activity} to {free}..")
            data.rename_activity(activity, free)
            log.record(RenameKind.ACTIVITY, activity, free)
            names.add(free)
        else:
            names.add(activity)

    for object_type in data.get_object_types():
        if object_type in names:
            free = _free_name(object_type, names | set(data.get_object_types()))
            logger.info(f"  renaming object {object_type} to {free}..")
            data.rename_object(object_type, free)
            log.record(RenameKind.OBJECT_TYPE, object_type, free)
            names.add(free)
        else:
            names.add(object_type)

    current_activities = list(data.get_event_types())
    current_object_types = list(data.get_object_types())

    names_casefolded = {n.casefold() for n in names}
    attributes_names_overlap = [
        a for a in attribute_names if a.casefold() in names_casefolded
    ]
    for attribute in attributes_names_overlap:
        taken = names | attribute_names
        candidate = attribute + "_attr"
        if candidate in taken:
            candidate = _free_name(candidate, taken)
        renaming = {attribute: candidate}
        log.record(RenameKind.ATTRIBUTE, attribute, candidate)
        names.add(candidate)
        attribute_names.add(candidate)
        for activity in current_activities:
            df = data.get_event_table(activity).data
            if attribute in df.columns:
                data.replace_event_data(activity, df.rename(columns=renaming))
        for object_type in current_object_types:
            df = data.get_object_table(object_type).data
            if attribute in df.columns:
                data.replace_object_data(object_type, df.rename(columns=renaming))

    all_names = frozenset(names | attribute_names)

    def rename_values(
        df: pd.DataFrame,
        key_cols: set[str],
        schema_cols: Collection[str] | None = None,
    ) -> pd.DataFrame:
        """Suffix the cell values that collide with a schema name."""
        skip = set(schema_cols or ())
        for column in df.columns:
            if column in skip:
                continue
            collided = [
                v
                for v in df[column].dropna().unique()
                if isinstance(v, str) and v in all_names
            ]
            for value in collided:
                if column in key_cols:
                    log.record(RenameKind.ID, value, value + "_attr_val")
                else:
                    before = vocab.attribute_value(column, value)
                    known = log.of_kind(RenameKind.VALUE).get(before)
                    log.record(
                        RenameKind.VALUE, before, before + "_attr_val", known or value
                    )

        renamed = df.copy()
        for column in df.columns:
            if column in skip or pd.api.types.is_numeric_dtype(df[column]):
                continue
            if pd.api.types.is_datetime64_any_dtype(df[column]):
                continue
            renamed[column] = df[column].map(lambda v: _rename_attr_val(v, all_names))
        return renamed

    for activity in current_activities:
        table = data.get_event_table(activity)
        data.replace_event_data(
            activity, rename_values(table.data, {table.id, table.time})
        )
    for object_type in current_object_types:
        obj = data.get_object_table(object_type)
        data.replace_object_data(object_type, rename_values(obj.data, {obj.id}))

    _rename_relation_values(data, rename_values)

    if data.ocel_mode:
        data.replace_object_changes(
            {
                obj_type: rename_values(
                    data.get_object_changes(obj_type),
                    key_cols={"ocel:oid"},
                    schema_cols=_OCEL_SCHEMA_COLUMNS,
                )
                for obj_type in data.get_object_change_types()
            }
        )

    return data


_OCEL_SCHEMA_COLUMNS = frozenset({"ocel:type", "ocel:field", "ocel:timestamp"})


def _rename_relation_values(
    data: OCEDStore,
    rename_values: Callable[..., pd.DataFrame],
) -> None:
    """Apply the value renaming to both relation families."""
    for family in (data.get_e2o(), data.get_o2o()):
        for table in family.values():
            id_columns = {table.to_id}
            if table.from_id is not None:
                id_columns.add(table.from_id)
            table.data = rename_values(table.data, id_columns)


def clean_oced_data(data: OCEDStore, casefold: bool = False) -> OCEDStore:
    """Make every name in a store safe to emit, and record what changed.

    This runs before any build. It rejects names that would clean to the same
    name, rewrites the unsafe characters out of schema names, ids and text
    values, resolves collisions with the names the build reserves, and leaves
    the resulting map of new name to source name on the store as its backmap.

    Args:
        data: Store to clean, modified in place.
        casefold: Also lower-case every cleaned name. This merges names that
            differ only in case, so the collision check runs against the
            folded form as well.

    Returns:
        The same store.

    Raises:
        ValueError: If two names in one group clean to the same name, which
            no rename could resolve without choosing one of them.
    """
    logger.info("Cleaning data..")
    _check_name_collisions(data, casefold)

    log = RenameLog()
    _record_id_renames(data, log, casefold=casefold)
    _record_value_renames(data, log, casefold=casefold)

    data.replace_tables(
        events=_clean_events(data, casefold, log),
        objects=_clean_objects(data, casefold, log),
        relations=_clean_relations(data, casefold, log),
    )
    if data.ocel_mode:
        data.replace_object_changes(
            {
                clean_string(obj_type, casefold=casefold): clean_object_changes(
                    data.get_object_changes(obj_type), casefold=casefold
                )
                for obj_type in data.get_object_change_types()
            }
        )
    data = fix_duplicate_names(data, log)
    data.update_used_ids()
    data.backmap = log.flat()
    return data


def _clean_columns(
    df: pd.DataFrame, skip: set[str], log: RenameLog, casefold: bool
) -> None:
    """Record the attribute columns that cleaning is about to rename."""
    for column in df.columns:
        if column in skip:
            continue
        cleaned = clean_string(str(column), casefold=casefold)
        if cleaned != str(column):
            log.record(RenameKind.ATTRIBUTE, str(column), cleaned)


def _clean_events(
    data: OCEDStore, casefold: bool, log: RenameLog
) -> dict[str, EventTable]:
    """Return the event tables with activity, column and cell names cleaned."""
    events = {}
    for activity in data.get_event_types():
        table = data.get_event_table(activity)
        cleaned = clean_string(activity, casefold=casefold)
        if cleaned != activity:
            log.record(RenameKind.ACTIVITY, activity, cleaned)
        _clean_columns(table.data, {table.id, table.time}, log, casefold)
        events[cleaned] = EventTable(
            id=clean_string(table.id, casefold=casefold),
            time=clean_string(table.time, casefold=casefold),
            data=clean_dataframe(table.data, casefold=casefold),
        )
    return events


def _clean_objects(
    data: OCEDStore, casefold: bool, log: RenameLog
) -> dict[str, ObjectTable]:
    """Return the object tables with type, column and cell names cleaned."""
    objects = {}
    for obj_type in data.get_object_types():
        obj = data.get_object_table(obj_type)
        cleaned = clean_string(obj_type, casefold=casefold)
        if cleaned != obj_type:
            log.record(RenameKind.OBJECT_TYPE, obj_type, cleaned)
        _clean_columns(obj.data, {obj.id}, log, casefold)
        objects[cleaned] = ObjectTable(
            id=clean_string(obj.id, casefold=casefold),
            data=clean_dataframe(obj.data, casefold=casefold),
        )
    return objects


def _clean_relations(data: OCEDStore, casefold: bool, log: RenameLog) -> Relations:
    """Return both relation families with their names and endpoints cleaned."""

    def clean_types(types: list[str] | None) -> list[str] | None:
        if types is None:
            return None
        return [clean_string(str(t), casefold=casefold) for t in types]

    def clean_family(kind: RelationKind) -> dict[str, RelationTable]:
        family = data.get_e2o() if kind is RelationKind.E2O else data.get_o2o()
        cleaned_family = {}
        for name, rel in family.items():
            cleaned = clean_string(name, casefold=casefold)
            if cleaned != name:
                log.record(RenameKind.RELATION, name, cleaned)
            cleaned_family[cleaned] = RelationTable(
                from_id=clean_string(rel.from_id, casefold=casefold),
                from_type=clean_types(rel.from_type),
                to_id=clean_string(rel.to_id, casefold=casefold),
                to_type=cast(list, clean_types(rel.to_type)),
                data=clean_dataframe(rel.data, casefold=casefold),
            )
        return cleaned_family

    return Relations(
        event_to_object=clean_family(RelationKind.E2O),
        object_to_object=clean_family(RelationKind.O2O),
    )
