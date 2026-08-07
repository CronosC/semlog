import logging
import re
from collections.abc import Callable, Collection
from typing import Any, cast

import numpy as np
import pandas as pd

from semanticlog.data.oced import (
    EventTable,
    ObjectTable,
    OCEDStore,
    RelationKind,
    Relations,
    RelationTable,
)

_ISO8601_RE = re.compile(
    r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2})?"
    r"(?:\.\d+)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?)?$"
)

_RESERVED_OWL_NAMES: frozenset[str] = frozenset(
    {
        "is_a",
        "observes",
        "observed_at",
        "df",
        "ef",
        "has_attribute",
        "attribute_of",
        "event_object_relation",
        "object_object_relation",
        "Starts",
        "Ends",
    }
)


def is_timestamp(string: str) -> bool:
    """Return True if string is a strict ISO 8601 date or datetime, otherwise False."""
    return bool(_ISO8601_RE.match(string))


def clean_string(string: str, casefold: bool = True) -> str:
    """Replace special characters with underscores; optionally casefold."""
    string = re.sub(r"[^A-Za-z0-9_]", "_", string)
    if casefold:
        string = string.casefold()
    return string


def _apply_to_tree(
    value: Any,
    fn: Callable[[str], Any],
    df_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> Any:
    """Recursively apply fn to all non-timestamp string leaves in a nested structure.

    Args:
        fn: Applied to each non-timestamp string.
        df_fn: Applied to DataFrames when provided; DataFrames are returned unchanged if omitted.
    """
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
    return value  # int, float, bool, pd.Timestamp, None, pd.NA


def apply_function_to_value(
    value: dict | str | int | float | bool | pd.DataFrame,
    function: Callable[..., Any],
    rename_df_columns: bool = True,
    replace_df_values: bool = True,
) -> Any:
    """Apply function to all string leaves; for DataFrames, optionally rename columns and/or map cell values.

    Args:
        value: Input value — may be a nested dict, list, DataFrame, or scalar.
        function: Called on each non-timestamp string leaf.
        rename_df_columns: Whether to apply function to DataFrame column names.
        replace_df_values: Whether to apply function to DataFrame cell values.
    """
    if isinstance(value, pd.DataFrame):
        if replace_df_values:
            value = value.map(function)
        if rename_df_columns:
            value = value.rename(
                columns={c: _apply_to_tree(c, function) for c in value.columns}
            )
        return value
    return _apply_to_tree(value, function)


def clean_value(
    value: dict | str | int | float | bool | pd.DataFrame, casefold: bool = True
) -> dict | str | int | float | bool | pd.DataFrame:
    """Apply string cleaning to a value, handling all types appropriately."""
    return _apply_to_tree(
        value,
        lambda s: clean_string(s, casefold=casefold),
        df_fn=lambda df: clean_dataframe(df, casefold=casefold),
    )


def clean_dataframe(
    df: pd.DataFrame, clean_columns: bool = True, casefold: bool = True
) -> pd.DataFrame:
    """Apply string cleaning to all cell values and optionally column names."""
    clean_df = df.map(lambda v: clean_value(v, casefold=casefold))
    if clean_columns:
        col_map = {
            col_str: clean_string(col_str, casefold=casefold)
            for col_str in clean_df.columns
        }
        clean_df = clean_df.rename(columns=col_map)
    return clean_df


def clean_object_changes_df(df: pd.DataFrame, casefold: bool = False) -> pd.DataFrame:
    """Clean attribute field names and values in an object-changes DataFrame.

    ``ocel:*`` column names are never renamed so downstream code keeps working.
    ``ocel:oid`` values are cleaned via :func:`clean_string` to stay consistent
    with how object IDs are cleaned in ``_objects`` by :func:`clean_oced_data`.
    ``ocel:field`` values are cleaned to match the renamed attribute columns.
    Attribute columns (all non-``ocel:*`` columns) are renamed and their string
    cell values are cleaned.  Timestamps are preserved unchanged.
    """
    df = df.copy()

    # Object IDs must match the cleaned IDs asserted by instantiate_object_data()
    if "ocel:oid" in df.columns:
        df["ocel:oid"] = df["ocel:oid"].apply(lambda v: _clean_str_value(v, casefold))

    # Object type names must match cleaned keys in _objects
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


def clean_column_values(log: pd.DataFrame, to_clean: dict) -> pd.DataFrame:
    """Apply column-specific functions to a DataFrame; raise if a column is missing."""
    for column, function in to_clean.items():
        if column in log.columns:
            log[column] = log[column].apply(function)
        else:
            raise ValueError(f"Column '{column}' not found in log DataFrame")
    return log


def _clean_str_value(v: Any, casefold: bool) -> Any:
    return clean_string(str(v), casefold=casefold) if isinstance(v, str) else v


def _rename_attr_val(value: Any, reserved: Collection[str]) -> Any:
    """Append ``_attr_val`` suffix to any string value that collides with a reserved name."""
    return value + "_attr_val" if value in reserved else value


def _log_null_counts(label: str, df: pd.DataFrame) -> None:
    for col in df.columns:
        n_null = df[col].isnull().sum()
        if n_null > 0:
            logging.info(f"{label}[{col}]: {n_null} null values.")


def null_check(data: OCEDStore) -> None:
    for activity in data.get_activities():
        _log_null_counts(activity, data.get_events(activity).data)
    for object_type in data.get_object_types():
        _log_null_counts(object_type, data.get_object(object_type).data)


def _collect_entity_ids(data: OCEDStore) -> set[str]:
    """Return all current event and object ID values as strings."""
    ids: set[str] = set()
    for activity in data.get_activities():
        table = data.get_events(activity)
        ids.update(str(v) for v in table.data[table.id].unique())
    for obj_type in data.get_object_types():
        obj = data.get_object(obj_type)
        ids.update(str(v) for v in obj.data[obj.id].unique())
    return ids


def _collect_schema_names(data: OCEDStore) -> set[str]:
    """Return all current schema-level names that become OWL IRI fragments."""
    names: set[str] = set()
    names.update(data.get_activities())
    names.update(data.get_object_types())
    names.update(data.get_relation_types())
    for activity in data.get_activities():
        table = data.get_events(activity)
        names.update(set(table.data.columns) - {table.id, table.time})
    for obj_type in data.get_object_types():
        obj = data.get_object(obj_type)
        names.update(set(obj.data.columns) - {obj.id})
    return names


def _check_schema_name_collisions(data: OCEDStore, casefold: bool) -> None:
    """Raise ValueError if any schema-level names would collide after cleaning.

    Called before mutation so the store is never left in a partially-cleaned state.
    """
    groups: dict[str, list[str]] = {
        "activities": data.get_activities(),
        "object types": data.get_object_types(),
        "e2o relations": data.get_relation_types(RelationKind.E2O),
        "o2o relations": data.get_relation_types(RelationKind.O2O),
    }
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
    if errors:
        raise ValueError(
            "clean_oced_data: the following schema names collide after cleaning "
            "(cannot preserve both):\n" + "\n".join(errors)
        )


def _build_backmap(
    pre: set[str], post: set[str], casefold: bool = False
) -> dict[str, str]:
    """Build cleaned_name → original_name mapping from pre- and post-clean snapshots.

    Covers both entity IDs and schema names (activities, object types, attributes).
    Predicts the post-clean form of each pre-clean name via clean_string(), then checks
    for suffixes that fix_duplicate_names() may append on reserved-name collisions:
    ``_attr_val`` (entity ID values), ``_attr`` (attribute columns), ``_`` (activities,
    objects, relations).  Names unchanged by cleaning are omitted.
    """
    backmap: dict[str, str] = {}
    for pre_name in pre:
        predicted = clean_string(str(pre_name), casefold=casefold)
        if predicted in post:
            if predicted != str(pre_name):
                backmap[predicted] = str(pre_name)
        elif predicted + "_attr_val" in post:
            backmap[predicted + "_attr_val"] = str(pre_name)
        elif predicted + "_attr" in post:
            backmap[predicted + "_attr"] = str(pre_name)
        elif predicted + "_" in post:
            backmap[predicted + "_"] = str(pre_name)
    return backmap


def _collect_attr_value_backmap(
    data: OCEDStore, casefold: bool = False
) -> dict[str, str]:
    """Map attribute-value individual local names to their original cell values.

    Snapshots the pre-clean string attribute cell values (the same leaves
    :func:`clean_value` would alter) and predicts each value individual's local
    name ``{cleaned_column}_{cleaned_value}`` — matching the ``value_to_id``
    convention used in :mod:`semanticlog.onto.instantiate`.  Only values whose
    cleaned form differs from the original are recorded; unchanged values fall
    back to the cleaned text at annotation time.  Must be called *before*
    cleaning mutates the store.
    """
    backmap: dict[str, str] = {}

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
                backmap[f"{cleaned_col}_{cleaned_val}"] = value

    for activity in data.get_activities():
        table = data.get_events(activity)
        _collect(table.data, {table.id, table.time})
    for obj_type in data.get_object_types():
        obj = data.get_object(obj_type)
        _collect(obj.data, {obj.id})

    return backmap


def fix_duplicate_names(data: OCEDStore) -> OCEDStore:
    """Rename entities whose names collide with reserved OWL property/class names.

    Checks activities, object types, relation names, attribute column names, and
    attribute cell values against ``_RESERVED_OWL_NAMES`` (plus names accumulated
    during the pass).  Appends ``_`` to entity names, ``_attr`` to attribute
    columns, and ``_attr_val`` to cell values that collide.

    Args:
        data: OCEDStore to fix in-place (also returned for chaining).

    Returns:
        The same OCEDStore with collisions resolved.
    """
    names: set[str] = set(_RESERVED_OWL_NAMES)
    attribute_names: set[str] = set()

    for relation_type in data.get_relation_types():
        if relation_type in names:
            logging.info(f"  renaming relation {relation_type} to avoid collision..")
            data.rename_relation(relation_type, relation_type + "_")
            names.add(relation_type + "_")
        else:
            names.add(relation_type)

    for activity in data.get_activities():
        if activity in names:
            logging.info(f"  renaming activity {activity} to avoid collision..")
            data.rename_activity(activity, activity + "_")
            names.add(activity + "_")
        else:
            names.add(activity)

    for object_type in data.get_object_types():
        if object_type in names:
            logging.info(f"  renaming object {object_type} to avoid collision..")
            data.rename_object(object_type, object_type + "_")
            names.add(object_type + "_")
        else:
            names.add(object_type)

    # Re-fetch after rename so subsequent passes use current (possibly suffixed) names.
    current_activities = list(data.get_activities())
    current_object_types = list(data.get_object_types())

    # Collect attribute names from events
    for activity in current_activities:
        table = data.get_events(activity)
        attribute_names |= set(table.data.columns) - {table.id, table.time}

    # Collect attribute names from objects
    for object_type in current_object_types:
        obj = data.get_object(object_type)
        attribute_names |= set(obj.data.columns) - {obj.id}

    # Rename attribute columns that clash with OWL property/class names
    names_casefolded = {n.casefold() for n in names}
    attributes_names_overlap = [
        a for a in attribute_names if a.casefold() in names_casefolded
    ]
    for attribute in attributes_names_overlap:
        renaming = {attribute: attribute + "_attr"}
        for activity in current_activities:
            df = data.get_events(activity).data
            if attribute in df.columns:
                data.set_event_data(activity, df.rename(columns=renaming))
        for object_type in current_object_types:
            df = data.get_object(object_type).data
            if attribute in df.columns:
                data.set_object_data(object_type, df.rename(columns=renaming))

    # Rename attribute values that clash with any known OWL name.
    # frozenset, not a list: this is membership-tested once per DataFrame cell.
    all_names = frozenset(names | attribute_names)

    def rename_values(df: pd.DataFrame) -> pd.DataFrame:
        return df.map(lambda v: _rename_attr_val(v, all_names))

    for activity in current_activities:
        data.set_event_data(activity, rename_values(data.get_events(activity).data))
    for object_type in current_object_types:
        data.set_object_data(
            object_type, rename_values(data.get_object(object_type).data)
        )

    return data


def clean_oced_data(data: OCEDStore, casefold: bool = False) -> OCEDStore:
    """Sanitise all names and values in an OCEDStore so they are safe as OWL IRI fragments.

    Applies :func:`clean_string` (``[A-Za-z0-9_]`` only) to all string keys and
    values throughout the store — events, objects, relations, and object-change
    history.  Then calls :func:`fix_duplicate_names` to resolve any collisions
    with reserved OWL property names.  Finally, builds a backmap from
    post-clean IRI fragments back to their original names.

    Must be called before :meth:`~semanticlog.onto.instantiate.OCEDOntoBuilder.build_ontology`;
    skipping it produces malformed IRIs.

    Args:
        data: OCEDStore to clean in-place (also returned for chaining).
        casefold: If True, also lowercase all cleaned strings.

    Returns:
        The same OCEDStore with all names sanitised and ``_backmap`` populated.
    """
    logging.info("Cleaning data..")
    _check_schema_name_collisions(data, casefold)
    pre_ids = _collect_entity_ids(data)
    pre_names = _collect_schema_names(data)
    value_backmap = _collect_attr_value_backmap(data, casefold=casefold)
    data.replace_tables(
        events=_clean_events(data, casefold),
        objects=_clean_objects(data, casefold),
        relations=_clean_relations(data, casefold),
    )
    if data.has_ocel_mode():
        data.replace_object_changes(
            {
                clean_string(obj_type, casefold=casefold): clean_object_changes_df(
                    data.get_object_changes(obj_type), casefold=casefold
                )
                for obj_type in data.get_object_change_types()
            }
        )
    data = fix_duplicate_names(data)
    data.update_used_ids()
    post = _collect_entity_ids(data) | _collect_schema_names(data)
    backmap = _build_backmap(pre_ids | pre_names, post, casefold=casefold)
    backmap.update(value_backmap)
    data.backmap = backmap
    return data


def _clean_events(data: OCEDStore, casefold: bool) -> dict[str, EventTable]:
    return {
        clean_string(activity, casefold=casefold): EventTable(
            id=clean_string(table.id, casefold=casefold),
            time=clean_string(table.time, casefold=casefold),
            data=clean_dataframe(table.data, casefold=casefold),
        )
        for activity, table in ((a, data.get_events(a)) for a in data.get_activities())
    }


def _clean_objects(data: OCEDStore, casefold: bool) -> dict[str, ObjectTable]:
    return {
        clean_string(obj_type, casefold=casefold): ObjectTable(
            id=clean_string(obj.id, casefold=casefold),
            data=clean_dataframe(obj.data, casefold=casefold),
        )
        for obj_type, obj in ((o, data.get_object(o)) for o in data.get_object_types())
    }


def _clean_relations(data: OCEDStore, casefold: bool) -> Relations:
    def clean_types(types: list[str] | None) -> list[str] | None:
        if types is None:
            return None
        return [clean_string(str(t), casefold=casefold) for t in types]

    def clean_family(kind: RelationKind) -> dict[str, RelationTable]:
        family = cast(dict[str, RelationTable], data.get_relations(kind))
        return {
            clean_string(name, casefold=casefold): RelationTable(
                from_id=clean_string(rel.from_id, casefold=casefold),
                from_type=clean_types(rel.from_type),
                to_id=clean_string(rel.to_id, casefold=casefold),
                to_type=cast(list, clean_types(rel.to_type)),
                data=clean_dataframe(rel.data, casefold=casefold),
            )
            for name, rel in family.items()
        }

    return Relations(
        event_to_object=clean_family(RelationKind.E2O),
        object_to_object=clean_family(RelationKind.O2O),
    )
