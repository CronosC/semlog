"""Resolving the collisions that cleaning a name can cause.

Three kinds of collision are possible. Two source names can clean to one
string, one name can be claimed by two different kinds of thing, and a name can
clash with one the build reserves for its own schema. Each is resolved by
renaming the source before the ordinary cleaning pass runs, so the rows behind
two names never become indistinguishable.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Collection, Iterable, Mapping
from typing import Any

import pandas as pd

from semanticlog.core import vocabulary as vocab
from semanticlog.data.cleaning.names import (
    clean_series,
    clean_string,
    cleaned,
    is_cleaned_column,
    is_timestamp,
)
from semanticlog.data.cleaning.renames import RenameKind, RenameLog
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import TableView

logger = logging.getLogger(__name__)

_RESERVED_OWL_NAMES: frozenset[str] = vocab.RESERVED_NAMES | {"is_a"}

_MAX_REPORTED_COLLISIONS = 20

# A property name is written as an RDF/XML element, so it cannot open with a
# digit, a dot or a hyphen. Only relation types reach a property name unchanged.
_UNSAFE_PROPERTY_START = re.compile(r"^[^A-Za-z_]")

_ATTR_VALUE_SUFFIX = "_attr_val"

_CLASS_KIND_PREFIX = {
    RenameKind.ACTIVITY: "event_",
    RenameKind.OBJECT_TYPE: "object_",
    RenameKind.ATTRIBUTE: "attribute_",
}


def _rename_attr_val(value: Any, reserved: Collection[str]) -> Any:
    """Suffix a value that would otherwise collide with a schema name."""
    return value + _ATTR_VALUE_SUFFIX if value in reserved else value


def collect_attribute_names(data: OCEDStore) -> set[str]:
    """Collect every attribute column of events, objects and their history.

    A change table can name an attribute the object table no longer carries,
    since a column that is empty for every object is dropped on load. Its
    class and its `change_value_` property are still emitted, so the name
    belongs in the reserved set like any other.
    """
    return {name for view in data.tables() for name in view.attributes}


def _entity_id_series(data: OCEDStore) -> list[tuple[str, pd.Series]]:
    """Collect every event and object id as text, one column per table.

    Ids are compared across the whole store rather than per table, because the
    store holds them in a single namespace and each one becomes an individual.
    They stay in columns so the passes that compare them do it without a Python
    object per id, which on a real log is most of what cleaning would allocate.
    """
    columns = []
    for view in data.tables(changes=False):
        if view.id_column is None:
            continue
        ids = pd.Series(view.frame[view.id_column].dropna().unique())
        columns.append((view.origin, ids.astype(str)))
    return columns


def _all_entity_ids(data: OCEDStore) -> pd.Series:
    """Return every event and object id of the store as one column."""
    columns = [ids for _, ids in _entity_id_series(data)]
    if not columns:
        return pd.Series(dtype="object")
    return pd.concat(columns, ignore_index=True)


def _text_values(view: TableView, column: str) -> list[str]:
    """Return the distinct text values of one column that cleaning may rewrite."""
    if not is_cleaned_column(view.frame[column]):
        return []
    return [
        value
        for value in view.frame[column].dropna().unique()
        if isinstance(value, str) and not is_timestamp(value)
    ]


def _value_groups(data: OCEDStore) -> dict[str, list[tuple[str, str]]]:
    """Collect the text values of each attribute, across every table carrying it.

    An attribute value becomes an individual named after its attribute, and
    two tables with one attribute name share that individual, so the values
    are grouped by the cleaned attribute name rather than by table.
    """
    groups: dict[str, list[tuple[str, str]]] = {}
    for view in data.tables():
        for column in view.attributes:
            group = groups.setdefault(cleaned(column), [])
            group += [(value, view.origin) for value in _text_values(view, column)]
    return groups


def verify_unique_names(data: OCEDStore) -> None:
    """Raise if two things would still be emitted under one name.

    Every rename has run by the time this does, so what it reports is a
    duplicate the input carries itself, such as one id naming both an event and
    an object, which no rename can resolve without choosing one of them.
    """
    claims: dict[str, list[str]] = {}

    def claim(name: object, description: str) -> None:
        claims.setdefault(str(name), []).append(description)

    for activity in data.get_event_types():
        claim(activity, f"activity {activity!r}")
    for obj_type in data.get_object_types():
        claim(obj_type, f"object type {obj_type!r}")
    for attribute in sorted(collect_attribute_names(data)):
        claim(attribute, f"attribute {attribute!r}")

    # Ids are the bulk of a log. They are compared as columns, and only the
    # ones that actually clash are described, so a sound log allocates nothing
    # per id here.
    per_table = _entity_id_series(data)
    ids = _all_entity_ids(data)
    clashing = ids[ids.duplicated(keep=False) | ids.isin(set(claims))].unique()
    for name in clashing:
        for origin, table_ids in per_table:
            if (table_ids == name).any():
                claim(name, f"id from {origin}")

    errors = [
        f"  {name!r} names {len(owners)} things: {', '.join(sorted(owners))}"
        for name, owners in claims.items()
        if len(owners) > 1
    ]
    if not errors:
        return
    shown = errors[:_MAX_REPORTED_COLLISIONS]
    more = len(errors) - len(shown)
    raise ValueError(
        "clean_oced_data: the following names are claimed more than once:\n"
        + "\n".join(shown)
        + (f"\n  ... and {more} more" if more else "")
        + "\n\nRename the duplicate in the input data. The same id under two "
        "object types is the usual cause."
    )


def record_value_renames(data: OCEDStore, log: RenameLog) -> None:
    """Record the attribute values that cleaning is about to rewrite.

    Values are recorded against their column, so that a later rename of the
    column carries them with it.
    """

    for view in data.tables():
        for column in view.attributes:
            cleaned_column = cleaned(column)
            for value in _text_values(view, column):
                cleaned_value = clean_string(value)
                if cleaned_value != value:
                    log.record_value(cleaned_column, cleaned_value, value)


def record_id_renames(data: OCEDStore, log: RenameLog) -> None:
    """Record the event and object ids that cleaning is about to rewrite."""
    ids = _all_entity_ids(data)
    if ids.empty:
        return
    clean = clean_series(ids)
    for value, cleaned_value in zip(
        ids[clean != ids], clean[clean != ids], strict=True
    ):
        log.record(RenameKind.ID, value, cleaned_value)


def _free_name(name: str, taken: Collection[str]) -> str:
    """Append underscores until the name is unused."""
    candidate = name + "_"
    while candidate in taken:
        candidate += "_"
    return candidate


def _free_source_name(name: str, taken: Collection[str]) -> str:
    """Count up a suffix on a source name until it cleans to something unused."""
    counter = 2
    while cleaned(f"{name}_{counter}") in taken:
        counter += 1
    return f"{name}_{counter}"


def _merged_names(names: Iterable[str] | pd.Series) -> dict[str, str]:
    """Return a new source name for each name cleaning would merge into another.

    Names are walked in sorted order, so the first of a colliding set keeps the
    name it would have had and the result does not depend on the order the
    store happens to hold them in.

    Nothing is allocated per name unless something actually collides, since
    this runs over every id in the log and most logs have no collision at all.

    Args:
        names: Source names that share one namespace in the ontology.

    Returns:
        Source name to replacement, holding only the names that need one.
    """
    series = names if isinstance(names, pd.Series) else pd.Series(list(names))
    if series.empty:
        return {}

    distinct = series.astype(str).drop_duplicates().sort_values()
    colliding = clean_series(distinct).duplicated()
    if not colliding.any():
        return {}

    taken = set(clean_series(distinct))
    renames: dict[str, str] = {}
    for name in distinct[colliding]:
        free = _free_source_name(name, taken)
        taken.add(cleaned(free))
        renames[name] = free
    return renames


def _remap(
    df: pd.DataFrame, columns: Iterable[str], renames: Mapping[str, str]
) -> pd.DataFrame:
    """Return the table with the named columns rewritten through a rename map."""
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].map(
                lambda v: renames.get(v, v) if isinstance(v, str) else v
            )
    return out


def _apply_id_renames(data: OCEDStore, renames: Mapping[str, str]) -> None:
    """Rewrite an id in every column the store names an entity by.

    A change table names the object it belongs to, so its `ocel:oid` follows
    the rename along with the id columns and the relation endpoints.
    """
    for view in data.tables():
        columns = [view.id_column] if view.id_column is not None else ["ocel:oid"]
        view.replace(_remap(view.frame, columns, renames))
    for name, relation in data.relations():
        endpoints = [c for c in (relation.from_id, relation.to_id) if c is not None]
        data.replace_relation_data(name, _remap(relation.data, endpoints, renames))


def _apply_value_renames(
    data: OCEDStore, attribute: str, renames: Mapping[str, str]
) -> None:
    """Rewrite a value in every column that carries one attribute.

    The change history carries the same values as the object table it belongs
    to, and they become the same individuals, so it is rewritten alongside.
    """
    for view in data.tables():
        columns = [c for c in view.attributes if cleaned(c) == attribute]
        if columns:
            view.replace(_remap(view.frame, columns, renames))


def apply_attribute_renames(data: OCEDStore, renames: Mapping[str, str]) -> None:
    """Rename an attribute column in every table that carries it.

    A change table names its attribute twice, as a column and as the
    `ocel:field` of each row, and both follow the rename. Without that the
    history emitter would go on using the name the rename moved away from.
    """
    for view in data.tables():
        frame = view.frame.rename(columns=renames)
        if "ocel:field" in frame.columns:
            frame = _remap(frame, ["ocel:field"], renames)
        view.replace(frame)


def resolve_merged_names(data: OCEDStore, log: RenameLog) -> None:
    """Rename the source names that cleaning would otherwise merge into one.

    This runs before anything is cleaned. Once two names have become one string
    the rows behind them can no longer be told apart, so the rename is applied
    to the source name and the ordinary cleaning pass carries it through.

    Both relation families become properties of one ontology, so they are
    checked as one namespace.

    Args:
        data: Store to fix, modified in place.
        log: Rename log to append to.
    """
    for activity, free in _merged_names(data.get_event_types()).items():
        data.rename_activity(activity, free)
        log.record(RenameKind.ACTIVITY, activity, free)

    for obj_type, free in _merged_names(data.get_object_types()).items():
        data.rename_object(obj_type, free)
        log.record(RenameKind.OBJECT_TYPE, obj_type, free)

    for relation, free in _merged_names(data.get_relation_types()).items():
        data.rename_relation(relation, free)
        log.record(RenameKind.RELATION, relation, free)

    # Per table, since two tables may each carry a colliding pair of their own.
    for view in data.tables():
        renames = _merged_names(view.attributes)
        if renames:
            apply_attribute_renames(data, renames)
            _record_attribute_renames(renames, log)

    ids = _merged_names(_all_entity_ids(data))
    if ids:
        logger.info("  renaming %d id(s) that would clean to another..", len(ids))
        _apply_id_renames(data, ids)
        for old, free in ids.items():
            log.record(RenameKind.ID, old, free)

    for attribute, values in _value_groups(data).items():
        renames = _merged_names(value for value, _ in values)
        if not renames:
            continue
        logger.info(
            "  renaming %d value(s) of attribute %s that would clean to another..",
            len(renames),
            attribute,
        )
        _apply_value_renames(data, attribute, renames)
        for old, free in renames.items():
            log.record_value(attribute, cleaned(free), old)


def _record_attribute_renames(renames: Mapping[str, str], log: RenameLog) -> None:
    """Record renamed attribute columns, and carry their values across.

    The log holds a value under the cleaned name of its column, so that is the
    name the values follow, while the rename itself is recorded as the source
    wrote it.
    """
    for old, free in renames.items():
        log.record(RenameKind.ATTRIBUTE, old, free)
        log.rename_value_column(cleaned(old), cleaned(free))


def resolve_kind_collisions(data: OCEDStore, log: RenameLog) -> None:
    """Prefix a class name that two different kinds of thing both claim.

    An activity, an object type and an attribute all become a class named after
    themselves, so one name reaching the sink twice would merge them. The kind
    it came from goes in front of each claimant.

    Args:
        data: Store to fix, modified in place.
        log: Rename log to append to.
    """
    claims: dict[str, list[tuple[RenameKind, str]]] = {}
    for kind, names in (
        (RenameKind.ACTIVITY, data.get_event_types()),
        (RenameKind.OBJECT_TYPE, data.get_object_types()),
        (RenameKind.ATTRIBUTE, sorted(collect_attribute_names(data))),
    ):
        for name in names:
            claims.setdefault(cleaned(name), []).append((kind, str(name)))

    taken = set(claims)
    for claimants in claims.values():
        if len({kind for kind, _ in claimants}) < 2:
            continue
        for kind, name in claimants:
            free = _CLASS_KIND_PREFIX[kind] + name
            if cleaned(free) in taken:
                free = _free_source_name(free, taken)
            taken.add(cleaned(free))
            if kind is RenameKind.ACTIVITY:
                data.rename_activity(name, free)
            elif kind is RenameKind.OBJECT_TYPE:
                data.rename_object(name, free)
            else:
                apply_attribute_renames(data, {name: free})
                log.rename_value_column(cleaned(name), cleaned(free))
            log.record(kind, name, free)


def resolve_property_names(data: OCEDStore, log: RenameLog) -> None:
    """Prefix a relation whose name cannot open an RDF/XML element.

    A relation type becomes a property name unchanged, and RDF/XML writes a
    property as an element, so a name opening with a digit, a dot or a hyphen
    would produce a file no reasoner can read.

    Args:
        data: Store to fix, modified in place.
        log: Rename log to append to.
    """
    taken = {cleaned(name) for name in data.get_relation_types()}
    for relation in data.get_relation_types():
        if not _UNSAFE_PROPERTY_START.match(cleaned(relation)):
            continue
        free = "_" + relation
        if cleaned(free) in taken:
            free = _free_source_name(free, taken)
        taken.add(cleaned(free))
        data.rename_relation(relation, free)
        log.record(RenameKind.RELATION, relation, free)


def fix_duplicate_names(
    data: OCEDStore,
    log: RenameLog | None = None,
    bin_labels: Collection[str] = (),
) -> OCEDStore:
    """Rename anything that would collide with a name the build reserves.

    The reserved set covers the upper schema and every name the build will
    derive from this log, such as `has_price` for an attribute or `df_order`
    for a case concept. Relations, activities and object types are renamed
    with a trailing underscore, attributes with an `_attr` suffix, and
    colliding text values with an `_attr_val` suffix.

    Args:
        data: Store to fix, modified in place.
        log: Rename log to append to. A new one is used when none is given.
        bin_labels: Bin labels the build will use, so that a bin class name
            is reserved too. The labels come from the settings, which cleaning
            does not otherwise see.

    Returns:
        The same store.
    """
    log = log if log is not None else RenameLog()
    attribute_names = collect_attribute_names(data)
    names: set[str] = set(_RESERVED_OWL_NAMES) | vocab.derived_schema_names(
        data.get_object_types(), attribute_names, bin_labels
    )

    for kind, current, rename in (
        (RenameKind.RELATION, data.get_relation_types, data.rename_relation),
        (RenameKind.ACTIVITY, data.get_event_types, data.rename_activity),
        (RenameKind.OBJECT_TYPE, data.get_object_types, data.rename_object),
    ):
        for name in current():
            if name not in names:
                names.add(name)
                continue
            free = _free_name(name, names | set(current()))
            rename(name, free)
            log.record(kind, name, free)
            names.add(free)

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
        log.rename_value_column(attribute, candidate)
        names.add(candidate)
        attribute_names.add(candidate)
        apply_attribute_renames(data, renaming)

    reserved = frozenset(names | attribute_names)
    for view in data.tables():
        ids = {view.id_column} if view.id_column is not None else {"ocel:oid"}
        view.replace(
            _suffix_colliding_values(
                view.frame, ids, view.key_columns - ids, reserved, log
            )
        )
    for name, relation in data.relations():
        endpoints = {c for c in (relation.from_id, relation.to_id) if c is not None}
        data.replace_relation_data(
            name, _suffix_colliding_values(relation.data, endpoints, (), reserved, log)
        )

    return data


def _suffix_colliding_values(
    frame: pd.DataFrame,
    id_columns: Collection[str],
    skip: Collection[str],
    reserved: Collection[str],
    log: RenameLog,
) -> pd.DataFrame:
    """Suffix the cell values that collide with a name the build reserves.

    Args:
        frame: Table to rewrite.
        id_columns: Columns whose values name an entity, so that a rename in
            one is recorded as an id rather than as an attribute value.
        skip: Columns to leave alone, such as the `ocel:` columns that name a
            type or a field instead of holding a value of one.
        reserved: Names the build has claimed for itself.
        log: Rename log to append to.

    Returns:
        A new table. The input is not modified.
    """
    skip, ids = set(skip), set(id_columns)
    renamed = frame.copy()
    for column in frame.columns:
        if column in skip:
            continue
        for value in frame[column].dropna().unique():
            if isinstance(value, str) and value in reserved:
                if column in ids:
                    log.record(RenameKind.ID, value, value + _ATTR_VALUE_SUFFIX)
                else:
                    log.rename_value(column, value, value + _ATTR_VALUE_SUFFIX)
        if is_cleaned_column(frame[column]):
            renamed[column] = frame[column].map(lambda v: _rename_attr_val(v, reserved))
    return renamed
