"""Renaming the source names that cleaning would merge into one.

`resolve_merged_names` runs on the source names before cleaning. Of two names
that clean to one string, one is given a clean name of its own.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterable

import pandas as pd

from semanticlog.data.cleaning.allocate import KIND_PREFIX, VALUE_PREFIX
from semanticlog.data.cleaning.names import clean_series, cleaned, free_name
from semanticlog.data.cleaning.qualify import qualified_attribute
from semanticlog.data.cleaning.renames import RenameKind, RenameLog
from semanticlog.data.cleaning.rewrite import (
    apply_attribute_renames,
    apply_id_renames,
    apply_value_renames,
    entity_ids,
    value_groups,
)
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind

logger = logging.getLogger(__name__)


def merged_names(
    names: Iterable[str] | pd.Series, prefix: str, fixed: Collection[str] = ()
) -> dict[str, str]:
    """Return a replacement source name for each name that cleaning would merge into another.

    Names are taken in sorted order, and the first of a colliding set keeps
    its name.

    Args:
        names: Source names that share one namespace in the ontology.
        fixed: Names that keep their name in every collision.

    Returns:
        Source name to replacement, for the names that need one.
    """
    series = names if isinstance(names, pd.Series) else pd.Series(list(names))
    if series.empty:
        return {}

    distinct = series.astype(str).drop_duplicates().sort_values()
    if fixed:
        held = distinct.isin(set(fixed))
        distinct = pd.concat([distinct[held], distinct[~held]])
    colliding = clean_series(distinct).duplicated()
    if not colliding.any():
        return {}

    taken = set(clean_series(distinct))
    renames: dict[str, str] = {}
    for name in distinct[colliding]:
        free = free_name(cleaned(name), taken, prefix)
        taken.add(free)
        renames[name] = free
    return renames


def resolve_merged_names(data: OCEDStore, log: RenameLog) -> None:
    """Rename, in place, the source names that cleaning would merge into one.

    Both relation families form one namespace. Attributes are checked per
    table, and the id and time columns keep their names. Values are checked
    per type and attribute.
    """
    activities = merged_names(
        data.types(EntityKind.EVENT), KIND_PREFIX[RenameKind.ACTIVITY]
    )
    for activity, free in activities.items():
        data.rename(EntityKind.EVENT, activity, free)
        log.record(RenameKind.ACTIVITY, activity, free)

    obj_types = merged_names(
        data.types(EntityKind.OBJECT), KIND_PREFIX[RenameKind.OBJECT_TYPE]
    )
    for obj_type, free in obj_types.items():
        data.rename(EntityKind.OBJECT, obj_type, free)
        log.record(RenameKind.OBJECT_TYPE, obj_type, free)

    relations = merged_names(
        data.get_relation_types(), KIND_PREFIX[RenameKind.RELATION]
    )
    for relation, free in relations.items():
        data.rename_relation(relation, free)
        log.record(RenameKind.RELATION, relation, free)

    for view in data.tables():
        fixed = view.key_columns if view.kind is not None else ()
        renames = merged_names(
            [*fixed, *view.attributes], KIND_PREFIX[RenameKind.ATTRIBUTE], fixed
        )
        if renames:
            apply_attribute_renames(data, renames)
            for old, free in renames.items():
                log.record(RenameKind.ATTRIBUTE, old, free)

    ids = merged_names(entity_ids(data), KIND_PREFIX[RenameKind.ID])
    if ids:
        logger.info("  renaming %s id(s) that would clean to another", len(ids))
        apply_id_renames(data, ids)
        for old, free in ids.items():
            log.record(RenameKind.ID, old, free)

    for (type_name, attribute), values in value_groups(data).items():
        renames = merged_names(values, VALUE_PREFIX)
        if not renames:
            continue
        column = qualified_attribute(cleaned(type_name), attribute)
        logger.info(
            "  renaming %s value(s) of attribute %s that would clean to another",
            len(renames),
            column,
        )
        apply_value_renames(data, attribute, renames, type_name=type_name)
        for old, free in renames.items():
            log.record_value(column, cleaned(free), old)
