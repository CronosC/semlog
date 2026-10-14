"""Renaming the ids and values whose individual name another claimant holds.

`resolve_individual_names` composes every individual name a build emits, from
ids, attribute values, timestamps and change records, and renames one
claimant of each name that is taken.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import pandas as pd

from semanticlog.core import vocabulary as vocab
from semanticlog.core.errors import InputError
from semanticlog.core.names import NameRegistry, candidates
from semanticlog.data.cleaning.allocate import KIND_PREFIX, VALUE_PREFIX
from semanticlog.data.cleaning.names import free_name
from semanticlog.data.cleaning.renames import RenameKind, RenameLog
from semanticlog.data.cleaning.rewrite import (
    apply_id_renames,
    apply_value_renames,
    attribute_names,
    entity_id_columns,
    entity_ids,
)
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind

logger = logging.getLogger(__name__)

_MAX_REPORTED_COLLISIONS = 20

# A rename can put a change record onto another name. Composing runs again
# after each round of renames, this many times at most.
_ROUNDS = 3

_RANK = {"time": 0, "value": 1, "id": 2, "change": 3}

_COLUMNS = ["name", "kind", "attribute", "source", "text"]


def _claims(
    names: Iterable[str],
    kind: str,
    *,
    attribute: str = "",
    source: Iterable[Any] | None = None,
    text: bool | Iterable[bool] = True,
) -> pd.DataFrame:
    """Return one claim row per name."""
    listed = [str(name) for name in names]
    count = len(listed)
    return pd.DataFrame(
        {
            "name": pd.Series(listed, dtype="object"),
            "kind": [kind] * count,
            "attribute": [attribute] * count,
            "source": pd.Series(
                list(source) if source is not None else listed, dtype="object"
            ),
            "text": pd.Series(
                list(text) if not isinstance(text, bool) else [text] * count,
                dtype=bool,
            ),
        },
        columns=_COLUMNS,
    )


def _time_claims(data: OCEDStore) -> list[pd.DataFrame]:
    stamps = []
    for activity in data.types(EntityKind.EVENT):
        table = data.table(EntityKind.EVENT, activity)
        if table.time is not None:
            stamps.append(table.data[table.time])
    for changes in _change_tables(data):
        if "ocel:timestamp" in changes.columns:
            stamps.append(changes["ocel:timestamp"])
    claims = []
    for column in stamps:
        distinct = pd.Series(column.dropna().unique())
        if distinct.empty:
            continue
        claims.append(_claims(vocab.timestamp_ids(distinct), "time", text=False))
    if not claims:
        return []
    merged = pd.concat(claims, ignore_index=True).drop_duplicates("name")
    return [merged]


def _distinct_values(data: OCEDStore, attribute: str) -> list[Any]:
    """Return every distinct value one attribute takes, across every table."""
    found: dict[Any, None] = {}
    for view in data.tables():
        if attribute in view.attributes:
            found.update(dict.fromkeys(view.frame[attribute].dropna().unique()))
    return list(found)


def _value_claims(data: OCEDStore) -> list[pd.DataFrame]:
    claims = []
    for attribute in sorted(attribute_names(data)):
        values = _distinct_values(data, attribute)
        names, sources, text = [], [], []
        for value in values:
            # A value with no individual name, such as an infinity, is left
            # out. The emitter rejects it later.
            try:
                names.append(vocab.attribute_value(attribute, value))
            except InputError:
                continue
            sources.append(value)
            text.append(isinstance(value, str))
        if names:
            claims.append(
                _claims(names, "value", attribute=attribute, source=sources, text=text)
            )
    return claims


def _change_tables(data: OCEDStore) -> list[pd.DataFrame]:
    """Return the change tables of the store, none outside OCEL mode."""
    if not data.ocel_mode:
        return []
    return [data.get_object_changes(t) for t in data.get_object_change_types()]


def _change_claims(data: OCEDStore) -> list[pd.DataFrame]:
    claims = []
    for changes in _change_tables(data):
        if "ocel:field" not in changes.columns:
            continue
        for attribute in sorted(changes["ocel:field"].dropna().astype(str).unique()):
            if attribute not in changes.columns:
                continue
            rows = changes[
                (changes["ocel:field"] == attribute) & changes[attribute].notna()
            ]
            if rows.empty:
                continue
            stamps = vocab.timestamp_ids(rows["ocel:timestamp"])
            names = vocab.attribute_change_ids(rows["ocel:oid"], attribute, stamps)
            claims.append(
                _claims(
                    names,
                    "change",
                    attribute=attribute,
                    source=rows["ocel:oid"].astype(str),
                )
            )
    return claims


def compose_individual_names(data: OCEDStore) -> pd.DataFrame:
    """Return every individual name a build emits, with what claims it.

    Returns:
        One row per claim, with the name, its kind (`id`, `value`, `time` or
        `change`), the attribute for a value or a change, the id or value it
        comes from, and whether that is text.
    """
    parts = [
        _claims(entity_ids(data), "id"),
        *_time_claims(data),
        *_value_claims(data),
        *_change_claims(data),
    ]
    return pd.concat(parts, ignore_index=True)


def _describe(row: Any) -> str:
    """Return how a message names one claimant."""
    if row.kind == "id":
        return f"id {row.source!r}"
    if row.kind == "value":
        return f"a value of attribute {row.attribute!r}"
    if row.kind == "change":
        return f"a change of attribute {row.attribute!r} of object {row.source!r}"
    return "a timestamp"


def _contested(claims: pd.DataFrame, registry: NameRegistry) -> pd.DataFrame:
    """Return the claims whose name another claim or a schema name holds."""
    names = claims["name"]
    return claims[names.duplicated(keep=False) | names.isin(registry.names)]


def _resolve(
    contested: pd.DataFrame, registry: NameRegistry, taken: set[str]
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Decide which claimant of each contested name moves, and where to.

    Returns:
        The id renames, and the value renames keyed by attribute.

    Raises:
        InputError: If a claimant that cannot be renamed has to move.
    """
    ids: dict[str, str] = {}
    values: dict[str, dict[str, str]] = {}
    ordered = contested.assign(
        movable=contested["text"] & contested["kind"].ne("time"),
        rank=contested["kind"].map(_RANK),
        source_text=contested["source"].astype(str),
    ).sort_values(["name", "movable", "rank", "source_text", "attribute"])

    for name, group in ordered.groupby("name", sort=False):
        held = registry.owner(name)
        rows = list(group.itertuples(index=False))
        keeper = held if held is not None else _describe(rows[0])
        movers = rows if held is not None else rows[1:]
        for row in movers:
            if not row.movable:
                why = (
                    "The value is not text and cannot be renamed."
                    if row.kind == "value"
                    else "Neither can be renamed."
                )
                raise InputError(
                    f"{name!r} is the name of both {_describe(row)} and {keeper}. "
                    f"{why}\n\nRename {keeper} in the input data."
                )
            if row.kind == "value":
                renames = values.setdefault(row.attribute, {})
                if row.source in renames:
                    continue
                free = next(
                    c
                    for c in candidates(row.source, VALUE_PREFIX)
                    if vocab.attribute_value(row.attribute, c) not in taken
                )
                taken.add(vocab.attribute_value(row.attribute, free))
                renames[row.source] = free
            else:
                if row.source in ids:
                    continue
                free = free_name(row.source, taken, KIND_PREFIX[RenameKind.ID])
                taken.add(free)
                ids[row.source] = free
    return ids, values


def resolve_individual_names(
    data: OCEDStore, log: RenameLog, registry: NameRegistry
) -> None:
    """Rename the ids and values whose individual name another claimant holds.

    Only an id and a text value can be renamed. Among the claimants of one
    name that can move, a value keeps it before an id, and an id before a
    change record. Two of one kind are sorted by id or value, and the first
    keeps the name.

    Args:
        data: Store to fix, modified in place. Its schema names must be
            allocated.
        log: Rename log to append to.
        registry: The schema names the build will emit.

    Raises:
        InputError: If a claimant that cannot be renamed has to move, or if
            names are shared after the last round of renames.
    """
    for _ in range(_ROUNDS):
        claims = compose_individual_names(data)
        contested = _contested(claims, registry)
        if contested.empty:
            return
        taken = set(claims["name"]) | registry.names
        ids, values = _resolve(contested, registry, taken)
        if ids:
            logger.info(
                "  renaming %s id(s) whose individual name another claimant holds",
                len(ids),
            )
            apply_id_renames(data, ids)
            data.update_used_ids()
            for old, free in ids.items():
                log.record(RenameKind.ID, old, free)
        for attribute, renames in values.items():
            logger.info(
                "  renaming %s value(s) of attribute %s that another name claims",
                len(renames),
                attribute,
            )
            apply_value_renames(data, attribute, renames)
            for old, free in renames.items():
                log.rename_value(attribute, old, free)

    contested = _contested(compose_individual_names(data), registry)
    if contested.empty:
        return
    name = contested["name"].iloc[0]
    claimants = ", ".join(
        _describe(row) for row in contested[contested["name"] == name].itertuples()
    )
    raise InputError(
        f"{name!r} is the name of {claimants} after renaming.\n\n"
        "Rename one of them in the input data."
    )


def verify_unique_names(data: OCEDStore) -> None:
    """Raise if two names in the store are the same after every rename has run.

    Raises:
        InputError: Listing every name claimed more than once.
    """
    claims: dict[str, list[str]] = {}

    def claim(name: object, description: str) -> None:
        claims.setdefault(str(name), []).append(description)

    for activity in data.types(EntityKind.EVENT):
        claim(activity, f"activity {activity!r}")
    for obj_type in data.types(EntityKind.OBJECT):
        claim(obj_type, f"object type {obj_type!r}")
    for attribute in sorted(attribute_names(data)):
        claim(attribute, f"attribute {attribute!r}")
    for relation in data.get_relation_types():
        claim(relation, f"relation {relation!r}")

    # Ids are compared as Series. Only the clashing ones get a description.
    per_table = entity_id_columns(data)
    ids = entity_ids(data)
    clashing = ids[ids.duplicated(keep=False) | ids.isin(set(claims))].unique()
    for name in clashing:
        for origin, table_ids in per_table:
            if (table_ids == name).any():
                claim(name, f"id from {origin}")

    errors = [
        f"  {name!r} is claimed {len(owners)} times: {', '.join(sorted(owners))}"
        for name, owners in claims.items()
        if len(owners) > 1
    ]
    if not errors:
        return
    shown = errors[:_MAX_REPORTED_COLLISIONS]
    more = len(errors) - len(shown)
    raise InputError(
        "These names are claimed more than once:\n"
        + "\n".join(shown)
        + (f"\n  ... and {more} more" if more else "")
        + "\n\nRename the duplicate in the input data. The same id under two "
        "object types is the usual cause."
    )
