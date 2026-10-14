"""Referential checks on a store, run before a build.

Every event and object needs an id, and every relation endpoint needs to name
an entity the store holds. Each violation is reported with its table and ids.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from semanticlog.core.checks import CheckReport, Fixed, raise_reports
from semanticlog.data.readable import ReadableStore
from semanticlog.data.tables import EntityKind, RelationKind

_SAMPLE = 5


def _sample(ids: set[Any]) -> str:
    """Render a few ids for an error message, counting the rest."""
    shown = sorted(str(i) for i in ids)[:_SAMPLE]
    more = len(ids) - len(shown)
    return ", ".join(shown) + (f" and {more} more" if more else "")


def _missing_ids(data: ReadableStore) -> list[str]:
    """Return one line per table with ids that are missing, empty or blank."""
    violations: list[str] = []
    for view in data.tables(changes=False):
        if view.id_column is None:
            continue
        ids = view.frame[view.id_column]
        missing = int((ids.isna() | ids.astype(str).str.strip().eq("")).sum())
        if missing:
            noun = "event" if view.kind is EntityKind.EVENT else "object"
            violations.append(
                f"{view.origin}: {missing} {noun}(s) have no {view.id_column!r}"
            )
    return violations


def check_missing_ids(data: ReadableStore) -> CheckReport | None:
    """Report the events and objects with no id, or None when every row has one.

    Run it before cleaning, which rewrites blank text into an underscore.
    """
    violations = _missing_ids(data)
    if not violations:
        return None
    return CheckReport(
        header="The store holds rows with no id:",
        problems=tuple(violations),
        advice="Give every event and object an id, or drop the rows before building.",
        about=Fixed.INPUT,
    )


def check_integrity(data: ReadableStore) -> CheckReport | None:
    """Report the rows a build could not emit an individual for, or None.

    A row is reported when it has no id, or when a relation endpoint names no
    entity of the types the relation declares.
    """
    violations: list[str] = _missing_ids(data)

    event_ids: set[Any] = set()
    for activity in data.types(EntityKind.EVENT):
        table = data.table(EntityKind.EVENT, activity)
        event_ids.update(table.data[table.id].unique())

    object_ids: dict[str, set[Any]] = {
        obj_type: set(table.data[table.id].unique())
        for obj_type, table in data.tables_of(EntityKind.OBJECT).items()
    }

    def known(types: Iterable[str] | None) -> set[Any]:
        """Return the ids held by the given object types."""
        found: set[Any] = set()
        for obj_type in types or ():
            found |= object_ids.get(obj_type, set())
        return found

    def empty_endpoints(family: str, name: str, column: str) -> None:
        relation = data.get_relation(name)
        blank = int(relation.data[column].isna().sum())
        if blank:
            violations.append(f"{family} {name!r}: {blank} row(s) have no {column!r}")

    for name in data.get_relation_types(RelationKind.E2O):
        relation = data.get_relation(name)
        empty_endpoints("e2o", name, relation.from_id)
        empty_endpoints("e2o", name, relation.to_id)
        missing = set(relation.data[relation.from_id].dropna().unique()) - event_ids
        if missing:
            violations.append(
                f"e2o {name!r}: {len(missing)} event id(s) are in no event table "
                f"({_sample(missing)})"
            )
        missing = set(relation.data[relation.to_id].dropna().unique()) - known(
            relation.to_type
        )
        if missing:
            violations.append(
                f"e2o {name!r} -> {list(relation.to_type)}: {len(missing)} object "
                f"id(s) are in none of those types ({_sample(missing)})"
            )

    for name in data.get_relation_types(RelationKind.O2O):
        relation = data.get_relation(name)
        for column, types, side in (
            (relation.from_id, relation.from_type, "from"),
            (relation.to_id, relation.to_type, "to"),
        ):
            empty_endpoints("o2o", name, column)
            missing = set(relation.data[column].dropna().unique()) - known(types)
            if missing:
                violations.append(
                    f"o2o {name!r} {side} -> {list(types or [])}: {len(missing)} "
                    f"id(s) are in none of those types ({_sample(missing)})"
                )

    if not violations:
        return None
    return CheckReport(
        header="The store holds rows a build could not emit an individual for:",
        problems=tuple(violations),
        advice="Fix the source data, or drop the rows before building.",
        about=Fixed.INPUT,
    )


def require_integrity(data: ReadableStore) -> None:
    """Raise unless the store passes `check_integrity`.

    Raises:
        InputError: Listing every violation found.
    """
    raise_reports([check_integrity(data)])
