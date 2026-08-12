"""Referential checks on a store, run before a build.

A relation row that names an entity the store does not hold would still become
an assertion in the ontology, about an individual that appears nowhere else.
These checks find those rows while the failure can still name the relation and
the ids.
"""

from __future__ import annotations

from semanticlog.data.oced import OCEDStore, RelationKind

_SAMPLE = 5


def _sample(ids: set) -> str:
    """Render a few ids for an error message, counting the rest."""
    shown = sorted(str(i) for i in ids)[:_SAMPLE]
    more = len(ids) - len(shown)
    return ", ".join(shown) + (f" and {more} more" if more else "")


def check_integrity(data: OCEDStore) -> list[str]:
    """Find relation rows pointing at entities the store does not hold.

    Both endpoints of every relation are checked, and an object endpoint has
    to be in one of the types the relation declares, not merely somewhere in
    the store.

    Args:
        data: Store to check.

    Returns:
        One message per violation, empty when the store is sound.
    """
    violations: list[str] = []

    event_ids: set = set()
    for activity in data.get_event_types():
        table = data.get_event_table(activity)
        event_ids.update(table.data[table.id].unique())

    object_ids: dict[str, set] = {
        obj_type: set(table.data[table.id].unique())
        for obj_type, table in data.get_object_tables().items()
    }

    def known(types) -> set:
        """Collect the ids held by the given object types."""
        found: set = set()
        for obj_type in types or ():
            found |= object_ids.get(obj_type, set())
        return found

    for name in data.get_relation_types(RelationKind.E2O):
        relation = data.get_relation(name)
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
            missing = set(relation.data[column].dropna().unique()) - known(types)
            if missing:
                violations.append(
                    f"o2o {name!r} {side} -> {list(types or [])}: {len(missing)} "
                    f"id(s) are in none of those types ({_sample(missing)})"
                )

    return violations


def require_integrity(data: OCEDStore) -> None:
    """Raise unless the store passes `check_integrity`.

    Raises:
        ValueError: Listing every violation found.
    """
    violations = check_integrity(data)
    if violations:
        raise ValueError(
            "The store has relations pointing at entities it does not hold:\n"
            + "\n".join(f"  {v}" for v in violations)
            + "\n\nEvery such row would become an assertion about an individual "
            "that appears nowhere else in the ontology. Fix the source data, or "
            "drop the rows before building."
        )
