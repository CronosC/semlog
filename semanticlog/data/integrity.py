"""Referential checks on a store, run before a build.

A relation row that names an entity the store does not hold would still become
an assertion in the ontology, about an individual that appears nowhere else.
An entity with no id has no name to be asserted under at all. These checks
find both while the failure can still name the table and the ids.
"""

from __future__ import annotations

from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind, RelationKind

_SAMPLE = 5


def _sample(ids: set) -> str:
    """Render a few ids for an error message, counting the rest."""
    shown = sorted(str(i) for i in ids)[:_SAMPLE]
    more = len(ids) - len(shown)
    return ", ".join(shown) + (f" and {more} more" if more else "")


def _missing_ids(data: OCEDStore) -> list[str]:
    """Find events and objects whose id is empty.

    An id becomes the local name of an individual, so a row without one has
    nothing to be emitted under, and every such row across the log would
    otherwise arrive at the sink as the same name.
    """
    violations: list[str] = []
    for view in data.tables(changes=False):
        if view.id_column is None:
            continue
        missing = int(view.frame[view.id_column].isna().sum())
        if missing:
            noun = "event" if view.kind is EntityKind.EVENT else "object"
            violations.append(
                f"{view.origin}: {missing} {noun}(s) have no {view.id_column!r}"
            )
    return violations


def check_integrity(data: OCEDStore) -> list[str]:
    """Find rows a build could not emit an individual for.

    Every event and object has to carry an id. Both endpoints of every
    relation have to name an entity the store holds, and an object endpoint
    has to be in one of the types the relation declares, not merely somewhere
    in the store.

    Args:
        data: Store to check.

    Returns:
        One message per violation, empty when the store is sound.
    """
    violations: list[str] = _missing_ids(data)

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
            "The store holds rows a build could not emit an individual for:\n"
            + "\n".join(f"  {v}" for v in violations)
            + "\n\nEvery such row would become an assertion about an individual "
            "that appears nowhere else in the ontology, or that has no name at "
            "all. Fix the source data, or drop the rows before building."
        )
