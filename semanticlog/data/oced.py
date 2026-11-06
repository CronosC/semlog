"""The in-memory model every stage of a build reads and writes.

`OCEDStore` holds events grouped by activity, objects grouped by type, the
relations between them, and optionally the OCEL 2.0 attribute change history.
It holds the frames it is given without copying them.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from functools import partial
from types import MappingProxyType

import pandas as pd

from semanticlog.core.enums import IndividualKind
from semanticlog.core.errors import InputError
from semanticlog.core.vocabulary import DEFAULT_IRI as _DEFAULT_IRI
from semanticlog.data.readable import ReadableStore
from semanticlog.data.tables import (
    OCEL_PREFIX,
    EntityKind,
    EntityTable,
    RelationKind,
    Relations,
    RelationTable,
    TableView,
)


class OCEDStore:
    """Object-centric event data, held as typed tables.

    Ids are unique across the whole store. Ids are compared as Python values,
    so `1` and `"1"` are two ids here. `clean_oced_data` rejects such a pair.

    Args:
        ocel_mode: Whether the store may hold OCEL 2.0 attribute change
            history. The change methods raise without it.
    """

    def __init__(self, ocel_mode: bool = False) -> None:
        self._entities: dict[EntityKind, dict[str, EntityTable]] = {
            kind: {} for kind in EntityKind
        }
        self._relations = Relations()
        self._used_ids: set[object] = set()
        self._ocel_mode = ocel_mode
        self._backmap: dict[str, str] = {}
        self._object_changes: dict[str, pd.DataFrame] = {}

    @property
    def backmap(self) -> dict[str, str]:
        """Cleaned name to original name, for everything cleaning renamed."""
        return self._backmap

    @backmap.setter
    def backmap(self, mapping: dict[str, str]) -> None:
        self._backmap = mapping

    def get_original(
        self, iri_or_local_name: str, base_iri: str = _DEFAULT_IRI
    ) -> str | None:
        """Look up what a name in the store was called in the source log.

        Args:
            iri_or_local_name: A local name, or a full IRI to strip first. The
                path of an individual kind below the base is stripped too.

        Returns:
            The original name, or None if the name was never changed.
        """
        local_name = iri_or_local_name.removeprefix(base_iri)
        for kind in IndividualKind:
            if local_name.startswith(kind.path):
                local_name = local_name.removeprefix(kind.path)
                break
        return self._backmap.get(local_name)

    @property
    def ocel_mode(self) -> bool:
        """Whether this store may hold attribute change history."""
        return self._ocel_mode

    def add(
        self,
        kind: EntityKind,
        name: str,
        data: pd.DataFrame,
        *,
        id_col: str,
        time_col: str | None = None,
        position_col: str | None = None,
    ) -> None:
        """Add the entities of one activity or object type.

        Args:
            data: One row per entity. Every column other than `id_col`,
                `time_col` and `position_col` becomes an attribute. Dtypes are
                taken as given. The frame is held, not copied.
            time_col: Column holding the timestamp. Required for events, not
                allowed for objects.
            position_col: Column holding the place of each event in the
                source log, numbered across all activities. Events with the
                same timestamp are ordered by it. Not allowed for objects.

        Raises:
            InputError: If the timestamp column is missing or misplaced, the
                position column is misplaced, a named column is not in the
                data, the type is already in the store, or an id is repeated
                or held by another table. The store is left as it was.
        """
        if kind is EntityKind.EVENT and time_col is None:
            raise InputError(
                f"Activity {name!r} needs a time_col naming its timestamp column."
            )
        if kind is EntityKind.OBJECT and time_col is not None:
            raise InputError(
                f"Object type {name!r} takes no time_col, got {time_col!r}. "
                "Record attribute changes as OCEL attribute history."
            )
        if kind is EntityKind.OBJECT and position_col is not None:
            raise InputError(
                f"Object type {name!r} takes no position_col, got "
                f"{position_col!r}. Only events are ordered."
            )

        named = [c for c in (id_col, time_col, position_col) if c is not None]
        absent = [column for column in named if column not in data.columns]
        if absent:
            raise InputError(
                f"{kind.word.capitalize()} {name!r} names {absent}, which the "
                f"given data does not have. Its columns are "
                f"{sorted(map(str, data.columns))}."
            )

        tables = self._of_kind(kind)
        if name in tables:
            raise InputError(
                f"{kind.word.capitalize()} {name!r} is already in this store."
            )
        ids = data[id_col].dropna()
        counts = Counter(ids)
        taken = {i for i, n in counts.items() if n > 1 or i in self._used_ids}
        if taken:
            raise InputError(
                f"{kind.word.capitalize()} {name!r} reuses {len(taken)} id(s) "
                f"held by another table or repeated within its own, such as "
                f"{sorted(map(str, taken))[:5]}. Make ids unique across the "
                "store, for example by prefixing them with their type."
            )

        tables[name] = EntityTable(
            id=id_col, data=data, time=time_col, position=position_col
        )
        self._used_ids.update(counts)

    def types(self, kind: EntityKind) -> list[str]:
        """Return every activity or object type name in the store."""
        return list(self._of_kind(kind))

    def tables_of(self, kind: EntityKind) -> Mapping[str, EntityTable]:
        """Return the tables of one kind, keyed by type and read-only."""
        return MappingProxyType(self._of_kind(kind))

    def table(self, kind: EntityKind, name: str) -> EntityTable:
        """Return the table of one activity or object type.

        Raises:
            InputError: If the type is not in the store.
        """
        return self._require(kind, name, "return the table of")

    def ids(self, kind: EntityKind, name: str) -> Iterable[object]:
        """Return the distinct ids of one activity or object type.

        Returns:
            The ids, in order of first appearance.

        Raises:
            InputError: If the type is not in the store.
        """
        table = self._require(kind, name, "read the ids of")
        return table.data[table.id].unique()

    def types_holding(self, kind: EntityKind, ids: Iterable[object]) -> list[str]:
        """Return the sorted types of one kind whose tables hold any of the ids."""
        wanted = set(ids)
        return sorted(
            name
            for name, table in self.tables_of(kind).items()
            if table.data[table.id].isin(wanted).any()
        )

    def rename(self, kind: EntityKind, old_name: str, new_name: str) -> None:
        """Rename an activity or object type, in relation types and change tables too.

        Raises:
            InputError: If the old name is unknown, or the new name is taken.
        """
        tables = self._of_kind(kind)
        self._require(kind, old_name, "rename")
        if new_name in tables:
            raise InputError(
                f"Cannot rename {kind.word} {old_name!r}: {new_name!r} is "
                f"already an {kind.word}."
            )

        tables[new_name] = tables.pop(old_name)
        if kind is EntityKind.EVENT:
            return

        def renamed(types: list[str]) -> list[str]:
            return [new_name if t == old_name else t for t in types]

        for _, relation in self._relations.items():
            if relation.from_type is not None:
                relation.from_type = renamed(relation.from_type)
            relation.to_type = renamed(relation.to_type)

        if self._ocel_mode and old_name in self._object_changes:
            self._object_changes[new_name] = self._object_changes.pop(old_name)

    def get_most_common_objects(self, n: int = 1) -> list[str]:
        """Return the n object types with the most instances, most first.

        Raises:
            ValueError: If n is negative.
        """
        if n < 0:
            raise ValueError(f"get_most_common_objects needs n >= 0, got {n}.")
        objects = self._entities[EntityKind.OBJECT]
        return sorted(objects, key=lambda k: len(objects[k].data), reverse=True)[:n]

    def _of_kind(self, kind: EntityKind) -> dict[str, EntityTable]:
        """Return the tables of one kind.

        Raises:
            ValueError: If `kind` is not an `EntityKind`.
        """
        try:
            return self._entities[kind]
        except KeyError:
            raise ValueError(
                f"{kind!r} is not an EntityKind. Pass "
                f"{' or '.join(f'EntityKind.{k.name}' for k in EntityKind)}."
            ) from None

    def _require(self, kind: EntityKind, name: str, doing: str) -> EntityTable:
        """Return one table by kind and name.

        Args:
            doing: Verb phrase for the error message, such as `rename`.

        Raises:
            InputError: If the type is not in the store, listing what is.
        """
        tables = self._of_kind(kind)
        if name not in tables:
            raise InputError(
                f"Cannot {doing} {kind.word} {name!r}, which this store does "
                f"not hold. It holds {sorted(tables)}."
            )
        return tables[name]

    def _set_data(self, kind: EntityKind, name: str, data: pd.DataFrame) -> None:
        self._of_kind(kind)[name].data = data

    def add_relation(
        self,
        kind: RelationKind,
        name: str,
        data: pd.DataFrame,
        *,
        from_id: str,
        to_id: str,
        to_type: Sequence[str],
        from_type: Sequence[str] | None = None,
        overwrite: bool = False,
    ) -> None:
        """Add a named relation of one family.

        Args:
            kind: Which family this belongs to. `RelationKind.ALL` is not
                accepted.
            data: One row per related pair. The frame is held, not copied.
            to_type: Object types the targets may belong to. All must be in
                the store.
            from_type: Object types the sources may belong to. Required for an
                object-to-object relation, not allowed for an event-to-object
                relation.
            overwrite: Replace a relation of this name in the same family.

        Raises:
            ValueError: If `kind` is `RelationKind.ALL`.
            InputError: If `from_type` is missing or misplaced, the name is
                taken by the other family, the name exists in this one and
                overwrite is off, a named type is unknown, or a named column
                is missing.
        """
        if kind is RelationKind.ALL:
            raise ValueError(
                f"{kind!r} is not a family to add to. Pass RelationKind.E2O or "
                "RelationKind.O2O."
            )
        if kind is RelationKind.E2O and from_type is not None:
            raise InputError(
                f"Relation {name!r} is event-to-object and takes no from_type."
            )
        if kind is RelationKind.O2O and from_type is None:
            raise InputError(
                f"Relation {name!r} is object-to-object and needs a from_type "
                "naming the object types its sources may belong to."
            )

        sources = list(from_type or [])
        targets = list(to_type)
        self._check_relation(
            kind, name, data, from_id, to_id, sources + targets, overwrite
        )
        self._relations.of(kind)[name] = RelationTable(
            from_id=from_id,
            to_id=to_id,
            to_type=targets,
            data=data,
            from_type=sources if from_type is not None else None,
        )

    def _check_relation(
        self,
        kind: RelationKind,
        name: str,
        data: pd.DataFrame,
        from_id: str,
        to_id: str,
        object_types: list[str],
        overwrite: bool,
    ) -> None:
        """Raise for a relation the store cannot hold.

        Raises:
            InputError: If the name is taken by the other family, the name
                exists in this one and overwrite is off, a named object type
                is unknown, or a named column is missing.
        """
        other = RelationKind.O2O if kind is RelationKind.E2O else RelationKind.E2O
        if name in self._relations.of(other):
            raise InputError(
                f"Relation {name!r} already exists in the {other.word} family. "
                f"Give it another name."
            )
        if not overwrite and name in self._relations.of(kind):
            raise InputError(
                f"Relation {name!r} already exists in the {kind.word} family. "
                f"Pass overwrite=True to replace it."
            )
        objects = self._entities[EntityKind.OBJECT]
        unknown = set(object_types) - set(objects)
        if unknown:
            raise InputError(
                f"Relation {name!r} names {len(unknown)} unknown object type(s): "
                f"{sorted(unknown)}. Add those object types before the relation. "
                f"This store holds {sorted(objects)}."
            )
        missing = [
            f"{param} {column!r}"
            for param, column in (("from_id", from_id), ("to_id", to_id))
            if column not in data.columns
        ]
        if missing:
            raise InputError(
                f"Relation {name!r} names {', '.join(missing)}, which "
                f"the given data does not have. Its columns are "
                f"{sorted(map(str, data.columns))}."
            )

    def relations_of(self, kind: RelationKind) -> Mapping[str, RelationTable]:
        """Return one relation family, keyed by name and read-only."""
        return MappingProxyType(self._relations.of(kind))

    def relations(self) -> Iterator[tuple[str, RelationTable]]:
        """Yield every relation with its name, event-to-object first."""
        yield from self._relations.items()

    def replace_relation_data(self, relation: str, data: pd.DataFrame) -> None:
        """Replace the rows of one relation, keeping its endpoints and types.

        Raises:
            InputError: If no relation of that name exists.
        """
        self.get_relation(relation).data = data

    def get_relation(self, relation: str) -> RelationTable:
        """Return one relation table from either family.

        Raises:
            InputError: If no relation of that name exists.
        """
        for kind in RelationKind.families():
            family = self._relations.of(kind)
            if relation in family:
                return family[relation]
        raise InputError(
            f"Unknown relation {relation!r}. This store holds "
            f"{self.get_relation_types()}."
        )

    def get_relation_types(self, kind: RelationKind = RelationKind.ALL) -> list[str]:
        """Return the relation names of one family or both, event-to-object first."""
        wanted = RelationKind.families() if kind is RelationKind.ALL else (kind,)
        return [name for family in wanted for name in self._relations.of(family)]

    def rename_relation(self, old_name: str, new_name: str) -> None:
        """Rename a relation.

        Raises:
            InputError: If the new name is taken, or the old name is unknown.
        """
        held = self.get_relation_types()
        if new_name in held:
            raise InputError(
                f"Cannot rename relation: {new_name!r} already exists as a relation."
            )
        for kind in RelationKind.families():
            family = self._relations.of(kind)
            if old_name in family:
                family[new_name] = family.pop(old_name)
                return
        raise InputError(
            f"Cannot rename relation {old_name!r}, which this store does not "
            f"hold. It holds {sorted(held)}."
        )

    def add_object_changes(self, obj_type: str, data: pd.DataFrame) -> None:
        """Record the attribute changes of one object type.

        Args:
            obj_type: Object type the changes belong to. Must be in the store.
            data: One row per change. Must carry the OCEL columns `ocel:oid`,
                `ocel:type`, `ocel:timestamp` and `ocel:field`. The frame is
                held, not copied.

        Raises:
            ValueError: If the store is not in OCEL mode.
            InputError: If the object type is not in the store, or a required
                column is missing.
        """
        self._require_ocel_mode("add_object_changes")
        self._require(EntityKind.OBJECT, obj_type, "record changes of")
        required = {"ocel:oid", "ocel:type", "ocel:timestamp", "ocel:field"}
        missing = required - set(data.columns)
        if missing:
            raise InputError(
                f"Changes of object type {obj_type!r} are missing the required "
                f"column(s) {sorted(missing)}."
            )
        self._object_changes[obj_type] = data

    def get_object_changes(self, obj_type: str) -> pd.DataFrame:
        """Return the recorded attribute changes of one object type.

        Raises:
            ValueError: If the store is not in OCEL mode.
            InputError: If nothing was recorded for the type.
        """
        self._require_ocel_mode("get_object_changes")
        if obj_type not in self._object_changes:
            raise InputError(f"No object changes recorded for type {obj_type!r}.")
        return self._object_changes[obj_type]

    def get_object_change_types(self) -> list[str]:
        """Return the object types that have recorded changes.

        Raises:
            ValueError: If the store is not in OCEL mode.
        """
        self._require_ocel_mode("get_object_change_types")
        return list(self._object_changes)

    def replace_object_changes(self, changes: dict[str, pd.DataFrame]) -> None:
        """Replace all recorded changes at once, keyed by object type.

        Raises:
            ValueError: If the store is not in OCEL mode.
        """
        self._require_ocel_mode("replace_object_changes")
        self._object_changes = dict(changes)

    def _require_ocel_mode(self, method: str) -> None:
        if not self._ocel_mode:
            raise ValueError(
                f"{method} requires ocel_mode=True. "
                "Create the OCEDStore with OCEDStore(ocel_mode=True)."
            )

    def _set_object_changes(self, obj_type: str, data: pd.DataFrame) -> None:
        self._object_changes[obj_type] = data

    def replace_tables(
        self,
        entities: Mapping[EntityKind, dict[str, EntityTable]],
        relations: Relations,
    ) -> None:
        """Replace every entity and relation table.

        The ids in use are recounted.

        Raises:
            InputError: If one id is used more than once. The store keeps the
                new tables.
        """
        self._entities = {kind: dict(entities.get(kind, {})) for kind in EntityKind}
        self._relations = relations
        self.update_used_ids()

    def tables(self, *, changes: bool = True) -> Iterator[TableView]:
        """Yield every table in the store, with its structural columns named.

        Yields:
            The event tables, then the object tables, then the change tables.
        """
        for kind in EntityKind:
            for name in list(self._entities[kind]):
                table = self._entities[kind][name]
                yield TableView(
                    frame=table.data,
                    name=name,
                    kind=kind,
                    key_columns=table.key_columns,
                    origin=f"{kind.word} {name!r}",
                    id_column=table.id,
                    replace=partial(self._set_data, kind, name),
                )
        if not (changes and self._ocel_mode):
            return
        for obj_type in list(self._object_changes):
            frame = self._object_changes[obj_type]
            yield TableView(
                frame=frame,
                name=obj_type,
                kind=None,
                key_columns=frozenset(
                    str(c) for c in frame.columns if str(c).startswith(OCEL_PREFIX)
                ),
                origin=f"the history of object type {obj_type!r}",
                id_column=None,
                replace=partial(self._set_object_changes, obj_type),
            )

    def update_used_ids(self) -> None:
        """Recompute the set of ids in use. Rows with no id are skipped.

        Raises:
            InputError: If one id is used more than once.
        """
        ids: list[object] = []
        for kind in EntityKind:
            for table in self._entities[kind].values():
                ids.extend(table.data[table.id].dropna())
        unique_ids = set(ids)
        if len(unique_ids) != len(ids):
            repeated = sorted(str(i) for i, n in Counter(ids).items() if n > 1)
            raise InputError(
                f"{len(repeated)} id(s) are used more than once, such as "
                f"{repeated[:5]}. Give each row its own id, for example by "
                "prefixing ids with their type."
            )
        self._used_ids = unique_ids


def attribute_owners_by_kind(
    data: ReadableStore,
) -> dict[str, dict[EntityKind, list[str]]]:
    """Return each attribute with the sorted types carrying a value of it, per kind.

    A change table counts as its object type.
    """
    owners: dict[str, dict[EntityKind, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for view in data.tables():
        kind = view.kind if view.kind is not None else EntityKind.OBJECT
        for column in view.filled_attributes:
            owners[column][kind].add(view.name)
    return {
        attribute: {kind: sorted(types) for kind, types in by_kind.items()}
        for attribute, by_kind in owners.items()
    }
