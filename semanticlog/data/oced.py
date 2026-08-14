"""The in-memory model every stage of a build reads and writes.

`OCEDStore` holds events grouped by activity, objects grouped by type, the
relations between them, and optionally the attribute change history of OCEL
2.0. Loaders fill it, cleaning rewrites it, and emitters read it. The tables
themselves are private, so every stage goes through the methods here.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from functools import partial
from types import MappingProxyType

import pandas as pd

from semanticlog.core.vocabulary import DEFAULT_IRI as _DEFAULT_IRI
from semanticlog.data.tables import (
    OCEL_PREFIX,
    EntityKind,
    EventTable,
    ObjectTable,
    RelationKind,
    Relations,
    RelationTable,
    TableView,
)


class OCEDStore:
    """Object-centric event data, held as typed tables.

    Ids are unique across the whole store, not only within a type, because an
    id becomes an individual in the ontology and two entities sharing one id
    would become one individual.
    """

    def __init__(self, ocel_mode: bool = False):
        """Create an empty store.

        Args:
            ocel_mode: Whether this store may hold OCEL 2.0 attribute change
                history. The change methods refuse to run without it, so that
                a store built from a source that has no history cannot appear
                to have an empty one.
        """
        self._events: dict[str, EventTable] = {}
        self._objects: dict[str, ObjectTable] = {}
        self._relations = Relations()
        self._used_ids: set = set()
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
            iri_or_local_name: A local name, or a full IRI to strip first.
            base_iri: Prefix to strip when a full IRI is given.

        Returns:
            The original name, or None if the name was never changed.
        """
        local_name = iri_or_local_name.removeprefix(base_iri)
        return self._backmap.get(local_name)

    @property
    def ocel_mode(self) -> bool:
        """Whether this store may hold attribute change history."""
        return self._ocel_mode

    def add_object_changes(self, obj_type: str, data: pd.DataFrame) -> None:
        """Record the attribute changes of one object type.

        Args:
            obj_type: Object type the changes belong to.
            data: One row per change. Must carry the OCEL columns `ocel:oid`,
                `ocel:type`, `ocel:timestamp` and `ocel:field`.

        Raises:
            ValueError: If the store is not in OCEL mode, or a required column
                is missing.
        """
        self._require_ocel_mode("add_object_changes")
        required = {"ocel:oid", "ocel:type", "ocel:timestamp", "ocel:field"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f"object_changes data is missing required columns: {missing}"
            )
        self._object_changes[obj_type] = data.copy()

    def get_object_changes(self, obj_type: str) -> pd.DataFrame:
        """Return the recorded attribute changes of one object type.

        Raises:
            ValueError: If the store is not in OCEL mode, or nothing was
                recorded for the type.
        """
        self._require_ocel_mode("get_object_changes")
        if obj_type not in self._object_changes:
            raise ValueError(f"No object changes recorded for type '{obj_type}'.")
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

    def add_event(
        self, event_type: str, id_col: str, time_col: str, data: pd.DataFrame
    ) -> None:
        """Add the events of one activity.

        Args:
            event_type: Activity name.
            id_col: Column holding the event id.
            time_col: Column holding the timestamp.
            data: One row per event. Remaining columns become attributes.
                Dtypes are taken as given, so a loader infers them first.

        Raises:
            ValueError: If a named column is missing, the activity exists, or
                the ids are already used. The store is left as it was.
        """
        if not {id_col, time_col}.issubset(data.columns):
            raise ValueError(
                "The given event column headers are not all included in the data."
            )
        if event_type in self._events:
            raise ValueError(f"Events of activity {event_type!r} already exist.")
        if not self.ids_not_used(data[id_col].unique()):
            raise ValueError("IDs in this data are already used.")

        self._events[event_type] = EventTable(id=id_col, time=time_col, data=data)
        self.update_used_ids()

    def get_event_table(self, event_type: str) -> EventTable:
        """Return the event table of one activity."""
        return self._events[event_type]

    def get_event_types(self) -> list[str]:
        """Return every activity name in the store."""
        return list(self._events)

    def replace_event_data(self, event_type: str, data: pd.DataFrame) -> None:
        """Replace the rows of one activity, keeping its column roles.

        Raises:
            ValueError: If the activity is not in the store.
        """
        if event_type not in self._events:
            raise ValueError(f"Unknown activity {event_type}")
        self._events[event_type].data = data

    def add_object(self, obj_type: str, id_col: str, data: pd.DataFrame) -> None:
        """Add the objects of one type.

        Args:
            obj_type: Object type name.
            id_col: Column holding the object id.
            data: One row per object. Remaining columns become attributes.
                Dtypes are taken as given, so a loader infers them first.

        Raises:
            ValueError: If the id column is missing, the type exists, or the
                ids are already used. The store is left as it was.
        """
        if id_col not in data.columns:
            raise ValueError(
                f"The given object column header ({id_col}) is not included in the data."
            )
        if obj_type in self._objects:
            raise ValueError(f"Objects of type {obj_type!r} already exist.")
        if not self.ids_not_used(data[id_col].unique()):
            raise ValueError("IDs in this data are already used.")

        self._objects[obj_type] = ObjectTable(id=id_col, data=data)
        self.update_used_ids()

    def get_object_tables(self) -> Mapping[str, ObjectTable]:
        """Return every object table, keyed by type and read-only."""
        return MappingProxyType(self._objects)

    def get_object_types(self) -> list[str]:
        """Return every object type name in the store."""
        return list(self._objects)

    def get_object_table(self, obj_type: str) -> ObjectTable:
        """Return the object table of one type."""
        return self._objects[obj_type]

    def replace_object_data(self, obj_type: str, data: pd.DataFrame) -> None:
        """Replace the rows of one object type, keeping its column roles.

        Raises:
            ValueError: If the object type is not in the store.
        """
        if obj_type not in self._objects:
            raise ValueError(f"Unknown object {obj_type}")
        self._objects[obj_type].data = data

    def get_most_common_objects(self, n: int = 1) -> list[str]:
        """Return the object types with the most instances.

        This is how a build picks case concepts when none were named.

        Args:
            n: How many types to return.

        Returns:
            Type names, most instances first.

        Raises:
            ValueError: If n is negative, which would otherwise be read as a
                slice from the end and quietly drop the smallest types.
        """
        if n < 0:
            raise ValueError(f"get_most_common_objects needs n >= 0, got {n}.")
        objects = self.get_object_tables()
        return sorted(objects, key=lambda k: len(objects[k].data), reverse=True)[:n]

    def add_event_to_object_relation(
        self,
        relation: str,
        from_id_col: str,
        to_id_col: str,
        to_type: list[str],
        data: pd.DataFrame,
        overwrite: bool = False,
    ) -> None:
        """Add a named relation from events to objects.

        Args:
            relation: Relation name, which becomes a property in the ontology.
            from_id_col: Column holding the event id.
            to_id_col: Column holding the object id.
            to_type: Object types the targets may belong to. All of them must
                already be in the store.
            data: One row per related pair.
            overwrite: Replace a relation of this name.

        Raises:
            ValueError: If the name is taken by an object-to-object relation,
                the name exists and overwrite is off, a target type is
                unknown, or a named column is missing.
        """
        self._check_relation(
            relation, RelationKind.E2O, from_id_col, to_id_col, to_type, data, overwrite
        )
        self._relations.event_to_object[relation] = RelationTable(
            from_id=from_id_col,
            to_id=to_id_col,
            to_type=list(to_type),
            data=data,
        )

    def add_object_to_object_relation(
        self,
        relation: str,
        from_id_col: str,
        from_type: list[str],
        to_id_col: str,
        to_type: list[str],
        data: pd.DataFrame,
        overwrite: bool = False,
    ) -> None:
        """Add a named relation between objects.

        Args:
            relation: Relation name, which becomes a property in the ontology.
            from_id_col: Column holding the source object id.
            from_type: Object types the sources may belong to.
            to_id_col: Column holding the target object id.
            to_type: Object types the targets may belong to.
            data: One row per related pair.
            overwrite: Replace a relation of this name.

        Raises:
            ValueError: If the name is taken by an event-to-object relation,
                the name exists and overwrite is off, a named type is unknown,
                or a named column is missing.
        """
        self._check_relation(
            relation,
            RelationKind.O2O,
            from_id_col,
            to_id_col,
            [*to_type, *from_type],
            data,
            overwrite,
        )
        self._relations.object_to_object[relation] = RelationTable(
            from_id=from_id_col,
            from_type=list(from_type),
            to_id=to_id_col,
            to_type=list(to_type),
            data=data,
        )

    def _check_relation(
        self,
        relation: str,
        kind: RelationKind,
        from_id_col: str,
        to_id_col: str,
        object_types: list[str],
        data: pd.DataFrame,
        overwrite: bool,
    ) -> None:
        """Refuse a relation the store could not hold.

        Both families share one namespace of names, since each relation
        becomes a property of one ontology, so a name taken by the other
        family is refused whichever family is adding it.

        Raises:
            ValueError: If the name is taken by the other family, the name
                exists in this one and overwrite is off, a named object type
                is unknown, or a named column is missing.
        """
        own, other = self._relations.event_to_object, self._relations.object_to_object
        own_name, other_name = "event_to_object", "object_to_object"
        if kind is RelationKind.O2O:
            own, other = other, own
            own_name, other_name = other_name, own_name

        if relation in other:
            raise ValueError(
                f"The relation ({relation}) already exists as a {other_name} relation!"
            )
        if not overwrite and relation in own:
            raise ValueError(
                f"Relation ({relation}) already exists! Set overwrite to True if you want to disregard this."
            )
        unknown = set(object_types) - set(self._objects)
        if unknown:
            raise ValueError(
                f"Can not instantiate relation ({relation}). Unknown object type in ({object_types})!"
            )
        missing = [
            f"{param} {column!r}"
            for param, column in (
                ("from_id_col", from_id_col),
                ("to_id_col", to_id_col),
            )
            if column not in data.columns
        ]
        if missing:
            raise ValueError(
                f"Can not instantiate relation ({relation}). {', '.join(missing)} "
                f"not in the given data ({list(data.columns)})!"
            )

    def get_e2o(self) -> Mapping[str, RelationTable]:
        """Return the event-to-object relations, keyed by name and read-only."""
        return MappingProxyType(self._relations.event_to_object)

    def get_o2o(self) -> Mapping[str, RelationTable]:
        """Return the object-to-object relations, keyed by name and read-only."""
        return MappingProxyType(self._relations.object_to_object)

    def relations(self) -> Iterator[tuple[str, RelationTable]]:
        """Yield every relation with its name, event-to-object first."""
        yield from self._relations.event_to_object.items()
        yield from self._relations.object_to_object.items()

    def replace_relation_data(self, relation: str, data: pd.DataFrame) -> None:
        """Replace the rows of one relation, keeping its endpoints and types.

        Raises:
            ValueError: If no relation of that name exists.
        """
        self.get_relation(relation).data = data

    def get_relation(self, relation: str) -> RelationTable:
        """Return one relation table from either family.

        Raises:
            ValueError: If no relation of that name exists.
        """
        if relation in self._relations.event_to_object:
            return self._relations.event_to_object[relation]
        if relation in self._relations.object_to_object:
            return self._relations.object_to_object[relation]
        raise ValueError(f"Relation {relation} does not exist!")

    def get_relation_types(self, kind: RelationKind = RelationKind.ALL) -> list[str]:
        """Return relation names, optionally of one family only.

        Args:
            kind: Which family to list.

        Returns:
            Relation names, event-to-object first.
        """
        relations: list[str] = []
        if kind in (RelationKind.E2O, RelationKind.ALL):
            relations += list(self._relations.event_to_object)
        if kind in (RelationKind.O2O, RelationKind.ALL):
            relations += list(self._relations.object_to_object)
        return relations

    def replace_tables(
        self,
        events: dict[str, EventTable],
        objects: dict[str, ObjectTable],
        relations: Relations,
    ) -> None:
        """Swap in a whole new set of tables.

        Cleaning rebuilds every table and uses this to put the result back.
        Ids are not rechecked, so call `update_used_ids` afterwards.
        """
        self._events = events
        self._objects = objects
        self._relations = relations

    def rename_relation(self, old_name: str, new_name: str) -> None:
        """Rename a relation in whichever family holds it.

        Raises:
            ValueError: If the new name is taken, or the old name is unknown.
        """
        all_relation_names = (
            self._relations.event_to_object.keys()
            | self._relations.object_to_object.keys()
        )
        if new_name in all_relation_names:
            raise ValueError(
                f"Cannot rename relation: '{new_name}' already exists as a relation."
            )
        for family in (
            self._relations.event_to_object,
            self._relations.object_to_object,
        ):
            if old_name in family:
                family[new_name] = family.pop(old_name)
                return
        raise ValueError("Cannot rename relation that does not exist!")

    def rename_object(self, old_name: str, new_name: str) -> None:
        """Rename an object type everywhere it is referenced.

        The type list of every relation and any recorded change table follow
        the rename, so nothing keeps pointing at the old name.

        Raises:
            ValueError: If the old name is unknown, or the new name is taken.
        """
        if old_name not in self._objects:
            raise ValueError("Cannot rename object that does not exist!")
        if new_name in self._objects:
            raise ValueError(
                f"Cannot rename object {old_name!r}: {new_name!r} already exists "
                "as an object type and renaming onto it would discard its rows."
            )

        self._objects[new_name] = self._objects.pop(old_name)

        def renamed(types: list[str]) -> list[str]:
            return [new_name if t == old_name else t for t in types]

        for relation in self._relations.event_to_object.values():
            relation.to_type = renamed(relation.to_type)
        for relation in self._relations.object_to_object.values():
            if relation.from_type is not None:
                relation.from_type = renamed(relation.from_type)
            relation.to_type = renamed(relation.to_type)

        if self._ocel_mode and old_name in self._object_changes:
            self._object_changes[new_name] = self._object_changes.pop(old_name)

    def rename_activity(self, old_name: str, new_name: str) -> None:
        """Rename an activity.

        Raises:
            ValueError: If the old name is unknown, or the new name is taken.
        """
        if old_name not in self._events:
            raise ValueError("Cannot rename activity that does not exist!")
        if new_name in self._events:
            raise ValueError(
                f"Cannot rename activity {old_name!r}: {new_name!r} already exists "
                "as an activity and renaming onto it would discard its rows."
            )
        self._events[new_name] = self._events.pop(old_name)

    def tables(self, *, changes: bool = True) -> Iterator[TableView]:
        """Yield every table in the store, with its structural columns named.

        Cleaning and the checks before a build walk the whole store rather
        than one half of it, so which columns carry structure rather than an
        attribute is decided here once for all of them.

        Args:
            changes: Include the attribute change tables. A pass that is only
                about entities passes False.

        Yields:
            The event tables, then the object tables, then the change tables.
        """
        for activity in list(self._events):
            table = self._events[activity]
            yield TableView(
                frame=table.data,
                name=activity,
                kind=EntityKind.EVENT,
                key_columns=frozenset({table.id, table.time}),
                origin=f"activity {activity!r}",
                id_column=table.id,
                replace=partial(self._set_event_data, activity),
            )
        for obj_type in list(self._objects):
            obj = self._objects[obj_type]
            yield TableView(
                frame=obj.data,
                name=obj_type,
                kind=EntityKind.OBJECT,
                key_columns=frozenset({obj.id}),
                origin=f"object type {obj_type!r}",
                id_column=obj.id,
                replace=partial(self._set_object_data, obj_type),
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

    def _set_event_data(self, event_type: str, data: pd.DataFrame) -> None:
        self._events[event_type].data = data

    def _set_object_data(self, obj_type: str, data: pd.DataFrame) -> None:
        self._objects[obj_type].data = data

    def _set_object_changes(self, obj_type: str, data: pd.DataFrame) -> None:
        self._object_changes[obj_type] = data

    def get_ids(self, kind: EntityKind, entity_type: str) -> Iterable:
        """Return the distinct ids of one activity or object type.

        Args:
            kind: Whether the type names an activity or an object type.
            entity_type: The activity or object type.

        Returns:
            The ids, in order of first appearance.

        Raises:
            ValueError: If the type is not in the store.
        """
        if kind is EntityKind.EVENT:
            if entity_type not in self.get_event_types():
                raise ValueError(f"Unknown activity {entity_type}")
            table: EventTable | ObjectTable = self._events[entity_type]
        else:
            if entity_type not in self.get_object_types():
                raise ValueError(f"Unknown object {entity_type}")
            table = self._objects[entity_type]
        return table.data[table.id].unique()

    def update_used_ids(self) -> None:
        """Recompute the set of ids in use, and reject duplicates.

        Raises:
            ValueError: If one id appears in more than one table, which would
                merge two entities into one individual.
        """
        ids: list = []
        for activity in self._events:
            ids.extend(self.get_ids(EntityKind.EVENT, activity))
        for object_type in self._objects:
            ids.extend(self.get_ids(EntityKind.OBJECT, object_type))
        unique_ids = set(ids)
        if len(unique_ids) != len(ids):
            raise ValueError("Duplicated ids!")
        self._used_ids = unique_ids

    def ids_not_used(self, ids: Iterable) -> bool:
        """Report whether none of the given ids are in the store yet."""
        return self._used_ids.isdisjoint(ids)
