"""The in-memory model every stage of a build reads and writes.

`OCEDStore` holds events grouped by activity, objects grouped by type, the
relations between them, and optionally the attribute change history of OCEL
2.0. Loaders fill it, cleaning rewrites it, and emitters read it. The tables
themselves are private, so every stage goes through the methods here.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType

import pandas as pd

from semanticlog.core.iri import DEFAULT_IRI as _DEFAULT_IRI
from semanticlog.data.dtypes import infer_dtypes
from semanticlog.data.tables import (
    EntityKind as EntityKind,
)
from semanticlog.data.tables import (
    EventTable as EventTable,
)
from semanticlog.data.tables import (
    ObjectTable as ObjectTable,
)
from semanticlog.data.tables import (
    RelationKind as RelationKind,
)
from semanticlog.data.tables import (
    Relations as Relations,
)
from semanticlog.data.tables import (
    RelationTable as RelationTable,
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

    def write(self, path: str | Path) -> Path:
        """Write the store to a directory of Parquet tables plus a manifest.

        Args:
            path: Directory to write. It is replaced if it already holds a
                store, and refused if it holds something else.

        Returns:
            The directory written.
        """
        from semanticlog.data.persist import write_store

        return write_store(self, path)

    @classmethod
    def read(cls, path: str | Path) -> OCEDStore:
        """Read a store written by `write`.

        Args:
            path: Directory holding the store.

        Returns:
            The store, with the dtypes it was written with.
        """
        from semanticlog.data.persist import read_store

        return read_store(path)

    def save_backmap(self, path: str | Path) -> None:
        """Write the cleaned name to original name map as JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._backmap, f, ensure_ascii=False, indent=2)

    def get_types(
        self, ids: list[str] | pd.Series, kind: EntityKind | None = None
    ) -> list[str]:
        """Find which types the given ids belong to.

        Args:
            ids: Ids to look up.
            kind: Restrict the search to events or to objects. None searches
                both, objects first.

        Returns:
            Every type holding at least one of the ids.
        """
        if isinstance(ids, list):
            ids = pd.Series(ids)
        types = []
        if kind is not EntityKind.EVENT:
            for object_type, table in self._objects.items():
                if ids.isin(table.data[table.id]).any():
                    types.append(object_type)
        if kind is not EntityKind.OBJECT:
            for activity, event_table in self._events.items():
                if ids.isin(event_table.data[event_table.id]).any():
                    types.append(activity)

        return types

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
        self,
        event_type: str,
        id_col: str,
        time_col: str,
        data: pd.DataFrame,
        overwrite: bool = False,
        infer: bool = True,
    ) -> None:
        """Add the events of one activity.

        Args:
            event_type: Activity name.
            id_col: Column holding the event id.
            time_col: Column holding the timestamp.
            data: One row per event. Remaining columns become attributes.
            overwrite: Replace an activity of this name, and accept ids that
                are already in the store.
            infer: Coerce the attribute columns to their apparent dtypes.

        Raises:
            ValueError: If a named column is missing, the activity exists, or
                the ids are already used.
        """
        if not {id_col, time_col}.issubset(data.columns):
            raise ValueError(
                "The given event column headers are not all included in the data."
            )
        if not overwrite and event_type in self.get_event_types():
            raise ValueError(
                "Events data already exists! Set overwrite to True if you want to disregard this."
            )
        if not overwrite and not self.ids_not_used(data[id_col].unique()):
            raise ValueError("IDs in this data are already used.")

        if infer:
            data = infer_dtypes(data, ignore_cols=[id_col], label=event_type)
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

    def add_object(
        self,
        obj_type: str,
        id_col: str,
        data: pd.DataFrame,
        overwrite: bool = False,
        infer: bool = True,
    ) -> None:
        """Add the objects of one type.

        Args:
            obj_type: Object type name.
            id_col: Column holding the object id.
            data: One row per object. Remaining columns become attributes.
            overwrite: Replace an object type of this name, and accept ids
                that are already in the store.
            infer: Coerce the attribute columns to their apparent dtypes.

        Raises:
            ValueError: If the id column is missing, the type exists, or the
                ids are already used.
        """
        if id_col not in data.columns:
            raise ValueError(
                f"The given object column header ({id_col}) is not included in the data."
            )
        if not overwrite and obj_type in self._objects:
            raise ValueError(
                f"Object ({obj_type}) already exists! Set overwrite to True if you want to disregard this."
            )
        if not overwrite and not self.ids_not_used(data[id_col].unique()):
            raise ValueError("IDs in this data are already used.")

        if infer:
            data = infer_dtypes(data, ignore_cols=[id_col], label=obj_type)
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
        """
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
        if relation in self._relations.object_to_object:
            raise ValueError(
                f"The relation ({relation}) already exists as a object_to_object relation!"
            )
        if not overwrite and relation in self._relations.event_to_object:
            raise ValueError(
                f"Relation ({relation}) already exists! Set overwrite to True if you want to disregard this."
            )
        if not set(to_type).issubset(set(self.get_object_types())):
            raise ValueError(
                f"Can not instantiate relation ({relation}). Unknown object type in ({to_type})!"
            )
        for param, col in (("from_id_col", from_id_col), ("to_id_col", to_id_col)):
            if col not in data.columns:
                raise ValueError(
                    f"{param} '{col}' not found in relation data columns: "
                    f"{list(data.columns)}"
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
        if relation in self._relations.event_to_object:
            raise ValueError(
                f"The relation ({relation}) already exists as a event_to_object relation!"
            )
        if not overwrite and relation in self._relations.object_to_object:
            raise ValueError(
                f"Relation ({relation}) already exists! Set overwrite to True if you want to disregard this."
            )
        if not (
            set(to_type).issubset(set(self.get_object_types()))
            and set(from_type).issubset(set(self.get_object_types()))
        ):
            raise ValueError(
                f"Can not instantiate relation ({relation}). Unknown object type in ({to_type + from_type})!"
            )

        missing = [c for c in (from_id_col, to_id_col) if c not in data.columns]
        if missing:
            raise ValueError(
                f"Can not instantiate relation ({relation}). Column(s) "
                f"{missing} are not in the given data ({list(data.columns)})! "
                "The e2o variant has always checked this; without it the failure "
                "surfaced as a KeyError at build time."
            )

        self._relations.object_to_object[relation] = RelationTable(
            from_id=from_id_col,
            from_type=list(from_type),
            to_id=to_id_col,
            to_type=list(to_type),
            data=data,
        )

    def get_e2o(self) -> dict[str, RelationTable]:
        """Return the event-to-object relations, keyed by name."""
        return self._relations.event_to_object

    def get_o2o(self) -> dict[str, RelationTable]:
        """Return the object-to-object relations, keyed by name."""
        return self._relations.object_to_object

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

        def renamed(types: list[str] | None) -> list[str] | None:
            if types is None:
                return None
            return [new_name if t == old_name else t for t in types]

        for relation in self._relations.event_to_object.values():
            relation.to_type = renamed(relation.to_type)
        for relation in self._relations.object_to_object.values():
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
