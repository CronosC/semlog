from __future__ import annotations

import copy
import json
import logging
import pickle
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd
import pm4py as pm

from semanticlog.data.dtypes import infer_dtypes

_DEFAULT_IRI: str = "https://oced#"


class EntityKind(str, Enum):
    """Whether an operation targets events or objects."""

    EVENT = "event"
    OBJECT = "object"


class RelationKind(str, Enum):
    """Which relation family an operation targets."""

    ALL = "all"
    E2O = "e2o"
    O2O = "o2o"


@dataclass
class EventTable:
    """One activity's events.

    Attributes:
        id: Column holding the unique event identifier.
        time: Column holding the event timestamp.
        data: The event rows, including ``id``, ``time`` and any attribute columns.
    """

    id: str
    time: str
    data: pd.DataFrame


@dataclass
class ObjectTable:
    """One object type's objects.

    Attributes:
        id: Column holding the unique object identifier.
        data: The object rows, including ``id`` and any attribute columns.
    """

    id: str
    data: pd.DataFrame


@dataclass
class RelationTable:
    """One named relation between events and objects, or between objects.

    ``from_type`` is ``None`` for event-to-object relations, where the source is
    always an event rather than a named object type.

    Attributes:
        from_id: Column holding the source identifier.
        to_id: Column holding the target identifier.
        to_type: Object types the target identifier may belong to.
        data: The relation rows.
        from_type: Object types the source identifier may belong to (o2o only).
    """

    from_id: str
    to_id: str
    to_type: list[str]
    data: pd.DataFrame
    from_type: list[str] | None = None


@dataclass
class Relations:
    """The two relation families, keyed by relation name within each."""

    event_to_object: dict[str, RelationTable] = field(default_factory=dict)
    object_to_object: dict[str, RelationTable] = field(default_factory=dict)


class OCEDStore:
    """
    Holds object-centric event data.

    The class maintains:
      - Events: :class:`EventTable` per activity.
      - Objects: :class:`ObjectTable` per object type.
      - Relations: :class:`RelationTable` per named relation, split into
        event-to-object and object-to-object families.
    """

    def __init__(self, ocel_mode: bool = False):
        """
        Constructor to initialize empty structures for events, objects, and relations.

        Args:
            ocel_mode: When True, enables storage of OCEL 2.0 extended data such as
                object attribute change history. Required before calling
                ``add_object_changes`` or ``get_object_changes``.
        """
        self._events: dict[str, EventTable] = {}
        self._objects: dict[str, ObjectTable] = {}
        self._relations = Relations()
        self._used_ids: set = set()
        self._ocel_mode = ocel_mode
        self._backmap: dict[str, str] = {}
        self._object_changes: dict[str, pd.DataFrame] = {}

    # BACKMAP

    @property
    def backmap(self) -> dict[str, str]:
        """Mapping from post-clean IRI local name to original pre-clean name (IDs and schema names)."""
        return self._backmap

    @backmap.setter
    def backmap(self, mapping: dict[str, str]) -> None:
        self._backmap = mapping

    def get_original(
        self, iri_or_local_name: str, base_iri: str = _DEFAULT_IRI
    ) -> str | None:
        """
        Return the original pre-cleaning name for an IRI or local name.

        Covers both entity IDs and schema names (activities, object types, attributes).

        Args:
            iri_or_local_name: A full IRI (e.g. ``https://oced#order__123``) or
                the local name alone (e.g. ``order__123``).
            base_iri: IRI prefix to strip before lookup. Defaults to
                ``"https://oced#"``.

        Returns:
            The original name as it appeared before cleaning, or ``None``
            if the local name is not in the backmap.
        """
        local_name = iri_or_local_name.removeprefix(base_iri)
        return self._backmap.get(local_name)

    def get_originals(
        self, iri_or_local_names: list[str], base_iri: str = _DEFAULT_IRI
    ) -> list[str | None]:
        """
        Return original pre-cleaning names for a list of IRIs or local names.

        Args:
            iri_or_local_names: List of full IRIs or local names.
            base_iri: IRI prefix to strip before lookup.

        Returns:
            List of original names in the same order, with ``None`` for entries
            not found in the backmap.
        """
        return [self.get_original(name, base_iri) for name in iri_or_local_names]

    def save(self, path: str | Path) -> None:
        """
        Persist the full OCEDStore to a pickle file.

        Args:
            path: Destination file path (e.g. ``store.pkl``).
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> OCEDStore:
        """
        Load an OCEDStore from a pickle file produced by :meth:`save`.

        Args:
            path: Path to the pickle file.

        Returns:
            Restored ``OCEDStore`` instance, including ``_backmap``.
        """
        with open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301

    def save_backmap(self, path: str | Path) -> None:
        """
        Save the backmap as a JSON file.

        Args:
            path: Destination file path (e.g. ``backmap.json``).
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._backmap, f, ensure_ascii=False, indent=2)

    def load_backmap(self, path: str | Path) -> None:
        """
        Load backmap from a JSON file, replacing any existing backmap.

        Args:
            path: Path to a JSON file produced by :meth:`save_backmap`.
        """
        with open(path, encoding="utf-8") as f:
            self._backmap = json.load(f)

    # MISCELLANEOUS

    def get_types(self, ids: list[str] | pd.Series) -> list[str]:
        """
        Find the object or event types that include any of the given ids.
        """
        if isinstance(ids, list):
            ids = pd.Series(ids)
        types = []
        for object_type, table in self._objects.items():
            if ids.isin(table.data[table.id]).any():
                types.append(object_type)
        for activity, event_table in self._events.items():
            if ids.isin(event_table.data[event_table.id]).any():
                types.append(activity)

        return types

    def get_log(self, case_concept: str) -> pd.DataFrame:
        """
        Generate a classical event log (as a pandas DataFrame) from the object-centric
        event data via a case-concept (object).

        Parameters:
            case_concept (str): The object type to be used as the case-concept in the event log.

        Returns:
            pd.DataFrame: A pm4py event log.

        Raises:
            ValueError: If the provided case_concept is not among the known object types.
        """
        if case_concept not in self.get_object_types():
            raise ValueError(f"Unknown object type: {case_concept}")

        # get all relevant e2o relations
        data_list = []
        for relation in self._relations.event_to_object.values():
            if case_concept in relation.to_type:
                data_list.append(
                    relation.data.rename(
                        columns={
                            relation.to_id: case_concept,
                            relation.from_id: "event_id",
                        }
                    )
                )

        if not data_list:
            raise ValueError(
                f"No event-to-object relations reference object type '{case_concept}'."
            )
        rel_data = pd.concat(data_list, ignore_index=True)
        valid_case_ids = set(self.get_ids(EntityKind.OBJECT, case_concept))
        dropped_ids = set(rel_data[case_concept]) - valid_case_ids
        rel_data = rel_data[rel_data[case_concept].isin(valid_case_ids)]

        if dropped_ids:
            self._warn_unknown_case_ids(case_concept, dropped_ids)

        events_data_list = []
        for event_type, event_table in self._events.items():
            event_data = event_table.data[[event_table.id, event_table.time]].copy()
            event_data["concept:name"] = event_type
            event_data = event_data.rename(
                columns={event_table.id: "event_id", event_table.time: "time:timestamp"}
            )
            events_data_list.append(event_data)

        if not events_data_list:
            raise ValueError("No event data available.")
        events_data = pd.concat(events_data_list, ignore_index=True)
        log = pd.merge(events_data, rel_data, on="event_id", how="inner")

        log = pm.format_dataframe(
            log,
            case_id=case_concept,
            activity_key="concept:name",
            timestamp_key="time:timestamp",
        )

        log = log.set_index("event_id")
        log = log[["case:concept:name", "concept:name", "time:timestamp"]].sort_values(
            by=["case:concept:name", "time:timestamp"], ascending=True
        )

        return log

    def _warn_unknown_case_ids(self, case_concept: str, dropped_ids: set) -> None:
        all_object_ids: set = set()
        for table in self._objects.values():
            all_object_ids.update(table.data[table.id].unique())
        genuinely_missing = dropped_ids - all_object_ids
        if genuinely_missing:
            logging.warning(
                f"get_log: {len(genuinely_missing)} object IDs not found in any object store "
                f"for case concept '{case_concept}' "
                f"(sample: {sorted(str(x) for x in genuinely_missing)[:5]})"
            )

    # OCEL MODE

    def has_ocel_mode(self) -> bool:
        """Returns True if this store was created with ocel_mode=True."""
        return self._ocel_mode

    def add_object_changes(self, obj_type: str, data: pd.DataFrame) -> None:
        """
        Store attribute change history for an object type.

        Args:
            obj_type: The object type these changes belong to.
            data: DataFrame with at least columns ``ocel:oid``, ``ocel:type``,
                ``ocel:timestamp``, and ``ocel:field``. Additional columns hold
                the attribute values (each column named after its attribute).

        Raises:
            ValueError: If the store was not created with ``ocel_mode=True``, or
                if required columns are missing from ``data``.
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
        """
        Return the attribute change history for the given object type.

        Args:
            obj_type: The object type to retrieve changes for.

        Returns:
            DataFrame with object change records for the given type.

        Raises:
            ValueError: If the store was not created with ``ocel_mode=True``, or
                if ``obj_type`` has no recorded changes.
        """
        self._require_ocel_mode("get_object_changes")
        if obj_type not in self._object_changes:
            raise ValueError(f"No object changes recorded for type '{obj_type}'.")
        return self._object_changes[obj_type]

    def get_object_change_types(self) -> list[str]:
        """
        Return the list of object types that have recorded attribute change history.

        Raises:
            ValueError: If the store was not created with ``ocel_mode=True``.
        """
        self._require_ocel_mode("get_object_change_types")
        return list(self._object_changes)

    def replace_object_changes(self, changes: dict[str, pd.DataFrame]) -> None:
        """
        Replace the whole attribute change history, keyed by object type.

        Intended for pipeline stages that rewrite every table at once, such as
        :func:`~semanticlog.data.cleaning.clean_oced_data`.

        Raises:
            ValueError: If the store was not created with ``ocel_mode=True``.
        """
        self._require_ocel_mode("replace_object_changes")
        self._object_changes = dict(changes)

    def _require_ocel_mode(self, method: str) -> None:
        if not self._ocel_mode:
            raise ValueError(
                f"{method} requires ocel_mode=True. "
                "Create the OCEDStore with OCEDStore(ocel_mode=True)."
            )

    # EVENTS
    def add_event(
        self,
        event_type: str,
        id_col: str,
        time_col: str,
        data: pd.DataFrame,
        overwrite: bool = False,
    ) -> None:
        """
        Add event data, storing both metadata and the event DataFrame.
        The DataFrame must contain the columns specified by `id_col` and `time_col`.

        Parameters:
            event_type: The activity of the event.
            id_col (str): The column name in `data` that contains the unique event identifiers.
            time_col (str): The column name in `data` that contains the event timestamps.
            data (pd.DataFrame): The DataFrame containing event data.
            overwrite (bool, optional): If True, overwrites existing event data. Defaults to False.

        Raises:
            ValueError: If the required columns are missing or if events already exist and overwrite is False.
        """
        if not {id_col, time_col}.issubset(data.columns):
            raise ValueError(
                "The given event column headers are not all included in the data."
            )
        if not overwrite and event_type in self.get_activities():
            raise ValueError(
                "Events data already exists! Set overwrite to True if you want to disregard this."
            )
        if not overwrite and not self.ids_not_used(data[id_col].unique()):
            raise ValueError("IDs in this data are already used.")

        data = infer_dtypes(data, ignore_cols=[id_col], label=event_type)
        self._events[event_type] = EventTable(id=id_col, time=time_col, data=data)
        self.update_used_ids()

    def get_events(self, event_type: str) -> EventTable:
        """Returns the :class:`EventTable` stored for an activity."""
        return self._events[event_type]

    def get_activities(self) -> list[str]:
        """Returns the names of all event types (activities) in the store."""
        return list(self._events)

    def set_event_data(self, event_type: str, data: pd.DataFrame) -> None:
        """Replace the rows of an existing activity, keeping its id and time columns."""
        if event_type not in self._events:
            raise ValueError(f"Unknown activity {event_type}")
        self._events[event_type].data = data

    # OBJECTS
    def add_object(
        self, obj_type: str, id_col: str, data: pd.DataFrame, overwrite: bool = False
    ) -> None:
        """
        Add object data of a specific type. The provided DataFrame must contain
        the unique identifier column specified by `id_col`.

        Parameters:
            obj_type (str): The type/category of the object to be added.
            id_col (str): The column name in `data` that uniquely identifies objects.
            data (pd.DataFrame): The DataFrame containing object data.
            overwrite (bool, optional): If True, overwrites existing object data for this type. Defaults to False.

        Raises:
            ValueError: If the identifier column is missing or if the object type already exists and overwrite is False.
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

        data = infer_dtypes(data, ignore_cols=[id_col], label=obj_type)
        self._objects[obj_type] = ObjectTable(id=id_col, data=data)
        self.update_used_ids()

    def get_objects(self) -> dict[str, ObjectTable]:
        """Returns every :class:`ObjectTable`, keyed by object type."""
        return self._objects

    def get_object_types(self) -> list[str]:
        """Returns the names of all object types in the store."""
        return list(self._objects)

    def get_object(self, obj_type: str) -> ObjectTable:
        """Returns the :class:`ObjectTable` stored for an object type."""
        return self._objects[obj_type]

    def set_object_data(self, obj_type: str, data: pd.DataFrame) -> None:
        """Replace the rows of an existing object type, keeping its id column."""
        if obj_type not in self._objects:
            raise ValueError(f"Unknown object {obj_type}")
        self._objects[obj_type].data = data

    def get_most_common_objects(self, n: int = 1) -> list[str]:
        """Returns the ``n`` object types with the most rows, largest first."""
        objects = self.get_objects()
        return sorted(objects, key=lambda k: len(objects[k].data), reverse=True)[:n]

    # RELATIONS
    def add_event_to_object_relation(
        self,
        relation: str,
        from_id_col: str,
        to_id_col: str,
        to_type: list[str],
        data: pd.DataFrame,
        overwrite: bool = False,
    ) -> None:
        """
        Add a new event-to-object relation.

        Parameters:
            relation (str): The name/identifier of the relation.
            from_id_col (str): The column name representing the event identifier in the relation DataFrame.
            to_id_col (str): The column name representing the object identifier in the relation DataFrame.
            to_type (list[str]): The object types the target identifier may belong to.
            data (pd.DataFrame): The DataFrame containing relation data.
            overwrite (bool, optional): If True, overwrites an existing relation with the same name. Defaults to False.

        Raises:
            ValueError: If the relation already exists and overwrite is False.
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
        """
        Add a new object-to-object relation.

        Parameters:
            relation (str): The name/identifier of the relation.
            from_id_col (str): The column name for the source object's identifier.
            from_type (list[str]): The object types the source identifier may belong to.
            to_id_col (str): The column name for the target object's identifier.
            to_type (list[str]): The object types the target identifier may belong to.
            data (pd.DataFrame): The DataFrame containing the relation mapping details.
            overwrite (bool, optional): If True, overwrites an existing relation with the same name. Defaults to False.

        Raises:
            ValueError: If the relation already exists and overwrite is False.
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

        self._relations.object_to_object[relation] = RelationTable(
            from_id=from_id_col,
            from_type=list(from_type),
            to_id=to_id_col,
            to_type=list(to_type),
            data=data,
        )

    def get_relations(
        self, relation_type: RelationKind | str = RelationKind.ALL
    ) -> dict[str, dict[str, RelationTable]] | dict[str, RelationTable]:
        """
        Returns the relations in the data.

        For ``RelationKind.ALL`` the result is keyed by family
        (``"event_to_object"`` / ``"object_to_object"``); otherwise it maps
        relation name to :class:`RelationTable` directly.

        Raises:
            ValueError: If ``relation_type`` is not a valid :class:`RelationKind`.
        """
        kind = _as_relation_kind(relation_type)
        if kind is RelationKind.ALL:
            return {
                "event_to_object": self._relations.event_to_object,
                "object_to_object": self._relations.object_to_object,
            }
        if kind is RelationKind.E2O:
            return self._relations.event_to_object
        return self._relations.object_to_object

    def get_relation(self, relation: str) -> RelationTable:
        """
        Returns a single :class:`RelationTable` from either family.

        Raises:
            ValueError: If no relation with that name exists.
        """
        if relation in self._relations.event_to_object:
            return self._relations.event_to_object[relation]
        if relation in self._relations.object_to_object:
            return self._relations.object_to_object[relation]
        raise ValueError(f"Relation {relation} does not exist!")

    def get_relation_types(
        self, relation_type: RelationKind | str = RelationKind.ALL
    ) -> list[str]:
        """
        Returns the names of the relations in the data.

        Raises:
            ValueError: If ``relation_type`` is not a valid :class:`RelationKind`.
        """
        kind = _as_relation_kind(relation_type)
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
        """
        Swap in a fully rebuilt set of events, objects and relations.

        Intended for pipeline stages that rewrite every table at once, such as
        :func:`~semanticlog.data.cleaning.clean_oced_data`.
        """
        self._events = events
        self._objects = objects
        self._relations = relations

    def rename_relation(self, old_name: str, new_name: str) -> None:
        """
        Renames an relation that exists in the data.
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
        """
        Renames an object that exists in the data, while ensuring that its relations stay consistent.
        """
        if old_name not in self._objects:
            raise ValueError("Cannot rename object that does not exist!")

        self._objects[new_name] = self._objects.pop(old_name)

        def renamed(types: list[str] | None) -> list[str] | None:
            if types is None:
                return None
            return [new_name if t == old_name else t for t in types]

        for relation in self._relations.event_to_object.values():
            relation.to_type = renamed(relation.to_type)  # type: ignore[assignment]
        for relation in self._relations.object_to_object.values():
            relation.from_type = renamed(relation.from_type)
            relation.to_type = renamed(relation.to_type)  # type: ignore[assignment]

        if self._ocel_mode and old_name in self._object_changes:
            self._object_changes[new_name] = self._object_changes.pop(old_name)

    def rename_activity(self, old_name: str, new_name: str) -> None:
        """
        Renames an activity that exists in the data, while ensuring that its relations stay consistent.
        """
        if old_name not in self._events:
            raise ValueError("Cannot rename activity that does not exist!")
        self._events[new_name] = self._events.pop(old_name)

    def change_ids(
        self,
        of: EntityKind | str,
        type_to_change: str,
        change_function: Callable[[pd.Series], pd.Series],
    ) -> None:
        """
        Rewrite the identifiers of one event or object type, keeping relations consistent.

        Raises:
            ValueError: If ``of`` is not a valid :class:`EntityKind`.
        """
        if _as_entity_kind(of) is EntityKind.EVENT:
            self.change_event_ids(type_to_change, change_function)
        else:
            self.change_object_ids(type_to_change, change_function)

    def change_object_ids(
        self, object_type: str, change_function: Callable[[pd.Series], pd.Series]
    ) -> None:
        """Rewrite one object type's identifiers, updating every relation that uses them."""
        # snapshot for rollback
        objects_backup = copy.deepcopy(self._objects[object_type])
        relations_backup = copy.deepcopy(self._relations)
        try:
            table = self._objects[object_type]
            unique_ids = table.data[table.id].unique()

            # change IDs in the object table
            table.data[table.id] = change_function(table.data[table.id])

            # change IDs in all o2o relations where this object type appears
            for relation in self._relations.object_to_object.values():
                if object_type in (relation.from_type or []):
                    _remap_column(
                        relation.data, relation.from_id, unique_ids, change_function
                    )
                if object_type in relation.to_type:
                    _remap_column(
                        relation.data, relation.to_id, unique_ids, change_function
                    )

            # change IDs in all e2o relations where this object type appears
            for relation in self._relations.event_to_object.values():
                if object_type in relation.to_type:
                    _remap_column(
                        relation.data, relation.to_id, unique_ids, change_function
                    )

            self.update_used_ids()
        except Exception:
            self._objects[object_type] = objects_backup
            self._relations = relations_backup
            raise

    def change_event_ids(
        self, activity: str, change_function: Callable[[pd.Series], pd.Series]
    ) -> None:
        """Rewrite one activity's event identifiers, updating every e2o relation that uses them."""
        # snapshot for rollback
        events_backup = copy.deepcopy(self._events[activity])
        e2o_backup = copy.deepcopy(self._relations.event_to_object)
        try:
            table = self._events[activity]
            unique_ids = table.data[table.id].unique()

            # change IDs in the event table
            table.data[table.id] = change_function(table.data[table.id])

            # change IDs in all e2o relations where this id_col appears
            for relation in self._relations.event_to_object.values():
                mask = relation.data[relation.from_id].isin(unique_ids)
                if mask.any():
                    relation.data.loc[mask, relation.from_id] = change_function(
                        relation.data.loc[mask, relation.from_id]
                    )

            self.update_used_ids()
        except Exception:
            self._events[activity] = events_backup
            self._relations.event_to_object = e2o_backup
            raise

    def get_ids(self, of: EntityKind | str, thing_type: str) -> Iterable:
        """
        Return the unique identifiers of one event or object type.

        Raises:
            ValueError: If ``of`` is invalid or ``thing_type`` is unknown.
        """
        if _as_entity_kind(of) is EntityKind.EVENT:
            if thing_type not in self.get_activities():
                raise ValueError(f"Unknown activity {thing_type}")
            table: EventTable | ObjectTable = self._events[thing_type]
        else:
            if thing_type not in self.get_object_types():
                raise ValueError(f"Unknown object {thing_type}")
            table = self._objects[thing_type]
        return table.data[table.id].unique()

    def update_used_ids(self) -> None:
        """
        Recompute the set of identifiers in use across all events and objects.

        Raises:
            ValueError: If the same identifier appears in more than one table.
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
        """Returns True if none of the given identifiers are already in the store."""
        return self._used_ids.isdisjoint(ids)


def _as_relation_kind(value: RelationKind | str) -> RelationKind:
    try:
        return RelationKind(value)
    except ValueError:
        valid = ", ".join(k.value for k in RelationKind)
        raise ValueError(
            f"Valid relation types are {valid}, but {value} was given."
        ) from None


def _as_entity_kind(value: EntityKind | str) -> EntityKind:
    try:
        return EntityKind(value)
    except ValueError:
        valid = ", ".join(k.value for k in EntityKind)
        raise ValueError(
            f"Valid entity kinds are {valid}, but {value} was given."
        ) from None


def _remap_column(
    df: pd.DataFrame,
    column: str,
    ids: Iterable,
    change_function: Callable[[pd.Series], pd.Series],
) -> None:
    """Apply change_function in place to the rows of ``column`` whose value is in ``ids``."""
    mask = df[column].isin(ids)
    if mask.any():
        df[column] = df[column].astype(object)
        df.loc[mask, column] = change_function(df.loc[mask, column])
