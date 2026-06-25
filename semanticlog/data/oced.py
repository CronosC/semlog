import copy
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pm4py as pm

_DEFAULT_IRI: str = "https://oced#"


def _try_datetime(
    series: pd.Series, non_null: int, min_ratio: float
) -> pd.Series | None:
    dt = pd.to_datetime(series, errors="coerce", utc=True, format="ISO8601")
    dt = dt.dt.floor("s")
    return dt if (dt.notna().sum() / non_null) >= min_ratio else None


def _try_numeric(
    series: pd.Series, non_null: int, min_ratio: float
) -> pd.Series | None:
    num = pd.to_numeric(series, errors="coerce")
    if (num.notna().sum() / non_null) < min_ratio:
        return None
    vals = set(num.dropna().unique())
    if vals <= {0, 1}:
        return num.astype("boolean")
    if (num.dropna() % 1 == 0).all():
        return num.astype("Int64")
    return num.astype("float64")


class OCEDStore:
    """
    Holds object-centric event data.

    The class maintains:
      - Events: Stored with the event id, type, time, and associated DataFrame.
      - Objects: Stored as a dictionary mapping object types to their identifier and data.
      - Relations: Specifies a type of relation. Contains event/object ids and object types.
          * Event-to-object relations
          * Object-to-object relations
    """

    def __init__(self, ocel_mode: bool = False):
        """
        Constructor to initialize empty structures for events, objects, and relations.

        Args:
            ocel_mode: When True, enables storage of OCEL 2.0 extended data such as
                object attribute change history. Required before calling
                ``add_object_changes`` or ``get_object_changes``.
        """
        self._events: dict[str, Any] = {}
        self._objects: dict[str, Any] = {}
        self._relations: dict[str, Any] = {
            "event_to_object": {},
            "object_to_object": {},
        }
        self._used_ids: set = set()
        self._ocel_mode = ocel_mode
        self._backmap: dict[str, str] = {}
        if ocel_mode:
            self._object_changes: dict[str, pd.DataFrame] = {}

    # BACKMAP

    @property
    def backmap(self) -> dict[str, str]:
        """Mapping from post-clean IRI local name to original pre-clean name (IDs and schema names)."""
        return self._backmap

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
    def load(cls, path: str | Path) -> "OCEDStore":
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
        for object_type in self._objects:
            object_df = self._objects[object_type]["data"]
            object_id = self._objects[object_type]["id"]
            if ids.isin(object_df[object_id]).any():
                types.append(object_type)
        for activity in self._events:
            event_df = self._events[activity]["data"]
            event_id = self._events[activity]["id"]
            if ids.isin(event_df[event_id]).any():
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
        for relation in self._relations["event_to_object"]:
            relation_dict = self._relations["event_to_object"][relation]
            to_id = relation_dict["to_id"]
            from_id = relation_dict["from_id"]
            if case_concept in relation_dict["to_type"]:
                data_list.append(
                    relation_dict["data"].rename(
                        columns={to_id: case_concept, from_id: "event_id"}
                    )
                )

        if not data_list:
            raise ValueError(
                f"No event-to-object relations reference object type '{case_concept}'."
            )
        rel_data = pd.concat(data_list, ignore_index=True)
        valid_case_ids = set(self.get_ids("object", case_concept))
        dropped_ids = set(rel_data[case_concept]) - valid_case_ids
        rel_data = rel_data[rel_data[case_concept].isin(valid_case_ids)]

        if dropped_ids:
            all_object_ids: set = set()
            for entry in self._objects.values():
                all_object_ids.update(entry["data"][entry["id"]].unique())
            genuinely_missing = dropped_ids - all_object_ids
            if genuinely_missing:
                logging.warning(
                    f"get_log: {len(genuinely_missing)} object IDs not found in any object store "
                    f"for case concept '{case_concept}' "
                    f"(sample: {sorted(str(x) for x in genuinely_missing)[:5]})"
                )

        events_data_list = []
        for event_type in self._events:
            id_col = self._events[event_type]["id"]
            time_col = self._events[event_type]["time"]
            event_data = self._events[event_type]["data"][[id_col, time_col]].copy()
            event_data["concept:name"] = event_type
            event_data = event_data.rename(
                columns={id_col: "event_id", time_col: "time:timestamp"}
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
        if not self._ocel_mode:
            raise ValueError(
                "add_object_changes requires ocel_mode=True. "
                "Create the OCEDStore with OCEDStore(ocel_mode=True)."
            )
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
        if not self._ocel_mode:
            raise ValueError(
                "get_object_changes requires ocel_mode=True. "
                "Create the OCEDStore with OCEDStore(ocel_mode=True)."
            )
        if obj_type not in self._object_changes:
            raise ValueError(f"No object changes recorded for type '{obj_type}'.")
        return self._object_changes[obj_type]

    def get_object_change_types(self) -> list[str]:
        """
        Return the list of object types that have recorded attribute change history.

        Raises:
            ValueError: If the store was not created with ``ocel_mode=True``.
        """
        if not self._ocel_mode:
            raise ValueError(
                "get_object_change_types requires ocel_mode=True. "
                "Create the OCEDStore with OCEDStore(ocel_mode=True)."
            )
        return list(self._object_changes)

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
        The DataFrame must contain the columns specified by `id_col`, `type_col`, and `time_col`.

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

        data = self.data_check(data, ignore_cols=[id_col])
        self._events[event_type] = {"id": id_col, "time": time_col, "data": data}
        self.update_used_ids()

    def get_events(self, event_type) -> dict[str, str | pd.DataFrame]:
        """Alias for get_event — returns the event dict for event_type."""
        return self._events[event_type]

    def get_activities(self) -> list[str]:
        return list(self._events)

    def get_event(self, event_type) -> dict[str, str | pd.DataFrame]:
        """Returns the event dict for event_type. Equivalent to get_events()."""
        return self.get_events(event_type)

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

        data = self.data_check(data, ignore_cols=[id_col])
        obj = {"id": id_col, "data": data}
        self._objects[obj_type] = obj
        self.update_used_ids()

    def get_objects(self) -> dict[str, dict[str, str | pd.DataFrame]]:
        return self._objects

    def get_object_types(self) -> list[str]:
        return list(self._objects)

    def get_object(self, obj_type) -> dict[str, str | pd.DataFrame]:
        return self._objects[obj_type]

    def get_most_common_objects(self, n: int = 1) -> list[str]:
        objects = self.get_objects()
        return sorted(
            objects,
            key=lambda k: len(objects[k]["data"]),
            reverse=True,
        )[:n]

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
            to_type (list[str]): The column names representing the object type in the relation DataFrame.
            data (pd.DataFrame): The DataFrame containing relation data.
            overwrite (bool, optional): If True, overwrites an existing relation with the same name. Defaults to False.

        Raises:
            ValueError: If the relation already exists and overwrite is False.
        """
        if relation in self._relations["object_to_object"]:
            raise ValueError(
                f"The relation ({relation}) already exists as a object_to_object relation!"
            )
        if not overwrite and relation in self._relations["event_to_object"]:
            raise ValueError(
                f"Relation ({relation}) already exists! Set overwrite to True if you want to disregard this."
            )
        if not set(to_type).issubset(set(self.get_object_types())):
            raise ValueError(
                f"Can not instantiate relation ({relation}). Unknown object type in ({to_type})!"
            )
        if from_id_col not in data.columns:
            raise ValueError(
                f"from_id_col '{from_id_col}' not found in relation data columns: {list(data.columns)}"
            )
        if to_id_col not in data.columns:
            raise ValueError(
                f"to_id_col '{to_id_col}' not found in relation data columns: {list(data.columns)}"
            )

        rel_value = {
            "from_id": from_id_col,
            "to_id": to_id_col,
            "to_type": to_type,
            "data": data,
        }

        self._relations["event_to_object"][relation] = rel_value

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
            from_type (str): The source object's type.
            to_id_col (str): The column name for the target object's identifier.
            to_type (str): The target object's type.
            data (pd.DataFrame): The DataFrame containing the relation mapping details.
            overwrite (bool, optional): If True, overwrites an existing relation with the same name. Defaults to False.

        Raises:
            ValueError: If the relation already exists and overwrite is False.
        """
        if relation in self._relations["event_to_object"]:
            raise ValueError(
                f"The relation ({relation}) already exists as a event_to_object relation!"
            )
        if not overwrite and relation in self._relations["object_to_object"]:
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

        rel_value = {
            "from_id": from_id_col,
            "from_type": from_type,
            "to_id": to_id_col,
            "to_type": to_type,
            "data": data,
        }

        self._relations["object_to_object"][relation] = rel_value

    def get_relations(
        self, relation_type="all"
    ) -> (
        dict[str, dict[str, dict[str, str | pd.DataFrame]]]
        | dict[str, dict[str, str | pd.DataFrame]]
    ):
        """
        Returns a dict of relations in the data. Valid relation types are 'all', 'e2o' or 'o2o'.
        """
        if relation_type not in ["all", "e2o", "o2o"]:
            raise ValueError(
                f"Valid relation types are all, e2o, o2o, but {relation_type} was given."
            )
        if relation_type == "all":
            return self._relations
        elif relation_type == "e2o":
            return self._relations["event_to_object"]
        else:
            return self._relations["object_to_object"]

    def get_relation(self, relation: str) -> dict[str, str | pd.DataFrame]:
        """
        Returns a single relation from the data.
        """
        if relation in self._relations["event_to_object"]:
            return self._relations["event_to_object"][relation]
        elif relation in self._relations["object_to_object"]:
            return self._relations["object_to_object"][relation]
        else:
            raise ValueError(f"Relation {relation} does not exist!")

    def get_relation_types(self, relation_type="all") -> list[str]:
        """
        Returns the types of relations in the data. Valid relation types are 'all', 'e2o' or 'o2o'.
        """
        if relation_type not in ["all", "e2o", "o2o"]:
            raise ValueError(
                f"Valid relation types are all, e2o, o2o, but {relation_type} was given."
            )
        relations: list[str] = []
        if relation_type == "e2o" or relation_type == "all":
            relations = relations + list(self._relations["event_to_object"])
        if relation_type == "o2o" or relation_type == "all":
            relations = relations + list(self._relations["object_to_object"])
        return relations

    def rename_relation(self, old_name: str, new_name: str) -> None:
        """
        Renames an relation that exists in the data.
        """
        all_relation_names = (
            self._relations["event_to_object"].keys()
            | self._relations["object_to_object"].keys()
        )
        if new_name in all_relation_names:
            raise ValueError(
                f"Cannot rename relation: '{new_name}' already exists as a relation."
            )
        if old_name in self._relations["event_to_object"]:
            self._relations["event_to_object"][new_name] = self._relations[
                "event_to_object"
            ].pop(old_name)
        elif old_name in self._relations["object_to_object"]:
            self._relations["object_to_object"][new_name] = self._relations[
                "object_to_object"
            ].pop(old_name)
        else:
            raise ValueError("Cannot rename relation that does not exist!")

    def rename_object(self, old_name: str, new_name: str) -> None:
        """
        Renames an object that exists in the data, while ensuring that its relations stay consistent.
        """
        if old_name in self._objects:
            self._objects[new_name] = self._objects.pop(old_name)
            for relation in self._relations["event_to_object"]:
                self._relations["event_to_object"][relation]["to_type"] = [
                    new_name if x == old_name else x
                    for x in self._relations["event_to_object"][relation]["to_type"]
                ]
            for relation in self._relations["object_to_object"]:
                self._relations["object_to_object"][relation]["from_type"] = [
                    new_name if x == old_name else x
                    for x in self._relations["object_to_object"][relation]["from_type"]
                ]
                self._relations["object_to_object"][relation]["to_type"] = [
                    new_name if x == old_name else x
                    for x in self._relations["object_to_object"][relation]["to_type"]
                ]
            if self._ocel_mode and old_name in self._object_changes:
                self._object_changes[new_name] = self._object_changes.pop(old_name)
        else:
            raise ValueError("Cannot rename object that does not exist!")

    def rename_activity(self, old_name: str, new_name: str) -> None:
        """
        Renames an activity that exists in the data, while ensuring that its relations stay consistent.
        """
        if old_name in self._events:
            self._events[new_name] = self._events.pop(old_name)
        else:
            raise ValueError("Cannot rename activity that does not exist!")

    def change_ids(self, of, type_to_change, change_function):
        if of not in ["event", "object"]:
            raise ValueError("Can only change the ids of events or objects!")
        if of == "event":
            self.change_event_ids(type_to_change, change_function)
        elif of == "object":
            self.change_object_ids(type_to_change, change_function)

    def change_object_ids(self, object_type, change_function):
        # snapshot for rollback
        objects_backup = copy.deepcopy(self._objects[object_type])
        o2o_backup = copy.deepcopy(self._relations["object_to_object"])
        e2o_backup = copy.deepcopy(self._relations["event_to_object"])
        try:
            object_data = self._objects[object_type]["data"]
            id_col = self._objects[object_type]["id"]
            unique_ids = object_data[id_col].unique()

            # change IDs in the object table
            self._objects[object_type]["data"][id_col] = change_function(
                object_data[id_col]
            )

            # change IDs in all o2o relations where this id_col appears
            for o2o_rel, rel in self._relations.get("object_to_object", {}).items():
                df = rel["data"]
                from_id = rel["from_id"]
                from_type = rel["from_type"]
                to_id = rel["to_id"]
                to_type = rel["to_type"]
                if object_type in from_type:
                    mask = df[from_id].isin(unique_ids)
                    if mask.any():
                        df[from_id] = df[from_id].astype(object)
                        df.loc[mask, from_id] = change_function(df.loc[mask, from_id])
                if object_type in to_type:
                    mask = df[to_id].isin(unique_ids)
                    if mask.any():
                        df[to_id] = df[to_id].astype(object)
                        df.loc[mask, to_id] = change_function(df.loc[mask, to_id])
                self._relations["object_to_object"][o2o_rel]["data"] = df

            # change IDs in all e2o relations where this id_col appears
            for e2o_rel, rel in self._relations.get("event_to_object", {}).items():
                df = rel["data"]
                to_id = rel["to_id"]
                to_type = rel["to_type"]
                if object_type in to_type:
                    mask = df[to_id].isin(unique_ids)
                    if mask.any():
                        df[to_id] = df[to_id].astype(object)
                        df.loc[mask, to_id] = change_function(df.loc[mask, to_id])
                self._relations["event_to_object"][e2o_rel]["data"] = df

            self.update_used_ids()
        except Exception:
            self._objects[object_type] = objects_backup
            self._relations["object_to_object"] = o2o_backup
            self._relations["event_to_object"] = e2o_backup
            raise

    def change_event_ids(self, activity, change_function):
        # snapshot for rollback
        events_backup = copy.deepcopy(self._events[activity])
        e2o_backup = copy.deepcopy(self._relations["event_to_object"])
        try:
            event_data = self._events[activity]["data"]
            id_col = self._events[activity]["id"]
            unique_ids = event_data[id_col].unique()

            # change IDs in the event table
            self._events[activity]["data"][id_col] = change_function(event_data[id_col])

            # change IDs in all e2o relations where this id_col appears
            for e2o_rel, rel in self._relations.get("event_to_object", {}).items():
                df = rel["data"]
                from_id = rel["from_id"]
                mask = df[from_id].isin(unique_ids)
                if mask.any():
                    df.loc[mask, from_id] = change_function(df.loc[mask, from_id])
                self._relations["event_to_object"][e2o_rel]["data"] = df

            self.update_used_ids()
        except Exception:
            self._events[activity] = events_backup
            self._relations["event_to_object"] = e2o_backup
            raise

    def get_ids(self, of, thing_type):
        ids = []
        if of not in ["event", "object"]:
            raise ValueError("get_ids only returns ids of objects or events")
        elif of == "event":
            if thing_type not in self.get_activities():
                raise ValueError(f"Unknown activity {thing_type}")
            id_col = self._events[thing_type]["id"]
            ids = self._events[thing_type]["data"][id_col].unique()
        else:
            if thing_type not in self.get_object_types():
                raise ValueError(f"Unknown object {thing_type}")
            id_col = self._objects[thing_type]["id"]
            ids = self._objects[thing_type]["data"][id_col].unique()
        return ids

    def update_used_ids(self):
        ids = []
        for activity in self._events:
            ids.extend(self.get_ids("event", activity))
        for object_type in self._objects:
            ids.extend(self.get_ids("object", object_type))
        unique_ids = set(ids)
        if len(unique_ids) != len(ids):
            raise ValueError("Duplicated ids!")
        self._used_ids = unique_ids

    def data_check(self, df, ignore_cols=None, min_parse_ratio=0.95):
        if ignore_cols is None:
            ignore_cols = []
        df = df.copy()  # ensure we write to the real frame, not a view
        df = df.dropna(how="all").dropna(axis=1, how="all")
        # normalize empties for object-like cols
        obj_cols = df.select_dtypes(include="object").columns
        df[obj_cols] = df[obj_cols].replace(r"^\s*$", np.nan, regex=True)

        for col in df.columns:
            if col in ignore_cols:
                continue

            non_null = df[col].notna().sum()
            if non_null == 0:
                continue

            # already numeric/bool? skip
            if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(
                df[col]
            ):
                continue

            dt = _try_datetime(df[col], non_null, min_parse_ratio)
            if dt is not None:
                df[col] = dt
                continue

            num = _try_numeric(df[col], non_null, min_parse_ratio)
            if num is not None:
                df[col] = num
        return df

    def ids_not_used(self, ids) -> bool:
        return self._used_ids.isdisjoint(ids)
