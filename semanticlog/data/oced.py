import pandas as pd
import pm4py as pm


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

    def __init__(self):
        """
        Constructor to initialize empty structures for events, objects, and relations.
        """
        self._events = {}
        self._objects = {}
        self._relations = {"event_to_object": {}, "object_to_object": {}}

    #### MISCELLANEOUS ####

    def get_types(self, ids: list[str] | pd.Series) -> list[str]:
        """
        Finds that object or event type that includes any of the given ids.
        """
        if type(ids) == list:
            ids = pd.Series(ids)
        types = []
        for object_type in self._objects.keys():
            object_df = self._objects[object_type]["data"]
            object_id = self._objects[object_type]["id"]
            if ids.isin(object_df[object_id]).any():
                types.append(object_type)
        for activity in self._events.keys():
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
            raise ValueError("Unknown object type: ", case_concept)

        # get all relevant e2o relations
        data_list = []
        for relation in self._relations["event_to_object"].keys():
            relation_dict = self._relations["event_to_object"][relation]
            to_id = relation_dict["to_id"]
            if case_concept in relation_dict["to_type"]:
                data_list.append(relation_dict["data"].rename(columns={to_id: case_concept}))

        rel_data = pd.concat(data_list, ignore_index=True)
        rel_data = rel_data.rename(columns={"from_id": "event_id"})

        events_data_list = []
        for event_type in self._events.keys():
            id_col = self._events[event_type]["id"]
            time_col = self._events[event_type]["time"]
            event_data = self._events[event_type]["data"][[id_col, time_col]]
            event_data["concept:name"] = event_type
            event_data = event_data.rename(
                columns={id_col: "event_id", time_col: "time:timestamp"}
            )
            events_data_list.append(event_data)

        events_data = pd.concat(events_data_list)
        log = pd.merge(events_data, rel_data, on="event_id", how="inner")
        log = log.set_index("event_id")

        log = pm.format_dataframe(
            log,
            case_id=case_concept,
            activity_key="concept:name",
            timestamp_key="time:timestamp",
        )
        
        return log[["case:concept:name", "concept:name", "time:timestamp"]].sort_values(
            by=["case:concept:name", "time:timestamp"], ascending=True
        )

    #### EVENTS ####
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
        if not set([id_col, time_col]).issubset(set(data.columns)):
            raise ValueError(
                "The given event column headers are not all included in the data."
            )
        if not overwrite and event_type in self.get_activities():
            raise ValueError(
                "Events data already exists! Set overwrite to True if you want to disregard this."
            )

        self._events[event_type] = {"id": id_col, "time": time_col, "data": data}

    def get_events(self, event_type) -> dict[str, str | pd.DataFrame]:
        return self._events[event_type]

    def get_activities(self) -> list[str]:
        return self._events.keys()

    def get_event(self, event_type) -> dict[str, str | pd.DataFrame]:
        return self._events[event_type]

    #### OBJECTS ####
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
        if not overwrite and obj_type in self._objects.keys():
            raise ValueError(
                f"Object ({obj_type}) already exists! Set overwrite to True if you want to disregard this."
            )

        obj = {"id": id_col, "data": data}

        self._objects[obj_type] = obj

    def get_objects(self) -> dict[str, [dict[str, str | pd.DataFrame]]]:
        return self._objects

    def get_object_types(self) -> list[str]:
        return list(self._objects.keys())

    def get_object(self, obj_type) -> dict[str, str | pd.DataFrame]:
        return self._objects[obj_type]

    #### RELATIONS ####
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
        if relation in self._relations["object_to_object"].keys():
            raise ValueError(
                f"The relation ({relation}) already exists as a object_to_object relation!"
            )
        if not overwrite and relation in self._relations["event_to_object"].keys():
            raise ValueError(
                f"Relation ({relation}) already exists! Set overwrite to True if you want to disregard this."
            )
        if not set(to_type).issubset(set(self.get_object_types())):
            raise ValueError(
                f"Can not instantiate relation ({relation}). Unknown object type in ({to_type})!"
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
        if relation in self._relations["event_to_object"].keys():
            raise ValueError(
                f"The relation ({relation}) already exists as a event_to_object relation!"
            )
        if not overwrite and relation in self._relations["object_to_object"].keys():
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
        elif relation_type == "o2o":
            return self._relations["object_to_object"]

    def get_relation(self, relation: str) -> dict[str, str | pd.DataFrame]:
        """
        Returns a single relation from the data.
        """
        if relation in self._relations["event_to_object"].keys():
            return self._relations["event_to_object"][relation]
        elif relation in self._relations["object_to_object"].keys():
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
        relations = []
        if relation_type == "e2o" or relation_type == "all":
            relations = relations + list(self._relations["event_to_object"].keys())
        if relation_type == "o2o" or relation_type == "all":
            relations = relations + list(self._relations["object_to_object"].keys())
        return relations

    def rename_relation(self, old_name: str, new_name: str) -> None:
        """
        Renames an relation that exists in the data.
        """
        if old_name in self._relations["event_to_object"].keys():
            self._relations["event_to_object"][new_name] = self._relations[
                "event_to_object"
            ].pop(old_name)
        elif old_name in self._relations["object_to_object"].keys():
            self._relations["object_to_object"][new_name] = self._relations[
                "object_to_object"
            ].pop(old_name)
        else:
            raise ValueError("Cant rename relation that does not exist!")

    def rename_object(self, old_name: str, new_name: str) -> None:
        """
        Renames an object that exists in the data, while ensuring that its relations stay consistent.
        """
        if old_name in self._objects.keys():
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
        else:
            raise ValueError("Cant rename object that does not exist!")

    def rename_activity(self, old_name: str, new_name: str) -> None:
        """
        Renames an activity that exists in the data, while ensuring that its relations stay consistent.
        """
        if old_name in self._events.keys():
            self._events[new_name] = self._events.pop(old_name)
        else:
            raise ValueError("Cant rename activity that does not exist!")
