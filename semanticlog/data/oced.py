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
    
    def get_type(self, ids: list[str]) -> str:
        """
        Finds that object or event type that includes any of the given ids.
        """

        for object_type in self._objects.keys():
            if self._objects[object_type]['data'][self._objects[object_type]["id"]].isin(ids).any():
                return object_type
        for event_type in self._events.keys():
            if self._events[event_type]['data'][self._events[event_type]["id"]].isin(ids).any():
                return event_type
        return None

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
            if relation_dict["to_type"] == case_concept:
                data_list.append(relation_dict["data"])

        rel_data = pd.concat(data_list, ignore_index=True)
        rel_data = rel_data.rename(columns={"from_id": "event_id"})
        
        events_data_list = []
        for event_type in self._events.keys():
            id_col = self._events[event_type]["id"]
            time_col = self._events[event_type]["time"]
            event_data = self._events[event_type]["data"][[id_col, time_col]]
            event_data["concept:name"] = event_type
            event_data = event_data.rename(columns={id_col: "event_id", time_col: "time:timestamp"})
            events_data_list.append(event_data)
            
        events_data = pd.concat(events_data_list)
        log = pd.merge(events_data, rel_data, on="event_id", how="inner")
        log = log.set_index("event_id")

        log = pm.format_dataframe(
            log,
            case_id="to_id",
            activity_key="concept:name",
            timestamp_key="time:timestamp",
        )

        return log[["case:concept:name", "concept:name", "time:timestamp"]].sort_values(by=['case:concept:name', 'time:timestamp'], ascending=True)

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
        return self._objects.keys()

    def get_object(self, obj_type) -> dict[str, str | pd.DataFrame]:
        return self._objects[obj_type]

    #### RELATIONS ####
    def add_event_to_object_relation(
        self,
        relation: str,
        from_id_col: str,
        to_id_col: str,
        to_type: str,
        data: pd.DataFrame,
        overwrite: bool = False,
    ) -> None:
        """
        Add a new event-to-object relation.

        Parameters:
            relation (str): The name/identifier of the relation.
            from_id_col (str): The column name representing the event identifier in the relation DataFrame.
            to_id_col (str): The column name representing the object identifier in the relation DataFrame.
            to_type (str): The column name representing the object type in the relation DataFrame.
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
        if to_type not in self.get_object_types():
            raise ValueError(
                f"Can not instantiate relation ({relation}). Unknown object type ({to_type})!"
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
        from_type: str,
        to_id_col: str,
        to_type: str,
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
        if (
            from_type not in self.get_object_types()
            or to_type not in self.get_object_types()
        ):
            raise ValueError(
                f"Can not instantiate relation ({relation}). Unknown object type ({from_type} or {to_type})!"
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