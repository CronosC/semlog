from semanticlog.data.abstractloader import AbstractLoader
from semanticlog.data.oced import OCEDStore
import sqlite3
import pandas as pd
import os


class OCELLoader(AbstractLoader):
    """ AbstractLoader implementation which transforms OCEL logs into OCEDStore objects."""

    def __init__(self, path: str, config: dict[str, str]):
        """Initialize the OCEL data loader with a configuration dict and path."""
        super().__init__(path, config)
        if "file_format" not in config.keys():
            raise Exception(
                "To load a OCEL file you need to specify its format (sqlite, json, xml)!"
            )

    def get_oced(self) -> OCEDStore:
        """Returns the data as a OCEDStore object"""
        if self.config["file_format"] == "xml":
            raise NotImplementedError(
                "The xml file format for OCEL is not implemented."
            )
        elif self.config["file_format"] == "json":
            raise NotImplementedError(
                "The json file format for OCEL is not implemented."
            )
        elif self.config["file_format"] == "sqlite":
            return self.get_data_from_sqlite_file(self.path)
        
    def get_data_from_sqlite_file(self, path: str) -> OCEDStore:
        print("\n",path)
        print("Current working directory:", os.getcwd())
        #db = sqlite3.connect(path)

        #df_event = pd.read_sql_query("SELECT * FROM event", db)
        #df_object = pd.read_sql_query("SELECT * FROM object", db)
        #df_e2o = pd.read_sql_query("SELECT * FROM event_object", db)
        #df_o2o = pd.read_sql_query("SELECT * FROM object_object", db)
        #df_order = pd.read_sql_query("SELECT * FROM object_Order", db)
        
        oced = OCEDStore()
        
        with sqlite3.connect(path) as dbcon:
            tables = list(pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", dbcon)['name'])
            dfs = {tbl : pd.read_sql_query(f"SELECT * from {tbl}", dbcon) for tbl in tables}

        # Objects #
        """
            NOTE:
            We map the OCEL meta-model to the generic OCED model. In the generic model
            timestamps can not be associated with object attribute values. For this reason we
            filter the OCEL data s.t. we get only the latest attribute values. This is a major
            loss of information and probably introduces artifacts.
        """
        objects = dfs["object_map_type"]["ocel_type_map"].unique()
        for object_type in objects:
            object_df = dfs["object_" + object_type]
            last_change_index = object_df.groupby('ocel_id')['ocel_time'].idxmax()
            latest_object_df = object_df.loc[last_change_index]
            latest_object_df = latest_object_df.drop(["ocel_changed_field", "ocel_time"], axis=1, errors='ignore')
            oced.add_object(object_type, "ocel_id", latest_object_df)

        # Events #
        events = dfs["event_map_type"]["ocel_type_map"].unique()
        for event_type in events:
            event_df = dfs["event_" + event_type]
            oced.add_event(event_type, "ocel_id", "ocel_time", event_df)
        
        """
            NOTE:
            Lots of ocel datasets contain None relations. We replace this with a generic "modifies" relation
            and hope for no collisions or other problems. 
        """
        dfs["event_object"]["ocel_qualifier"] = dfs["event_object"]["ocel_qualifier"].fillna("modifies")
        dfs["object_object"]["ocel_qualifier"] = dfs["object_object"]["ocel_qualifier"].fillna("modifies")
        
        # E2O relations #
        e2o_relations = dfs["event_object"]["ocel_qualifier"].unique()
        for relation_type in e2o_relations:
            relation_data =  dfs["event_object"][dfs["event_object"]["ocel_qualifier"] == relation_type]
            relation_data = relation_data.drop(["ocel_qualifier"], axis=1, errors='ignore')
            if relation_type == None or relation_type == "None":
                relation_type = "modifies"
            to_type = oced.get_type(relation_data["ocel_object_id"])
            oced.add_event_to_object_relation(relation_type, "ocel_event_id", "ocel_object_id", to_type, relation_data)
           
        # O2O relations # 
        o2o_relations = dfs["object_object"]["ocel_qualifier"].unique()
        for relation_type in o2o_relations:
            relation_data =  dfs["object_object"][dfs["object_object"]["ocel_qualifier"] == relation_type]
            relation_data = relation_data.drop(["ocel_qualifier"], axis=1, errors='ignore')
            if relation_type == None or relation_type == "None":
                relation_type = "modifies"
            from_type = oced.get_type(relation_data["ocel_source_id"])
            to_type = oced.get_type(relation_data["ocel_target_id"])
            oced.add_object_to_object_relation(relation_type, "ocel_source_id", from_type, "ocel_target_id", to_type, relation_data)
        
        return oced
        

