from semanticlog.data.abstractloader import AbstractLoader
from semanticlog.data.oced import OCEDStore
import sqlite3
import pandas as pd
import os
import logging
import pm4py as pm
import numpy as np


class PM4PYLoader(AbstractLoader):
    """AbstractLoader implementation which uses PM4PY to load OCEL logs and store them in OCEDStore objects."""

    def __init__(self, path: str, config: dict[str, str]):
        """Initialize the OCEL data loader with a configuration dict and path."""
        super().__init__(path, config)
        if "file_format" not in config.keys():
            if path.endswith(".xml") or path.endswith(".xmlocel"):
                config["file_format"] = "xml"
            elif path.endswith(".json") or path.endswith(".jsonocel"):
                config["file_format"] = "json"
            elif path.endswith(".sqlite"):
                config["file_format"] = "sqlite"
            else:
                raise Exception("Unknown file type.")

    def get_oced(self) -> OCEDStore:
        """Returns the data as a OCEDStore object"""
        logging.info(f"Loading ocel data from file {self.path}..")
        if self.config["file_format"] == "xml":
            log = pm.read_ocel2_xml(self.path)
        elif self.config["file_format"] == "json":
            log = pm.read_ocel2_json(self.path)
        elif self.config["file_format"] == "sqlite":
            log = pm.read_ocel2_sqlite(self.path)

        oced = OCEDStore()

        # Events
        event_dfs = {
            activity: df for activity, df in log.events.groupby("ocel:activity")
        }
        for activity in event_dfs.keys():
            df = event_dfs[activity]
            df = df.drop(["ocel:activity"], axis=1)
            oced.add_event(activity, "ocel:eid", "ocel:timestamp", df)

        # Objects
        object_dfs = {
            object_type: df for object_type, df in log.objects.groupby("ocel:type")
        }
        for object_type in object_dfs.keys():
            object_dfs[object_type] = object_dfs[object_type].dropna(axis=1, how="all")
            df = object_dfs[object_type]
            df = df.drop(["ocel:type"], axis=1)
            oced.add_object(object_type, "ocel:oid", df)

        # E2O Relations
        relations = log.relations
        """ NOTE:
        We assume all relations without qualifier are a generic "modifies" relation.
        """
        relations["ocel:qualifier"] = relations["ocel:qualifier"].replace(
            [None, np.nan, "None", "none", "NA", "N/A", ""], "modifies"
        )
        e2o_relation_dfs = {
            relation_type: df
            for relation_type, df in relations.groupby("ocel:qualifier")
        }
        for relation_type in e2o_relation_dfs.keys():
            df = e2o_relation_dfs[relation_type]
            object_types = e2o_relation_dfs[relation_type]["ocel:type"].unique()
            oced.add_event_to_object_relation(
                relation_type,
                "ocel:eid",
                "ocel:oid",
                object_types,
                df[["ocel:eid", "ocel:oid"]],
            )

        # O2O Relations
        assert not log.o2o.isnull().values.any()
        o2o_relation_dfs = {
            relation_type: df for relation_type, df in log.o2o.groupby("ocel:qualifier")
        }
        for relation_type in o2o_relation_dfs.keys():
            df = o2o_relation_dfs[relation_type]
            
            if relation_type in o2o_relation_dfs.keys():
                relation_type = relation_type + "_o2o"

            type_1 = oced.get_types(df["ocel:oid"])
            type_2 = oced.get_types(df["ocel:oid_2"])

            assert type_1 != [] and type_2 != []
            
            oced.add_object_to_object_relation(
                relation_type, "ocel:oid", type_1, "ocel:oid_2", type_2, df
            )

        return oced
