from semanticlog.data.abstractloader import AbstractLoader
from semanticlog.data.oced import OCEDStore
import sqlite3
import pandas as pd
import os
import logging
import pm4py as pm
from pm4py.objects.conversion.log import converter as log_converter



class ClassicalLoader(AbstractLoader):
    """Loads classical event logs in form of .csv or .xes files and transforms them into a OCEDStore"""

    def __init__(self, path: str, config: dict[str, str]):
        """Initialize the OCEL data loader with a configuration dict and path."""
        super().__init__(path, config)
        if not (self.path.endswith(".xes") or self.path.endswith(".csv")):
            raise Exception("Can only load .xes or .csv classical event logs.")

        config.setdefault("case_col", "case:concept:name")
        config.setdefault("act_col", "concept:name")
        config.setdefault("time_col", "time:timestamp")
        
        self.case_col = config["case_col"]
        self.act_col = config["act_col"]
        self.time_col = config["time_col"]

    def get_oced(self) -> OCEDStore:
        """Returns the data as a OCEDStore object"""
        logging.info(f"Loading classical data from file {self.path}..")
        if self.path.endswith(".xes"):
            log = pm.read_xes(self.path)
        elif self.path.endswith(".csv"):
            log = pd.read_csv(self.path)
            #log = log.rename({
            #    self.case_col: "case:concept:name",
            #    self.act_col: "concept:name",
            #    self.time_col: "time:timestamp",
            #                })
            #log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG)
        else:
            raise Exception("Can only load .xes or .csv classical event logs.")
        log["event_id"] = log.index

        oced = OCEDStore()
        
        # Events
        event_dfs = {
            activity: df for activity, df in log.drop(columns=[self.case_col]).groupby(self.act_col)
        }

        for activity in event_dfs.keys():
            df = event_dfs[activity]
            oced.add_event(activity, "event_id", self.time_col, df)
            
        # Objects
        object_df = log[[self.case_col]]
        oced.add_object(self.case_col, self.case_col, object_df)
        
        # E2O
        rel_df = log[["event_id", self.case_col]]
        oced.add_event_to_object_relation(
                "modifies",
                "event_id",
                self.case_col,
                [self.case_col],
                rel_df,
            )
        
        return oced
