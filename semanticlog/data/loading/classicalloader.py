import logging

import pandas as pd
import pm4py as pm

from semanticlog.data.loading.abstractloader import AbstractLoader
from semanticlog.data.oced import OCEDStore


class ClassicalLoader(AbstractLoader):
    """Loads classical event logs in form of .csv or .xes files and transforms them into a OCEDStore"""

    def __init__(self, path: str, config: dict[str, str]):
        """Initialize the OCEL data loader with a configuration dict and path."""
        super().__init__(path, config)
        if not (self.path.endswith(".xes") or self.path.endswith(".csv")):
            raise ValueError("Can only load .xes or .csv classical event logs.")

        config.setdefault("case_col", "case:concept:name")
        config.setdefault("act_col", "concept:name")
        config.setdefault("time_col", "time:timestamp")
        config.setdefault("object_cols", [])  # type: ignore[arg-type]

        self.case_col = config["case_col"]
        self.act_col = config["act_col"]
        self.time_col = config["time_col"]
        self.object_cols = config["object_cols"]

    def get_oced(self) -> OCEDStore:
        """Returns the data as a OCEDStore object"""
        logging.info(f"Loading classical data from file {self.path}..")
        if self.path.endswith(".xes"):
            log = pm.read_xes(self.path)
        elif self.path.endswith(".csv"):
            log = pd.read_csv(self.path)
        else:
            raise ValueError("Can only load .xes or .csv classical event logs.")
        log["event_id"] = "e" + log.index.astype(str)
        log["case:concept:name"] = "t" + log["case:concept:name"].astype(str)

        oced = OCEDStore()

        # Events
        event_dfs = {
            activity: df
            for activity, df in log.drop(columns=[self.case_col]).groupby(self.act_col)
        }

        for activity in event_dfs:
            df = event_dfs[activity]
            df = df[["event_id", self.time_col]]
            oced.add_event(activity, "event_id", self.time_col, df)

        # Objects
        object_df = log[[self.case_col]].drop_duplicates()
        oced.add_object(self.case_col, self.case_col, object_df)

        for object_col in self.object_cols:
            object_df = log[[object_col]].drop_duplicates()
            oced.add_object(object_col, object_col, object_df)

        # E2O
        rel_df = log[["event_id", self.case_col]].drop_duplicates()
        oced.add_event_to_object_relation(
            "modifies",
            "event_id",
            self.case_col,
            [self.case_col],
            rel_df,
        )

        for object_col in self.object_cols:
            rel_df = log[["event_id", object_col]].drop_duplicates()
            oced.add_event_to_object_relation(
                f"modifies_{object_col}",
                "event_id",
                object_col,
                [object_col],
                rel_df,
            )

        return oced
