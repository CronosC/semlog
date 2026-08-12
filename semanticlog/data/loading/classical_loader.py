"""Reader for flat, single-case event logs.

A classical log has one case notion and no objects, so the case column becomes
an object type of its own and every event relates to its case through
`modifies`. Any further column named in `object_cols` becomes another object
type the same way.
"""

import logging

import pandas as pd
import pm4py as pm

from semanticlog.data.loading.base import Loader
from semanticlog.data.oced import OCEDStore

logger = logging.getLogger(__name__)


class ClassicalLoader(Loader):
    """Loads a flat `.xes` or `.csv` event log."""

    def __init__(self, path: str, config: dict[str, str]):
        """Prepare to read a classical log.

        Args:
            path: Path to a `.xes` or `.csv` file.
            config: Column names to read the log with. Recognised keys are
                `case_col`, `act_col`, `time_col` and `object_cols`, which
                default to the XES conventions and an empty object list. The
                dict is filled in with the defaults that were used.

        Raises:
            ValueError: If the file is neither `.xes` nor `.csv`.
        """
        super().__init__(path, config)
        if not (self.path.endswith(".xes") or self.path.endswith(".csv")):
            raise ValueError("Can only load .xes or .csv classical event logs.")

        config.setdefault("case_col", "case:concept:name")
        config.setdefault("act_col", "concept:name")
        config.setdefault("time_col", "time:timestamp")
        config.setdefault("object_cols", [])

        self.case_col = config["case_col"]
        self.act_col = config["act_col"]
        self.time_col = config["time_col"]
        self.object_cols = config["object_cols"]

    def get_oced(self) -> OCEDStore:
        """Read the log and return it as a store.

        Event ids are generated from the row index with an `e` prefix, and
        case ids are prefixed with `t`, so that neither can collide with a
        value that came from the log itself.

        Returns:
            A store with one activity per distinct activity name, the case
            column as an object type, and a `modifies` relation between them.

        Raises:
            ValueError: If the case column is not in the file.
        """
        logger.info(f"Loading classical data from file {self.path}..")
        if self.path.endswith(".xes"):
            log = pm.read_xes(self.path)
        elif self.path.endswith(".csv"):
            log = pd.read_csv(self.path)
        else:
            raise ValueError("Can only load .xes or .csv classical event logs.")
        log["event_id"] = "e" + log.index.astype(str)
        if self.case_col not in log.columns:
            raise ValueError(
                f"case_col {self.case_col!r} is not a column of {self.path}. "
                f"Available columns: {sorted(log.columns)}"
            )
        log[self.case_col] = "t" + log[self.case_col].astype(str)

        oced = OCEDStore()

        event_dfs = {
            activity: df
            for activity, df in log.drop(columns=[self.case_col]).groupby(self.act_col)
        }

        for activity in event_dfs:
            df = event_dfs[activity]
            df = df[["event_id", self.time_col]]
            oced.add_event(activity, "event_id", self.time_col, df)

        object_df = log[[self.case_col]].drop_duplicates()
        oced.add_object(self.case_col, self.case_col, object_df)

        for object_col in self.object_cols:
            object_df = log[[object_col]].drop_duplicates()
            oced.add_object(object_col, object_col, object_df)

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
