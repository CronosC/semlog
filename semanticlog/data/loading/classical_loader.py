"""Reader for flat, single-case event logs.

A classical log has one case notion and no objects, so the case column becomes
an object type of its own and every event relates to its case through
`modifies`. Any further column named in `object_cols` becomes another object
type the same way.

The remaining columns are attributes. A column whose value never varies within
one object belongs to that object rather than to its events, so it is lifted
onto the object table and the events keep the rest. Which columns moved is
logged, since the decision is read off the data.
"""

import logging
from typing import Any

import pandas as pd
import pm4py as pm

from semanticlog.data.loading.base import Inference, Loader
from semanticlog.data.oced import OCEDStore

logger = logging.getLogger(__name__)

EVENT_ID = "event_id"


class ClassicalLoader(Loader):
    """Loads a flat `.xes` or `.csv` event log."""

    def __init__(self, path: str, config: dict[str, Any]):
        """Prepare to read a classical log.

        Args:
            path: Path to a `.xes` or `.csv` file.
            config: Column names to read the log with. Recognised keys are
                `case_col`, `act_col` and `time_col`, which name one column
                each, and `object_cols`, which names a list. They default to
                the XES conventions and an empty object list. The dict is
                filled in with the defaults that were used.

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

        self.case_col: str = config["case_col"]
        self.act_col: str = config["act_col"]
        self.time_col: str = config["time_col"]
        self.object_cols: list[str] = config["object_cols"]

    def get_oced(self) -> OCEDStore:
        """Read the log and return it as a store.

        Ids are prefixed so that no two of them can collide: events with `e`
        and the row number, cases with `t`, and the values of any other object
        column with the column name.

        Returns:
            A store with one activity per distinct activity name, the case
            column as an object type, a `modifies` relation between them, and
            the remaining columns as attributes of whichever they belong to.

        Raises:
            ValueError: If a named column is not in the file.
        """
        logger.info(f"Loading classical data from file {self.path}..")
        log = (
            pm.read_xes(self.path)
            if self.path.endswith(".xes")
            else pd.read_csv(self.path)
        )
        self._require_columns(log)

        log = log.copy()
        log[EVENT_ID] = "e" + log.index.astype(str)
        log[self.case_col] = "t" + log[self.case_col].astype(str)
        for column in self.object_cols:
            log[column] = f"{column}_" + log[column].astype(str)

        object_types = [self.case_col, *self.object_cols]
        attributes = [
            column
            for column in log.columns
            if column not in {EVENT_ID, self.act_col, self.time_col, *object_types}
        ]
        owned = self._object_attributes(log, object_types, attributes)
        event_attributes = [
            c for c in attributes if c not in set().union(*owned.values())
        ]

        oced = OCEDStore()
        inference = Inference()
        for object_type in object_types:
            columns = [object_type, *sorted(owned[object_type])]
            oced.add_object(
                object_type,
                object_type,
                inference.infer(
                    log[columns].drop_duplicates(subset=[object_type]), object_type
                ),
            )

        for activity, rows in log.groupby(self.act_col):
            oced.add_event(
                str(activity),
                EVENT_ID,
                self.time_col,
                inference.infer(
                    rows[[EVENT_ID, self.time_col, *event_attributes]], EVENT_ID
                ),
            )

        for object_type in object_types:
            name = (
                "modifies"
                if object_type == self.case_col
                else f"modifies_{object_type}"
            )
            oced.add_event_to_object_relation(
                name,
                EVENT_ID,
                object_type,
                [object_type],
                log[[EVENT_ID, object_type]].drop_duplicates(),
            )

        inference.report()
        return oced

    def _require_columns(self, log: pd.DataFrame) -> None:
        """Raise unless every column this loader was configured with is there.

        Raises:
            ValueError: Naming the setting, the column and what is available.
        """
        for setting, column in (
            ("case_col", self.case_col),
            ("act_col", self.act_col),
            ("time_col", self.time_col),
        ):
            if column not in log.columns:
                raise ValueError(
                    f"{setting} {column!r} is not a column of {self.path}. "
                    f"Available columns: {sorted(map(str, log.columns))}"
                )
        missing = [c for c in self.object_cols if c not in log.columns]
        if missing:
            raise ValueError(
                f"object_cols names {missing}, which are not columns of "
                f"{self.path}. Available columns: {sorted(map(str, log.columns))}"
            )

    def _object_attributes(
        self, log: pd.DataFrame, object_types: list[str], attributes: list[str]
    ) -> dict[str, set[str]]:
        """Decide which attribute columns describe an object rather than an event.

        A column that holds one value for every row of an object describes
        that object. One that varies within an object cannot, so it stays on
        the events. A column constant within two object types goes to the
        first one named, since it can only be an attribute of one.

        Args:
            log: The whole log, with its ids already prefixed.
            object_types: Columns that became object types.
            attributes: Columns that are neither ids nor the activity or time.

        Returns:
            Object type to the attribute columns lifted onto it.
        """
        owned: dict[str, set[str]] = {name: set() for name in object_types}
        taken: set[str] = set()
        for object_type in object_types:
            for column in attributes:
                if column in taken:
                    continue
                if (log.groupby(object_type)[column].nunique(dropna=False) <= 1).all():
                    owned[object_type].add(column)
                    taken.add(column)
        for object_type, columns in owned.items():
            if columns:
                logger.info(
                    f"classicalloader: {', '.join(sorted(columns))} do not vary "
                    f"within a '{object_type}' and are read as its attributes."
                )
        return owned
