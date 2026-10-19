"""Reader for flat, single-case event logs.

`ClassicalLoader` reads a `.csv` or `.xes` log. The case column and each
column in `object_cols` become object types, and the other columns become
attributes.
"""

import csv
import logging
from pathlib import Path

import pandas as pd
import pm4py as pm

from semanticlog.core.errors import ConfigError, InputError
from semanticlog.data.cleaning.names import free_name
from semanticlog.data.loading.base import Inference, Loader, LoaderColumns
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind, RelationKind

logger = logging.getLogger(__name__)

CLASSICAL_SUFFIXES: frozenset[str] = frozenset({".csv", ".xes"})

EVENT_ID = "event_id"

_SAMPLE = 5


def _free_column(log: pd.DataFrame, name: str) -> str:
    """Return `name`, or `name` with a number appended, whichever is not a column of the log."""
    free = free_name(name, set(log.columns))
    if free != name:
        logger.info(
            "Column %r is in the log and is read as an "
            "attribute. The generated event id goes in %r.",
            name,
            free,
        )
    return free


def _read_csv(path: str) -> pd.DataFrame:
    """Read a CSV log without the columns whose header is blank.

    A blank header is what pandas writes for its row index.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), [])
    blank = [i for i, name in enumerate(header) if not name.strip()]
    log = pd.read_csv(path)
    if blank:
        logger.warning(
            "Dropped %d column(s) of %s with no header, at position(s) %s.",
            len(blank),
            path,
            ", ".join(str(i + 1) for i in blank),
        )
        log = log.drop(columns=log.columns[blank])
    return log


class ClassicalLoader(Loader):
    """Loads a flat `.xes` or `.csv` event log.

    Args:
        columns: Which columns carry the structure of the log. The XES
            column names when None.

    Raises:
        InputError: If the file is neither `.xes` nor `.csv`.
    """

    def __init__(self, path: str, columns: LoaderColumns | None = None) -> None:
        self.path = path
        if Path(self.path).suffix.lower() not in CLASSICAL_SUFFIXES:
            raise InputError("Can only load .xes or .csv classical event logs.")
        self.columns = columns if columns is not None else LoaderColumns()

    def get_oced(self) -> OCEDStore:
        """Read the log and return it as a store.

        Ids are prefixed: events with `e` and the row number, cases with `t`,
        and the values of another object column with the column name. Every
        event relates to its case through `modifies`, and to another object
        column through `modifies_<column>`. A column constant within each
        object of a type becomes an attribute of that type. The other columns
        stay on the events.

        Returns:
            A store with one activity per distinct activity name.

        Raises:
            ConfigError: If a configured column is not in the file.
            InputError: If a row has no case id or no activity, or if the
                time column holds a value that is not a timestamp.
        """
        logger.info("Loading classical data from file %s", self.path)
        log = (
            pm.read_xes(self.path)
            if Path(self.path).suffix.lower() == ".xes"
            else _read_csv(self.path)
        )
        self._require_columns(log)
        self._require_filled(log)

        columns = self.columns
        log = log.copy()
        log[columns.time_col] = self._timestamps(log[columns.time_col])
        event_id = _free_column(log, EVENT_ID)
        log[event_id] = "e" + log.index.astype(str)
        log[columns.case_col] = "t" + log[columns.case_col].astype(str)
        for column in columns.object_cols:
            prefixed = f"{column}_" + log[column].astype(str)
            log[column] = prefixed.where(log[column].notna())

        object_types = [columns.case_col, *columns.object_cols]
        attributes = [
            column
            for column in log.columns
            if column
            not in {event_id, columns.act_col, columns.time_col, *object_types}
        ]
        owned = self._object_attributes(log, object_types, attributes)
        event_attributes = [
            c for c in attributes if c not in set().union(*owned.values())
        ]

        oced = OCEDStore()
        inference = Inference()
        for object_type in object_types:
            table = log[[object_type, *sorted(owned[object_type])]]
            table = table.dropna(subset=[object_type]).drop_duplicates(
                subset=[object_type]
            )
            oced.add(
                EntityKind.OBJECT,
                object_type,
                inference.infer(table, object_type),
                id_col=object_type,
            )

        for activity, rows in log.groupby(columns.act_col):
            oced.add(
                EntityKind.EVENT,
                str(activity),
                inference.infer(
                    rows[[event_id, columns.time_col, *event_attributes]], event_id
                ),
                id_col=event_id,
                time_col=columns.time_col,
            )

        for object_type in object_types:
            name = (
                "modifies"
                if object_type == columns.case_col
                else f"modifies_{object_type}"
            )
            oced.add_relation(
                RelationKind.E2O,
                name,
                log[[event_id, object_type]].dropna().drop_duplicates(),
                from_id=event_id,
                to_id=object_type,
                to_type=[object_type],
            )

        inference.report()
        return oced

    def _require_columns(self, log: pd.DataFrame) -> None:
        """Raise unless every configured column is in the log.

        Raises:
            ConfigError: If a configured column is not in the log.
        """
        columns = self.columns
        for setting, column in (
            ("case_col", columns.case_col),
            ("act_col", columns.act_col),
            ("time_col", columns.time_col),
        ):
            if column not in log.columns:
                raise ConfigError(
                    f"{setting} {column!r} is not a column of {self.path}. "
                    f"Available columns: {sorted(map(str, log.columns))}."
                )
        missing = [c for c in columns.object_cols if c not in log.columns]
        if missing:
            raise ConfigError(
                f"object_cols names {missing}, which are not columns of "
                f"{self.path}. Available columns: {sorted(map(str, log.columns))}."
            )

    def _require_filled(self, log: pd.DataFrame) -> None:
        """Raise if a row has no case id or no activity.

        Raises:
            InputError: Naming the column and the first rows without a value.
        """
        for setting, column in (
            ("case_col", self.columns.case_col),
            ("act_col", self.columns.act_col),
        ):
            values = log[column]
            blank = values.isna() | values.astype(str).str.strip().eq("")
            if blank.any():
                rows = list(log.index[blank][:_SAMPLE])
                raise InputError(
                    f"{int(blank.sum())} row(s) of {self.path} have no value in "
                    f"{setting} {column!r} (rows {rows}). Every row needs a case "
                    "id and an activity. Fill them in or drop the rows."
                )

    def _timestamps(self, values: pd.Series) -> pd.Series:
        """Return the time column as timestamps.

        Raises:
            InputError: If a value that is not empty does not parse as an
                ISO 8601 timestamp.
        """
        if pd.api.types.is_datetime64_any_dtype(values):
            return values
        parsed = pd.to_datetime(values, errors="coerce", utc=True, format="ISO8601")
        failed = values.notna() & parsed.isna()
        if failed.any():
            sample = list(values[failed].astype(str)[:_SAMPLE])
            raise InputError(
                f"time_col {self.columns.time_col!r} holds {int(failed.sum())} "
                f"value(s) that are not ISO 8601 timestamps, such as {sample}. "
                "Write the timestamps as ISO 8601, or name another column."
            )
        return parsed

    def _object_attributes(
        self, log: pd.DataFrame, object_types: list[str], attributes: list[str]
    ) -> dict[str, set[str]]:
        """Return the columns constant within each object, keyed by object type.

        A column constant within two object types goes to the first type
        named.
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
                    "%s do not vary within a '%s' and are read as its attributes.",
                    ", ".join(sorted(columns)),
                    object_type,
                )
        return owned
