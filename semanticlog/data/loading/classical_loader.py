"""Reader for flat, single-case event logs.

`ClassicalLoader` reads a `.csv` or `.xes` log. The case column and each
column in `object_cols` become object types. The columns in
`object_attribute_cols` become attributes of those objects, and the other
columns become attributes of the events.
"""

import csv
import logging
import re
from pathlib import Path

import pandas as pd

from semanticlog.core.errors import ConfigError, InputError
from semanticlog.data.cleaning.names import free_name
from semanticlog.data.loading.base import Inference, Loader, LoaderColumns
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind, RelationKind

logger = logging.getLogger(__name__)

CLASSICAL_SUFFIXES: frozenset[str] = frozenset({".csv", ".xes"})

EVENT_ID = "event_id"

_SAMPLE = 5

_UNCLOSED_QUOTE = re.compile(r"EOF inside string starting at row (\d+)")


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


def _read_csv(path: str, text_cols: list[str]) -> pd.DataFrame:
    """Read a CSV log without the columns whose header is blank.

    A blank header is what pandas writes for its row index.

    Args:
        text_cols: Columns read as text, such as id columns. A column the
            file does not have is ignored.

    Raises:
        InputError: If the file is not UTF-8, is empty, or does not parse as
            CSV.
    """
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), [])
        log = pd.read_csv(path, dtype=dict.fromkeys(text_cols, str))
    except UnicodeDecodeError as exc:
        raise InputError(f"{path} is not UTF-8 text. Save it as UTF-8.") from exc
    except pd.errors.EmptyDataError as exc:
        raise InputError(
            f"{path} is empty. Give it a header line and one line per event."
        ) from exc
    except pd.errors.ParserError as exc:
        raise InputError(_parser_problem(path, exc)) from exc
    blank = [i for i, name in enumerate(header) if not name.strip()]
    if blank:
        logger.warning(
            "Dropped %d column(s) of %s with no header, at position(s) %s.",
            len(blank),
            path,
            ", ".join(str(i + 1) for i in blank),
        )
        log = log.drop(columns=log.columns[blank])
    return log


def _parser_problem(path: str, exc: pd.errors.ParserError) -> str:
    """Return the message for a CSV that pandas cannot tokenize, naming the line."""
    unclosed = _UNCLOSED_QUOTE.search(str(exc))
    if unclosed is not None:
        # pandas counts the lines of the file from 0.
        line = int(unclosed.group(1)) + 1
        return (
            f"{path} has a quote opened on line {line} that is never closed. "
            "Close the quote or remove it."
        )
    detail = str(exc).splitlines()[0].removeprefix("Error tokenizing data. C error: ")
    return f"{path} is not well-formed CSV: {detail.rstrip('.')}. Fix that line."


def _read_xes(path: str) -> pd.DataFrame:
    """Read an XES log into one row per event.

    Raises:
        InputError: If the file is not well-formed XML.
    """
    # Imported here so that the CLI starts without loading PM4Py.
    import pm4py as pm

    # lxml's XMLSyntaxError, which pm4py raises, subclasses SyntaxError.
    try:
        return pm.read_xes(path)
    except SyntaxError as exc:
        # str(exc) appends the file name, which the message already gives.
        raise InputError(
            f"{path} is not well-formed XES: {exc.msg}. Fix the file or export "
            "the log again."
        ) from exc


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

        Each event carries its row number as its position in the log.
        Ids are prefixed: events with `e` and the row number, cases with `t`,
        and the values of another object column with the column name. A CSV
        log's id columns are read as text, so `007` and `7` stay apart. Every
        event relates to its case through `involves`, and to another object
        column through `involves_<column>`. A column named in
        `object_attribute_cols` becomes an attribute of the first object type,
        the case first, within whose objects it does not vary. The other
        columns stay on the events.

        Returns:
            A store with one activity per distinct activity name.

        Raises:
            ConfigError: If a configured column is not in the file.
            InputError: If the file cannot be read, a row has no case id or
                no activity, the time column holds a value that is not a
                timestamp, or an object attribute column varies within every
                object type.
        """
        logger.info("Loading classical data from file %s", self.path)
        log = (
            _read_xes(self.path)
            if Path(self.path).suffix.lower() == ".xes"
            else _read_csv(
                self.path, [self.columns.case_col, *self.columns.object_cols]
            )
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
        owned = self._object_attributes(log, object_types)
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

        position = free_name("position", {str(c) for c in log.columns})
        for activity, rows in log.groupby(columns.act_col):
            inferred = inference.infer(
                rows[[event_id, columns.time_col, *event_attributes]], event_id
            )
            oced.add(
                EntityKind.EVENT,
                str(activity),
                inferred.assign(**{position: inferred.index}),
                id_col=event_id,
                time_col=columns.time_col,
                position_col=position,
            )

        for object_type in object_types:
            name = (
                "involves"
                if object_type == columns.case_col
                else f"involves_{object_type}"
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
        for setting, named in (
            ("object_cols", columns.object_cols),
            ("object_attribute_cols", columns.object_attribute_cols),
        ):
            missing = [c for c in named if c not in log.columns]
            if missing:
                raise ConfigError(
                    f"{setting} names {missing}, which are not columns of "
                    f"{self.path}. Available columns: "
                    f"{sorted(map(str, log.columns))}."
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
                raise InputError(
                    f"{int(blank.sum())} row(s) of {self.path} have no value in "
                    f"{setting} {column!r} ({self._places(log.index[blank])}). "
                    "Every row needs a case id and an activity. Fill them in or "
                    "drop the rows."
                )

    def _places(self, index: pd.Index) -> str:
        """Return where the first rows of `index` are in the file, as CSV lines or XES events."""
        if Path(self.path).suffix.lower() == ".csv":
            # Line 1 holds the header.
            return "lines " + ", ".join(str(i + 2) for i in index[:_SAMPLE])
        return "events " + ", ".join(str(i + 1) for i in index[:_SAMPLE])

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
        self, log: pd.DataFrame, object_types: list[str]
    ) -> dict[str, set[str]]:
        """Return the named object attribute columns, keyed by the object type each goes to.

        A column goes to the first type, in the order given, within whose
        objects it does not vary.

        Raises:
            InputError: If a named column varies within the objects of every
                type, naming the column, the last type tried and a sample of
                its objects with more than one value.
        """
        owned: dict[str, set[str]] = {name: set() for name in object_types}
        for column in self.columns.object_attribute_cols:
            for object_type in object_types:
                varying = log.groupby(object_type)[column].nunique(dropna=False)
                if (varying <= 1).all():
                    owned[object_type].add(column)
                    break
            else:
                sample = sorted(map(str, varying[varying > 1].index[:_SAMPLE]))
                raise InputError(
                    f"object_attribute_cols names {column!r}, which varies within "
                    f"the objects of every object type, last {object_type!r} "
                    f"(for example {', '.join(sample)}). Name a column that has "
                    "one value per object, or leave it on the events."
                )
        return owned
