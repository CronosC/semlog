"""The interface every loader implements, and the columns of a flat log.

`Loader` is the interface. `LoaderColumns` names the structural columns of a
flat log, and `Inference` infers column dtypes while loading.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import pandas as pd

from semanticlog.core.errors import ConfigError
from semanticlog.core.summary import count_phrase, counted
from semanticlog.data.dtypes import infer_dtypes
from semanticlog.data.oced import OCEDStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoaderColumns:
    """Which columns of a flat log carry its structure.

    The defaults are the XES column names. Every other column becomes an
    attribute.

    Attributes:
        case_col: Column holding the case id, which becomes an object type.
        act_col: Column holding the activity name.
        time_col: Column holding the event timestamp.
        object_cols: Further columns that each become an object type.
    """

    case_col: str = "case:concept:name"
    act_col: str = "concept:name"
    time_col: str = "time:timestamp"
    object_cols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.object_cols, str):
            raise ConfigError(
                f"object_cols must be a list, got the string {self.object_cols!r}. "
                f"Pass [{self.object_cols!r}]."
            )
        object.__setattr__(self, "object_cols", tuple(self.object_cols))

    @classmethod
    def keys(cls) -> tuple[str, ...]:
        """Return the keys a config may set."""
        return tuple(f.name for f in fields(cls))

    @classmethod
    def from_mapping(cls, named: Mapping[str, Any] | None) -> LoaderColumns:
        """Return the columns a config named, with the defaults for the rest.

        Args:
            named: Column names keyed by field name. None means the defaults.

        Raises:
            ConfigError: If a key is not a field name.
        """
        given = dict(named or {})
        unknown = sorted(set(given) - set(cls.keys()))
        if unknown:
            raise ConfigError(
                f"Unknown column setting(s): {', '.join(map(repr, unknown))}. "
                f"Valid settings: {', '.join(cls.keys())}."
            )
        return cls(**given)

    @property
    def structural(self) -> tuple[str, ...]:
        """Every structural column, in order."""
        return (self.case_col, self.act_col, self.time_col, *self.object_cols)


class Inference:
    """Infers the dtype of loaded columns and counts the rows and columns dropped."""

    def __init__(self) -> None:
        self.rows = 0
        self.columns = 0
        self.tables = 0

    def infer(self, data: pd.DataFrame, id_col: str) -> pd.DataFrame:
        """Return the table with dtypes inferred, and count the all-empty rows and columns dropped.

        Args:
            id_col: Column holding the id. It is not inferred.

        Returns:
            A new table. The input is not modified.
        """
        inferred = infer_dtypes(data, ignore_cols=[id_col])
        rows = len(data) - len(inferred)
        columns = len(data.columns) - len(inferred.columns)
        if rows or columns:
            self.rows += rows
            self.columns += columns
            self.tables += 1
        return inferred

    def report(self) -> None:
        """Log what inference dropped across every table of one load."""
        if not self.tables:
            return
        dropped = count_phrase(
            [
                (self.rows, "all-empty row", "all-empty rows"),
                (self.columns, "all-empty column", "all-empty columns"),
            ]
        )
        across = counted(self.tables, "table", "tables")
        logger.info("  dropped %s across %s", dropped, across)


class Loader(ABC):
    """Turns one source file into an OCED store.

    A loader does not clean names. Its output has to pass through
    `clean_oced_data` before a build.

    Attributes:
        path: File this loader reads.
    """

    path: str

    @abstractmethod
    def get_oced(self) -> OCEDStore:
        """Read the file and return it as a store."""
