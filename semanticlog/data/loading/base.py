"""The interface every loader implements."""

import logging
from abc import ABC, abstractmethod

import pandas as pd

from semanticlog.core.summary import count_phrase
from semanticlog.data.dtypes import infer_dtypes
from semanticlog.data.oced import OCEDStore

logger = logging.getLogger(__name__)


class Inference:
    """Gives loaded columns the dtype their content implies, and counts the losses.

    A source log carries most attributes as text, so which of them are really
    timestamps, numbers or booleans has to be read off the values. That is a
    judgement about a file rather than a property of the model, so a loader
    makes it and the store holds whatever it is handed.
    """

    def __init__(self) -> None:
        self.rows = 0
        self.columns = 0
        self.tables = 0

    def infer(self, data: pd.DataFrame, id_col: str) -> pd.DataFrame:
        """Return one table with its dtypes inferred, counting what came out empty.

        Args:
            data: Table as the source file had it.
            id_col: Column holding the id, left alone so that a numeric-looking
                id stays text.

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
        across = count_phrase([(self.tables, "table", "tables")])
        logger.info(f"  dropped {dropped} across {across}")


class Loader(ABC):
    """Turns one source file into an OCED store.

    A loader reads and reshapes. It does not clean names, so its output is
    still in the vocabulary of the source and has to pass through
    `clean_oced_data` before a build.
    """

    @abstractmethod
    def __init__(self, path, config):
        """Record where to read from and how.

        Args:
            path: File to read.
            config: Loader-specific options. Each loader documents its own
                keys and fills in the ones it needs.
        """
        self.path = path
        self.config = config

    @abstractmethod
    def get_oced(self) -> OCEDStore:
        """Read the file and return it as a store."""
