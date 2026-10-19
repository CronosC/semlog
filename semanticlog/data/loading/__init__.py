"""Readers that turn source event logs into an OCED store.

`PM4PyLoader` reads OCEL 2.0. `ClassicalLoader` reads flat single-case logs.
`loader_for` picks one by file extension, and `load_and_clean` reads and
cleans a log in one call.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semanticlog.core.enums import OCELExtension
from semanticlog.core.errors import InputError
from semanticlog.data.binning import DEFAULT_BIN_LABELS
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.loading.base import Loader, LoaderColumns
from semanticlog.data.loading.classical_loader import (
    CLASSICAL_SUFFIXES,
    ClassicalLoader,
)
from semanticlog.data.loading.pm4py_loader import OCEL_SUFFIXES, PM4PyLoader
from semanticlog.data.oced import OCEDStore

LOG_SUFFIXES: frozenset[str] = OCEL_SUFFIXES | CLASSICAL_SUFFIXES


def loader_for(
    path: str,
    columns: Mapping[str, Any] | None = None,
    *,
    ocel_extensions: Sequence[OCELExtension] = (),
) -> Loader:
    """Return the loader that reads one file, chosen by its extension.

    Args:
        columns: Column names for a flat log. An OCEL log ignores them, but
            an unknown key is rejected for every log.
        ocel_extensions: OCEL 2.0 extensions to load. Ignored for a flat log.

    Raises:
        ConfigError: If a column setting names no column a loader reads.
        InputError: If no loader reads that extension.
    """
    named = LoaderColumns.from_mapping(columns)
    suffix = Path(path).suffix.lower()
    if suffix in OCEL_SUFFIXES:
        return PM4PyLoader(path, ocel_extensions=ocel_extensions)
    if suffix in CLASSICAL_SUFFIXES:
        return ClassicalLoader(path, named)
    raise InputError(
        f"No loader reads {suffix or path!r}. Supported extensions: "
        f"{', '.join(sorted(LOG_SUFFIXES))}."
    )


def load_and_clean(
    path: str | Path,
    extensions: Sequence[OCELExtension] = (),
    bin_labels: Sequence[str] = DEFAULT_BIN_LABELS,
    columns: Mapping[str, Any] | None = None,
    qualify_shared: bool = True,
) -> OCEDStore:
    """Read one event log and make its names safe to emit.

    Args:
        bin_labels: Bin labels the build will use. Cleaning reserves a class
            name for each.
        qualify_shared: Rename an attribute that more than one type carries
            to `<type>_<attribute>` in each type.
        columns: Column names for a flat `.csv` or `.xes` log. An OCEL log
            ignores them. An unknown key is rejected for every log.

    Returns:
        The cleaned store.

    Raises:
        ConfigError: If a column setting is unknown or names no column of the
            log.
        InputError: If no loader reads the extension, or the log holds
            something a store cannot, such as a repeated id or a missing id.
    """
    loader = loader_for(str(path), columns, ocel_extensions=extensions)
    return clean_oced_data(
        loader.get_oced(), bin_labels=bin_labels, qualify_shared=qualify_shared
    )


__all__ = [
    "CLASSICAL_SUFFIXES",
    "LOG_SUFFIXES",
    "OCEL_SUFFIXES",
    "ClassicalLoader",
    "Loader",
    "LoaderColumns",
    "PM4PyLoader",
    "load_and_clean",
    "loader_for",
]
