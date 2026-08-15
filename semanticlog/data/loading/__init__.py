"""Readers that turn source event logs into an OCED store.

`PM4PyLoader` is the one to use for OCEL 2.0. `ClassicalLoader` covers flat
single-case logs, which have no object-centric structure of their own.
`loader_for` picks between them by file extension, and `load_and_clean` is the
whole first step of a build.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semanticlog.core.enums import OCELExtension
from semanticlog.core.errors import InputError
from semanticlog.data.binning import DEFAULT_BIN_LABELS
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.loading.base import Loader
from semanticlog.data.loading.classical_loader import (
    CLASSICAL_SUFFIXES,
    ClassicalLoader,
)
from semanticlog.data.loading.pm4py_loader import OCEL_SUFFIXES, PM4PyLoader
from semanticlog.data.oced import OCEDStore

LOG_SUFFIXES: frozenset[str] = OCEL_SUFFIXES | CLASSICAL_SUFFIXES


def loader_for(
    path: str,
    config: dict[str, Any],
    *,
    ocel_extensions: Sequence[OCELExtension] = (),
) -> Loader:
    """Return the loader that reads one file, chosen by its extension.

    Args:
        path: Log to read.
        config: Reader settings. A classical log takes its column names from
            here, an OCEL log its format. Either loader fills in the defaults
            it used.
        ocel_extensions: OCEL 2.0 extensions to load as well. A classical log
            has none.

    Returns:
        A loader bound to that file.

    Raises:
        ValueError: If no loader reads that extension.
    """
    suffix = Path(path).suffix.lower()
    if suffix in OCEL_SUFFIXES:
        return PM4PyLoader(path, config, ocel_extensions=list(ocel_extensions))
    if suffix in CLASSICAL_SUFFIXES:
        return ClassicalLoader(path, config)
    raise InputError(
        f"no loader reads {suffix or path!r}. Supported extensions: "
        f"{', '.join(sorted(LOG_SUFFIXES))}."
    )


def load_and_clean(
    path: str | Path,
    extensions: Sequence[OCELExtension] = (),
    bin_labels: Sequence[str] = DEFAULT_BIN_LABELS,
    columns: Mapping[str, Any] | None = None,
) -> OCEDStore:
    """Read one event log and make its names safe to emit.

    This is the whole first step of a build. Nothing below it may be skipped,
    since a build refuses a store cleaning has not been through.

    Args:
        path: Log to read. The extension chooses the loader.
        extensions: OCEL extensions to load as well.
        bin_labels: Bin labels the build will use, which cleaning reserves
            along with the rest of the names a build derives for itself. The
            default is what a build reserves when the labels are left alone,
            so `inspect` reports the names a build would use. A run with its
            own labels passes them, and only then can the two differ.
        columns: Column names for a flat `.csv` or `.xes` log. An OCEL log
            carries its own structure and ignores them.

    Returns:
        The cleaned store.
    """
    loader = loader_for(str(path), dict(columns or {}), ocel_extensions=extensions)
    return clean_oced_data(loader.get_oced(), bin_labels=bin_labels)


__all__ = [
    "CLASSICAL_SUFFIXES",
    "LOG_SUFFIXES",
    "OCEL_SUFFIXES",
    "ClassicalLoader",
    "Loader",
    "PM4PyLoader",
    "load_and_clean",
    "loader_for",
]
