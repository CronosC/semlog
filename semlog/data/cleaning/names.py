"""Rewriting a source name into one an IRI can hold.

These functions take strings, Series and frames and do not read a store.
`clean_oced_data` decides what is cleaned and in which order.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection
from typing import Any

import numpy as np
import pandas as pd

from semlog.core.names import candidates

# Characters an IRI local name may hold. `-` and `.` are legal in an XML
# NCName, which RDF/XML writes, and in a prefixed name of functional syntax.
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]")

# Neither form accepts `.` or `-` as the first character, or `.` as the last.
# An empty name would leave a value individual's name ending in `.`.
_UNSAFE_EDGES = re.compile(r"^[.-]|\.$|^$")


def clean_string(string: str) -> str:
    """Replace each character an IRI local name cannot hold with an underscore."""
    return _UNSAFE_EDGES.sub("_", _UNSAFE_CHARS.sub("_", string))


def cleaned(name: object) -> str:
    """Return the cleaned text form of a name of any type."""
    return clean_string(str(name))


def clean_attribute(name: object) -> str:
    """Return the cleaned form of an attribute name, with each `.` replaced too.

    A value individual is named `<attribute>.<value>`, so the first `.` of
    the name ends the attribute.
    """
    return cleaned(name).replace(".", "_")


def clean_series(values: pd.Series) -> pd.Series:
    """Clean every value of a Series, as `clean_string` does one string."""
    replaced = values.str.replace(_UNSAFE_CHARS, "_", regex=True)
    return replaced.str.replace(_UNSAFE_EDGES, "_", regex=True)


def clean_attribute_series(values: pd.Series) -> pd.Series:
    """Clean every value of a Series, as `clean_attribute` does one name."""
    return clean_series(values).str.replace(".", "_", regex=False)


def free_name(name: str, taken: Collection[str], prefix: str = "") -> str:
    """Return the first of `candidates(name, prefix)` that is not in `taken`."""
    return next(c for c in candidates(name, prefix) if c not in taken)


def is_cleaned_column(column: pd.Series) -> bool:
    """Report whether cleaning rewrites the cells of a column."""
    return not (
        pd.api.types.is_numeric_dtype(column)
        or pd.api.types.is_datetime64_any_dtype(column)
        or pd.api.types.is_bool_dtype(column)
    )


def _apply_to_tree(
    value: Any,
    fn: Callable[[str], Any],
    df_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> Any:
    """Apply a cleaner to every string inside a nested value."""
    if isinstance(value, dict):
        return {
            _apply_to_tree(k, fn, df_fn): _apply_to_tree(v, fn, df_fn)
            for k, v in value.items()
        }
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, pd.DataFrame):
        return df_fn(value) if df_fn is not None else value
    if isinstance(value, (list, pd.arrays.StringArray, np.ndarray)):
        return [_apply_to_tree(x, fn, df_fn) for x in value]
    return value


def clean_value(
    value: dict | str | int | float | bool | pd.DataFrame,
) -> dict | str | int | float | bool | pd.DataFrame:
    """Return a value of the same shape with every string in it cleaned."""
    return _apply_to_tree(value, clean_string, df_fn=clean_dataframe)


def clean_dataframe(df: pd.DataFrame, keep: Collection[str] = ()) -> pd.DataFrame:
    """Return a copy of a table with its text cells and column names cleaned.

    Args:
        keep: Columns whose names are kept. Their cells are cleaned.
    """
    clean_df = df.copy()
    for column in clean_df.columns:
        if is_cleaned_column(clean_df[column]):
            clean_df[column] = clean_df[column].map(clean_value)
    return clean_df.rename(
        columns={c: cleaned(c) for c in clean_df.columns if c not in keep}
    )
