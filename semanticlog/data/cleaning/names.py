"""Rewriting a source name into one an IRI can hold.

Every activity, object type, relation, attribute, id and text value in a log
becomes part of an IRI, so anything outside `[A-Za-z0-9_.-]` has to go. These
are the primitives that do it, and they know nothing about a store: what has
to be cleaned is decided in `collisions`, and the order it happens in is
decided in `clean_oced_data`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

_ISO8601_RE = re.compile(
    r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2})?"
    r"(?:\.\d+)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?)?$"
)

# What an IRI local name may hold. Wider than it looks: `-` and `.` survive
# both an XML NCName and a Turtle prefixed name, which are the two forms the
# sinks write, so folding them into `_` would merge names for nothing.
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]")

# What marks a column of a change table as structure rather than an attribute.
_OCEL_PREFIX = "ocel:"


def is_timestamp(string: str) -> bool:
    """Report whether text looks like an ISO 8601 date or timestamp.

    Timestamps are exempt from cleaning, because rewriting their punctuation
    would destroy the format that later parsing depends on.
    """
    return bool(_ISO8601_RE.match(string))


def clean_string(string: str) -> str:
    """Replace every character that cannot appear in an IRI local name.

    Args:
        string: Text to clean.

    Returns:
        The cleaned text, with each unsafe character replaced by an underscore.
    """
    return _UNSAFE_CHARS.sub("_", string)


def cleaned(name: object) -> str:
    """Return the emitted name of anything the store holds a name under."""
    return clean_string(str(name))


def clean_series(values: pd.Series) -> pd.Series:
    """Clean a whole column of names at once, as `clean_string` does one.

    A log holds one id per event and per object, so the passes that compare
    them all go through this rather than calling `clean_string` per value.
    """
    return values.str.replace(_UNSAFE_CHARS, "_", regex=True)


def is_cleaned_column(column: pd.Series) -> bool:
    """Report whether cleaning will rewrite the cells of a column."""
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
        return value if is_timestamp(value) else fn(value)
    if isinstance(value, pd.DataFrame):
        return df_fn(value) if df_fn is not None else value
    if isinstance(value, (list, pd.arrays.StringArray, np.ndarray)):
        return [_apply_to_tree(x, fn, df_fn) for x in value]
    return value


def clean_value(
    value: dict | str | int | float | bool | pd.DataFrame,
) -> dict | str | int | float | bool | pd.DataFrame:
    """Clean every string in a value, whatever shape it has.

    Cells in an OCEL log can hold nested structures as well as plain values,
    so dicts, lists, arrays and frames are all walked. Numbers, booleans and
    timestamps are returned unchanged.

    Args:
        value: Value to clean.

    Returns:
        A cleaned value of the same shape.
    """
    return _apply_to_tree(value, clean_string, df_fn=clean_dataframe)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the text cells of a table and its column names.

    Numeric, boolean and datetime columns are left alone, because their values
    never reach an IRI as text.

    Args:
        df: Table to clean.

    Returns:
        A new table. The input is not modified.
    """
    clean_df = df.copy()
    for column in clean_df.columns:
        if is_cleaned_column(clean_df[column]):
            clean_df[column] = clean_df[column].map(clean_value)
    return clean_df.rename(columns={c: clean_string(c) for c in clean_df.columns})


def clean_object_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Clean an attribute change table, leaving its OCEL columns intact.

    The `ocel:` columns carry structure, so their names are kept. Their
    contents are still cleaned where they name an object, a type or a field,
    since those become part of an IRI.

    Args:
        df: Change table to clean.

    Returns:
        A new table. The input is not modified.
    """
    df = df.copy()

    if "ocel:oid" in df.columns:
        df["ocel:oid"] = df["ocel:oid"].apply(_clean_str_value)

    if "ocel:type" in df.columns:
        df["ocel:type"] = df["ocel:type"].apply(_clean_str_value)

    system_cols = {c for c in df.columns if c.startswith(_OCEL_PREFIX)}
    attr_cols = [c for c in df.columns if c not in system_cols]

    if not attr_cols:
        return df

    rename_map = {c: clean_string(c) for c in attr_cols}
    df = df.rename(columns=rename_map)

    if "ocel:field" in df.columns:
        df["ocel:field"] = df["ocel:field"].apply(_clean_str_value)

    for new_col in rename_map.values():
        df[new_col] = df[new_col].apply(
            lambda v: (
                clean_string(str(v))
                if isinstance(v, str) and not is_timestamp(str(v))
                else v
            )
        )

    return df


def _clean_str_value(v: Any) -> Any:
    """Clean a cell if it holds text, and pass anything else through."""
    return clean_string(str(v)) if isinstance(v, str) else v
