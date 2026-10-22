"""Reading names out of a store, and writing a rename into every table.

The `apply_*` functions write a rename into every table that holds the name.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

import pandas as pd

from semanticlog.data.cleaning.names import cleaned, is_cleaned_column
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import TableView

logger = logging.getLogger(__name__)


def attribute_names(data: OCEDStore) -> set[str]:
    """Return every attribute column of events, objects and change tables."""
    return {name for view in data.tables() for name in view.attributes}


def entity_id_columns(data: OCEDStore) -> list[tuple[str, pd.Series]]:
    """Return the distinct ids of each event and object table, as text."""
    columns = []
    for view in data.tables(changes=False):
        if view.id_column is None:
            continue
        ids = pd.Series(view.frame[view.id_column].dropna().unique())
        columns.append((view.origin, ids.astype(str)))
    return columns


def entity_ids(data: OCEDStore) -> pd.Series:
    """Return every event and object id of the store as one text Series."""
    columns = [ids for _, ids in entity_id_columns(data)]
    if not columns:
        return pd.Series(dtype="object")
    return pd.concat(columns, ignore_index=True)


def text_values(view: TableView, column: str) -> list[str]:
    """Return the distinct text values of one column."""
    if not is_cleaned_column(view.frame[column]):
        return []
    return [
        value
        for value in view.frame[column].dropna().unique()
        if isinstance(value, str)
    ]


def value_groups(data: OCEDStore) -> dict[tuple[str, str], list[str]]:
    """Return the distinct text values of each attribute, keyed by type and cleaned attribute.

    A change table counts as its object type.
    """
    groups: dict[tuple[str, str], list[str]] = {}
    for view in data.tables():
        for column in view.attributes:
            key = (view.name, cleaned(column))
            groups.setdefault(key, []).extend(text_values(view, column))
    return groups


def _is_text_column(values: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values)


def remap(
    df: pd.DataFrame,
    columns: Iterable[str],
    renames: Mapping[str, str],
    *,
    as_text: bool = False,
) -> pd.DataFrame:
    """Return a copy of the table with columns rewritten through a rename map.

    Args:
        columns: Columns to rewrite. A column the table lacks is skipped.
        as_text: Match every cell by its text form, and convert a column that
            is not text to text. Otherwise only text cells are matched.
    """
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            continue
        values = out[column]
        if as_text:
            if not _is_text_column(values):
                values = values.astype(str).where(values.notna())
            out[column] = values.map(lambda v: renames.get(str(v), v))
        else:
            out[column] = values.map(
                lambda v: renames.get(v, v) if isinstance(v, str) else v
            )
    return out


def apply_id_renames(data: OCEDStore, renames: Mapping[str, str]) -> None:
    """Rewrite ids in every id column, relation endpoint and change table `ocel:oid`.

    An id column that is not text becomes text.
    """
    for view in data.tables():
        columns = [view.id_column] if view.id_column is not None else ["ocel:oid"]
        view.replace(remap(view.frame, columns, renames, as_text=True))
    for name, relation in data.relations():
        endpoints = [relation.from_id, relation.to_id]
        data.replace_relation_data(
            name, remap(relation.data, endpoints, renames, as_text=True)
        )


def apply_value_renames(
    data: OCEDStore,
    attribute: str,
    renames: Mapping[str, str],
    *,
    type_name: str | None = None,
) -> None:
    """Rewrite values in every column that carries the attribute, change tables included.

    Only the tables of `type_name` are rewritten when it is given.
    """
    for view in data.tables():
        if type_name is not None and view.name != type_name:
            continue
        columns = [c for c in view.attributes if cleaned(c) == attribute]
        if columns:
            view.replace(remap(view.frame, columns, renames))


def apply_attribute_renames(data: OCEDStore, renames: Mapping[str, str]) -> None:
    """Rename attribute columns in every table, and in the `ocel:field` of change rows."""
    for view in data.tables():
        frame = view.frame.rename(columns=renames)
        if "ocel:field" in frame.columns:
            frame = remap(frame, ["ocel:field"], renames)
        view.replace(frame)
