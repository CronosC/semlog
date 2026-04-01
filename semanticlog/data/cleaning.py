import logging
import re
from typing import Any, Callable, cast

import numpy as np
import pandas as pd

from semanticlog.data.oced import OCEDStore

_ISO8601_RE = re.compile(
    r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2})?"
    r"(?:\.\d+)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?)?$"
)


def null_check(data: OCEDStore) -> None:
    for activity in data.get_activities():
        activity_dict = data.get_event(activity)
        df = cast(pd.DataFrame, activity_dict["data"])
        for col in df.columns:
            n_null = df[col].isnull().sum()
            if n_null > 0:
                logging.info(f"{activity}[{col}]: {n_null} null values.")
    for object_type in data.get_object_types():
        object_dict = data.get_object(object_type)
        df = cast(pd.DataFrame, object_dict["data"])
        for col in df.columns:
            n_null = df[col].isnull().sum()
            if n_null > 0:
                logging.info(f"{object_type}[{col}]: {n_null} null values.")


def fix_duplicate_names(data: OCEDStore) -> OCEDStore:
    names = [
        "is_a",
        "observes",
        "observed_at",
        "df",
        "ef",
        "has_attribute",
        "attribute_of",
        "event_object_relation",
        "object_object_relation",
        "Starts",
        "Ends",
    ]
    attribute_names: list[str] = []

    relation_types = data.get_relation_types()
    for relation_type in relation_types:
        if relation_type in names:
            logging.info(f"  renaming relation {relation_type} to avoid collision..")
            data.rename_relation(relation_type, relation_type + "_")
        else:
            names.append(relation_type)

    activities = data.get_activities()
    for activity in activities:
        if activity in names:
            logging.info(f"  renaming activity {activity} to avoid collision..")
            data.rename_activity(activity, activity + "_")
            names.append(activity + "_")
        else:
            names.append(activity)

    object_types = list(data.get_object_types())
    for object_type in object_types:
        if object_type in names:
            logging.info(f"  renaming object {object_type} to avoid collision..")
            data.rename_object(object_type, object_type + "_")
            names.append(object_type + "_")
        else:
            names.append(object_type)

    # get event attribute names
    activities = list(data.get_activities())
    for activity in activities:
        activity_data = data.get_event(activity)
        id_col = activity_data["id"]
        time_col = activity_data["time"]
        attributes = list(cast(pd.DataFrame, activity_data["data"]).columns.values)
        attributes.remove(id_col)
        attributes.remove(time_col)
        attribute_names = attribute_names + attributes

    # get object attribute names
    object_types = data.get_object_types()
    for object_type in object_types:
        object_data = data.get_object(object_type)
        id_col = object_data["id"]
        attributes = list(cast(pd.DataFrame, object_data["data"]).columns.values)
        attributes.remove(id_col)
        attribute_names = attribute_names + attributes

    # fix attribute names that are also objects/events/props
    attributes_names_overlap = [
        a for a in attribute_names if a.casefold() in [n.casefold() for n in names]
    ]

    def _make_attr_renamer(a: str) -> Callable[[Any], Any]:
        def _rename(x: Any) -> Any:
            return x if x != a else x + "_attr"

        return _rename

    for attribute in attributes_names_overlap:
        replace_function: Callable[[Any], Any] = _make_attr_renamer(attribute)
        for activity in activities:
            data._events[activity]["data"] = apply_function_to_value(
                data._events[activity]["data"],
                replace_function,
                replace_df_values=False,
            )
        for object_type in object_types:
            data._objects[object_type]["data"] = apply_function_to_value(
                data._objects[object_type]["data"],
                replace_function,
                replace_df_values=False,
            )

    # fix attribute values that are also objects/events/props
    all_names = names + attribute_names

    def replace_attr_val(x: Any) -> Any:
        return x if x not in all_names else x + "_attr_val"

    for activity in activities:
        data._events[activity]["data"] = apply_function_to_value(
            data._events[activity]["data"], replace_attr_val, rename_df_columns=False
        )
    for object_type in object_types:
        data._objects[object_type]["data"] = apply_function_to_value(
            data._objects[object_type]["data"],
            replace_attr_val,
            rename_df_columns=False,
        )
    return data


def clean_oced_data(data: OCEDStore, casefold=False) -> OCEDStore:
    """cleans all keys and values in all dicts and dataframes of a OCEDStore"""
    logging.info("Cleaning data..")
    data._events = clean_value(data._events, casefold=casefold)
    data._objects = clean_value(data._objects, casefold=casefold)
    data._relations = clean_value(data._relations, casefold=casefold)
    if data._ocel_mode:
        data._object_changes = {
            obj_type: clean_object_changes_df(df, casefold=casefold)
            for obj_type, df in data._object_changes.items()
        }
    data = fix_duplicate_names(data)
    data.update_used_ids()
    return data


def apply_function_to_value(
    value: dict | str | int | float | bool | pd.DataFrame,
    function: Callable[..., Any],
    rename_df_columns=True,
    replace_df_values=True,
) -> str | int | float | bool | pd.DataFrame:
    """
    Applies a given function recursively to values, for example in the OCEDStore. Value can be dicts, in which case this function is recursively applied to all keys and values. They can also be strings in which case they are replaced by the result of the function call on the string. Lastly they can also be dataframes, in that case all entries and all column names are replaced with the result of the function call on the value. In all other cases the functions just returns the value.
    """
    if isinstance(value, dict):
        return {
            apply_function_to_value(key, function): apply_function_to_value(
                inner_value, function
            )
            for key, inner_value in value.items()
        }
    if isinstance(value, str):
        if is_timestamp(value):
            return value
        else:
            return function(value)
    if isinstance(value, pd.DataFrame):
        if replace_df_values:
            value = value.map(function)
        if rename_df_columns:
            value = value.rename(
                columns={
                    col_str: apply_function_to_value(col_str, function)
                    for col_str in value.columns
                }
            )
        return value
    if isinstance(value, list):
        return [apply_function_to_value(x, function) for x in value]
    if isinstance(value, pd.arrays.StringArray):
        return [apply_function_to_value(x, function) for x in value]
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, pd.Timestamp):
        return value
    raise TypeError(f"Forgotten case: {type(value)}")


def clean_value(
    value: dict | str | int | float | bool | pd.DataFrame, casefold: bool = True
) -> str | int | float | bool | pd.DataFrame:
    """applies string cleaning to some value, handling different datatypes appropriately"""
    if isinstance(value, dict):
        return {
            clean_value(key, casefold=casefold): clean_value(
                inner_value, casefold=casefold
            )
            for key, inner_value in value.items()
        }
    if isinstance(value, str):
        if is_timestamp(value):
            return value
        else:
            return clean_string(value, casefold=casefold)
    if isinstance(value, pd.DataFrame):
        return clean_dataframe(value, casefold=casefold)
    if isinstance(value, list):
        return [clean_value(x, casefold=casefold) for x in value]
    if isinstance(value, pd.arrays.StringArray):
        return [clean_value(x, casefold=casefold) for x in value]
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, np.ndarray):
        return [clean_value(x, casefold=casefold) for x in value]
    raise TypeError(f"Forgotten case: {type(value)}")


def clean_string(string: str, casefold: bool = True) -> str:
    """replaces special characters with underscores and cleans capizalization"""
    string = re.sub(r"[^A-Za-z0-9_]", "_", string)
    if casefold:
        string = string.casefold()
    return string


def clean_dataframe(
    df: pd.DataFrame, clean_columns: bool = True, casefold: bool = True
) -> pd.DataFrame:
    """applies string cleaning to all column names and row cleaning to all entries in the dataframe"""
    clean_df = df.map(clean_value)
    if clean_columns:
        col_map = {
            col_str: clean_string(col_str, casefold=casefold)
            for col_str in clean_df.columns
        }
        clean_df = clean_df.rename(columns=col_map)
    return clean_df


def clean_object_changes_df(df: pd.DataFrame, casefold: bool = False) -> pd.DataFrame:
    """Clean attribute field names and values in an object-changes DataFrame.

    ``ocel:*`` column names are never renamed so downstream code keeps working.
    ``ocel:oid`` values are cleaned via :func:`clean_string` to stay consistent
    with how object IDs are cleaned in ``_objects`` by :func:`clean_oced_data`.
    ``ocel:field`` values are cleaned to match the renamed attribute columns.
    Attribute columns (all non-``ocel:*`` columns) are renamed and their string
    cell values are cleaned.  Timestamps are preserved unchanged.
    """
    df = df.copy()

    # Object IDs must match the cleaned IDs asserted by instantiate_object_data()
    if "ocel:oid" in df.columns:
        df["ocel:oid"] = df["ocel:oid"].apply(
            lambda v: clean_string(str(v), casefold=casefold) if isinstance(v, str) else v
        )

    system_cols = {c for c in df.columns if c.startswith("ocel:")}
    attr_cols = [c for c in df.columns if c not in system_cols]

    if not attr_cols:
        return df

    rename_map = {c: clean_string(c, casefold=casefold) for c in attr_cols}
    df = df.rename(columns=rename_map)

    if "ocel:field" in df.columns:
        df["ocel:field"] = df["ocel:field"].apply(
            lambda v: clean_string(str(v), casefold=casefold) if isinstance(v, str) else v
        )

    for new_col in rename_map.values():
        df[new_col] = df[new_col].apply(
            lambda v: clean_string(str(v), casefold=casefold)
            if isinstance(v, str) and not is_timestamp(str(v))
            else v
        )

    return df


def clean_column_values(log, to_clean):
    """gets a dictionary of columns and functions, applies the functions to the values in the corresponding columns"""
    for column, function in to_clean.items():
        if column in log.columns:
            log[column] = log[column].apply(function)
        else:
            raise ValueError(f"Column '{column}' not found in log DataFrame")
    return log


# Helper


def is_timestamp(string: str) -> bool:
    """Return True if string is a strict ISO 8601 date or datetime, otherwise False."""
    return bool(_ISO8601_RE.match(string))
