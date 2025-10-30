import pandas as pd
import re
from dateutil.parser import parse
from dateutil.parser import parse
from semanticlog.data.oced import OCEDStore
from collections import Counter
import logging
from collections.abc import Sequence


def null_check(data: OCEDStore) -> bool:
    res = False
    for activity in data.get_activities():
        activity_dict = data.get_event(activity)
        df = activity_dict["data"]
        for col in df.columns:
            n_null = df[col].isnull().sum()
            if n_null > 0:
                print(f"{activity}[{col}]: {n_null} null values.")
                res = True
    for object_type in data.get_object_types():
        object_dict = data.get_object(object_type)
        df = object_dict["data"]
        for col in df.columns:
            n_null = df[col].isnull().sum()
            if n_null > 0:
                print(f"{object_type}[{col}]: {n_null} null values.")
                res = True

def fix_duplicate_names(data: OCEDStore) -> None:
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
    attribute_names = []

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
        attributes = list(activity_data["data"].columns.values)
        attributes.remove(id_col)
        attributes.remove(time_col)
        attribute_names = attribute_names + attributes

    # get object attribute names
    object_types = data.get_object_types()
    for object_type in object_types:
        object_data = data.get_object(object_type)
        id_col = object_data["id"]
        attributes = list(object_data["data"].columns.values)
        attributes.remove(id_col)
        attribute_names = attribute_names + attributes

    # fix attribute names that are also objects/events/props
    attributes_names_overlap = [
        a for a in attribute_names if a.casefold() in [n.casefold() for n in names]
    ]
    for attribute in attributes_names_overlap:
        replace_function = lambda x: x if x != attribute else x + "_attr"
        for activitiy in activities:
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
    replace_function = lambda x: x if x not in all_names else x + "_attr_val"
    for activitiy in activities:
        data._events[activity]["data"] = apply_function_to_value(
            data._events[activity]["data"], replace_function, rename_df_columns=False
        )
    for object_type in object_types:
        data._objects[object_type]["data"] = apply_function_to_value(
            data._objects[object_type]["data"],
            replace_function,
            rename_df_columns=False,
        )
    return data


def clean_oced_data(data: OCEDStore, casefold=False) -> OCEDStore:
    """cleans all keys and values in all dicts and dataframes of a OCEDStore"""
    logging.info("Cleaning data..")
    data._events = clean_value(data._events, casefold=casefold)
    data._objects = clean_value(data._objects, casefold=casefold)
    data._relations = clean_value(data._relations, casefold=casefold)
    data = fix_duplicate_names(data)
    return data


def apply_function_to_value(
    value: dict | str | int | float | bool | pd.DataFrame,
    function: callable,
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
    raise(Exception("Forgotten case: ", type(value)))
    return value


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
    raise(Exception("Forgotten case: ", type(value)))
    return value


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
        clean_columns = {
            col_str: clean_string(col_str, casefold=casefold)
            for col_str in clean_df.columns
        }
        clean_df = clean_df.rename(columns=clean_columns)
    return clean_df


def clean_column_values(log, to_clean):
    """gets a dictionary of columns and functions, applies the functions to the values in the corresponding columns"""
    for column, function in to_clean.items():
        if column in log.columns:
            log[column] = log[column].apply(function)
        else:
            raise ValueError(f"Column '{column}' not found in log DataFrame")
    return log


#### Helper ####


def is_timestamp(string: str) -> bool:
    """Return True if string represents a timestamp, otherwise False."""
    try:
        parse(string)
        return True
    except Exception:
        return False
