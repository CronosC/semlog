import pandas as pd
import re
from dateutil.parser import parse
from dateutil.parser import parse
from semanticlog.data.oced import OCEDStore


def clean_oced_data(data: OCEDStore) -> OCEDStore:
    """cleans all keys and values in all dicts and dataframes of a OCEDStore"""
    data._events = clean_value(data._events, casefold=False)
    data._objects = clean_value(data._objects, casefold=False)
    data._relations = clean_value(data._relations, casefold=False)
    return data


def clean_value(
    value: dict | str | int | float | bool | pd.DataFrame, casefold: bool = True
) -> str | int | float | bool | pd.DataFrame:
    """applies string cleaning to some value, handling different datatypes appropriately"""
    if isinstance(value, dict):
        return {
            clean_string(key, casefold=casefold): clean_value(
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
    clean_df = df.applymap(clean_value)
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
