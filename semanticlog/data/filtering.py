
import pandas as pd
from datetime import datetime
from dateutil import tz
import pm4py as pm
from pm4py.algo.filtering.log.variants import variants_filter
import time
import random


def filter_to_timeintervall(log, mintime, maxtime, time_col="time:timestamp"):
    """takes a log, time intervall defined via a mintime and maxtime in unixtime seconds and returns the log filltered to this interval"""
    if log[time_col].dtype == 'datetime64[ns, UTC]':
        maxtime = pd.to_datetime(maxtime, unit="s", utc=True)
        mintime = pd.to_datetime(mintime, unit="s", utc=True)
    else:
        maxtime = pd.to_datetime(maxtime, unit="s", utc=False)
        mintime = pd.to_datetime(mintime, unit="s", utc=False)
    
    traces_outside_intervall = log[
        ((log["time:timestamp"] > maxtime) | (log["time:timestamp"] < mintime))
    ]
    log = log.drop(traces_outside_intervall.index)
    return log

def filter_remove_months(
    log, months, time_col="time:timestamp", case_col="case:concept:name"
):
    """given a log and list of ints representing a month, filters the by removing any cases which have events in the given months"""
    events_in_months_to_remove = log[log[time_col].dt.month.isin(months)]
    case_ids_to_remove = events_in_months_to_remove[case_col].unique()
    log = log[~log[case_col].isin(case_ids_to_remove)]
    return log

def filter_variants_by_minimum_occurence(log, minimum_occurences):
    """given a log and minimum number of occurences that variants should have, filter all variants with less occurences"""
    variants = variants_filter.get_variants(log)
    filtered_variants = {
        variant: cases
        for variant, cases in variants.items()
        if len(cases) >= minimum_occurences
    }
    log = variants_filter.apply(log, filtered_variants)
    return pm.convert_to_dataframe(log)

def filter_variants_by_maximum_repetition(
    log, maximum_repetition, case_col="case:concept:name"
):
    """given a log and maximum number of allowed repetitions for a variant, filters out traces for each variant s.t. it is
    within the maximum number of allowed repetitions"""
    variants = variants_filter.get_variants(log)

    cases_to_keep = []
    for variant, cases in variants.items():
        if len(cases) > maximum_repetition:
            variant_cases_to_keep = random.sample(cases, maximum_repetition)
        else:
            variant_cases_to_keep = cases
        cases_to_keep = cases_to_keep + variant_cases_to_keep

    case_col_dict_structure = case_col.removeprefix("case:")

    case_ids_to_keep = [
        x.attributes[case_col_dict_structure] for x in cases_to_keep
    ]

    log = log[log[case_col].isin(case_ids_to_keep)]

    return pm.convert_to_dataframe(log)

