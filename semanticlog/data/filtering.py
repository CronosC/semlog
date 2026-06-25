"""
Classical event-log filtering utilities.

These functions operate on flat pm4py-format DataFrames (the kind returned by
:meth:`OCEDStore.get_log` or loaded directly via pm4py), NOT on
:class:`~semanticlog.data.oced.OCEDStore` objects.  They are pre-processing
helpers intended for use before or after flattening OCED data to a classical log.
Column names are assumed to follow pm4py conventions (``"time:timestamp"``,
``"case:concept:name"``).
"""

import random

import pandas as pd
import pm4py as pm
from pm4py.algo.filtering.log.variants import variants_filter


def filter_to_time_interval(log, mintime, maxtime, time_col="time:timestamp"):
    """Filter a log to events whose timestamp falls within [mintime, maxtime].

    Args:
        log: pm4py-format DataFrame.
        mintime: Lower bound as Unix seconds (inclusive).
        maxtime: Upper bound as Unix seconds (inclusive).
        time_col: Name of the timestamp column.

    Returns:
        Filtered DataFrame with out-of-window events removed.
    """
    is_tz_aware = (
        hasattr(log[time_col].dtype, "tz") and log[time_col].dtype.tz is not None
    )
    maxtime = pd.to_datetime(maxtime, unit="s", utc=is_tz_aware)
    mintime = pd.to_datetime(mintime, unit="s", utc=is_tz_aware)

    traces_outside_interval = log[
        ((log[time_col] > maxtime) | (log[time_col] < mintime))
    ]
    log = log.drop(traces_outside_interval.index)
    return log


def filter_remove_months(
    log, months, time_col="time:timestamp", case_col="case:concept:name"
):
    """Remove all cases that contain any event in the specified months.

    Args:
        log: pm4py-format DataFrame.
        months: List of integer month numbers (1–12) to exclude.
        time_col: Name of the timestamp column.
        case_col: Name of the case-ID column.

    Returns:
        Filtered DataFrame with affected cases removed entirely.
    """
    events_in_months_to_remove = log[log[time_col].dt.month.isin(months)]
    case_ids_to_remove = events_in_months_to_remove[case_col].unique()
    log = log[~log[case_col].isin(case_ids_to_remove)]
    return log


def filter_variants_by_minimum_occurrence(log, minimum_occurrences):
    """Keep only cases whose variant appears at least ``minimum_occurrences`` times.

    Args:
        log: pm4py-format DataFrame.
        minimum_occurrences: Minimum number of cases a variant must have to be kept.

    Returns:
        Filtered DataFrame as a pm4py DataFrame.
    """
    variants = variants_filter.get_variants(log)
    filtered_variants = {
        variant: cases
        for variant, cases in variants.items()
        if len(cases) >= minimum_occurrences
    }
    log = variants_filter.apply(log, filtered_variants)
    return pm.convert_to_dataframe(log)


def filter_variants_by_maximum_repetition(
    log, maximum_repetition, case_col="case:concept:name", seed=None
):
    """Downsample over-represented variants to at most ``maximum_repetition`` cases each.

    For each variant with more cases than ``maximum_repetition``, a random
    subset of the allowed size is kept; all others are removed.

    Args:
        log: pm4py-format DataFrame.
        maximum_repetition: Maximum number of cases allowed per variant.
        case_col: Name of the case-ID column.
        seed: Optional random seed for reproducible sampling.

    Returns:
        Filtered DataFrame as a pm4py DataFrame.
    """
    variants = variants_filter.get_variants(log)

    rng = random.Random(seed)  # noqa: S311
    cases_to_keep = []
    for variant, cases in variants.items():
        if len(cases) > maximum_repetition:
            variant_cases_to_keep = rng.sample(cases, maximum_repetition)
        else:
            variant_cases_to_keep = cases
        cases_to_keep.extend(variant_cases_to_keep)

    case_col_dict_structure = case_col.removeprefix("case:")

    case_ids_to_keep = [x.attributes[case_col_dict_structure] for x in cases_to_keep]

    log = log[log[case_col].isin(case_ids_to_keep)]

    return pm.convert_to_dataframe(log)
