import logging
from typing import cast

import numpy as np
import pandas as pd
from pm4py.statistics.variants.pandas.get import get_variants_count
from scipy import stats

from semanticlog.data.oced import OCEDStore


def get_weighted_variant_mapping(
    log,
    case_column="case:concept:name",
    activity_column="concept:name",
    time_column="time:timestamp",
    event_column="event",
):
    """Gets variant weights from a classical event log"""
    log = log.sort_values(by=[case_column, time_column, activity_column, event_column])
    variant_weights = get_variants_count(log)

    return variant_weights


def quantize(data: OCEDStore) -> OCEDStore:
    for object_type in data.get_object_types():
        object_dict = data.get_object(object_type)
        df = cast(pd.DataFrame, object_dict["data"])
        id_col = cast(str, object_dict["id"])
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if col not in [id_col]:
                    df[col] = quantize_series(df[col])
        object_dict["data"] = df
        data._objects[object_type] = object_dict

    for activity in data.get_activities():
        activity_dict = data.get_event(activity)
        df = cast(pd.DataFrame, activity_dict["data"])
        time_col = cast(str, activity_dict["time"])
        id_col = cast(str, activity_dict["id"])
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if col not in [time_col, id_col]:
                    df[col] = quantize_series(df[col])
        activity_dict["data"] = df
        data._events[activity] = activity_dict
    return data


_DIST_NEEDS_POSITIVE = {"lognorm", "gamma"}
_Z_EDGE_LOW = -3
_Z_EDGE_HIGH = 3


def quantize_series(
    series,
    bins=5,
    labels=["very_low", "low", "average", "high", "very_high"],
    min_values=10,
):
    if series.nunique() <= max(bins, min_values):
        logging.warning(
            f"quantize_series: low cardinality ({series.nunique()} unique values) — "
            "converting numeric column to string labels instead of binning."
        )
        return series.astype(str)
    best_fit = get_best_fit_distribution(series)[1]
    if best_fit in ("norm", "lognorm", "expon", "gamma"):
        return _distribution_quantize(series, best_fit, bins=bins, labels=labels)
    return simple_quantize(series, bins=bins, labels=labels)


def get_best_fit_distribution(x, candidates=None, p=0.0):
    x = x.dropna().to_numpy()
    if candidates is None:
        candidates = ["norm", "lognorm", "expon", "gamma"]
    best = None
    for dist_name in candidates:
        dist = getattr(stats, dist_name)
        try:
            # fit with safe warnings off
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                if not (
                    len(x) == 0
                    or ((x <= 0).any() and dist_name in ["lognorm", "gamma"])
                ):
                    params = dist.fit(x)
                    # compute log-likelihood
                    loglik = np.sum(dist.logpdf(x, *params))
                    aic = 2 * len(params) - 2 * loglik  # Akaike Information Criterion
                    logging.debug(f"{dist_name} fits with aic {aic}")
                    # compute p-value via KS test
                    ks_stat, p_val = stats.kstest(x, dist_name, args=params)

                    if best is None or aic < best[0]:
                        best = (aic, dist_name, params, p_val)
        except Exception:
            continue
    if best is None:
        return (None, None, None, None)
    if best[3] < p:
        best = (None, None, None, None)
    return best  # (aic, name, params, pval)


def simple_quantize(series, bins=3, labels=["Low", "Average", "High"]):
    n_unique = series.dropna().nunique()
    effective_bins = min(bins, n_unique)
    if effective_bins < 2:
        return series.astype(str)
    try:
        binned = pd.qcut(series, q=effective_bins, labels=False, duplicates="drop")
        actual_bins = int(binned.dropna().max()) + 1
        label_map = dict(enumerate(labels[:actual_bins]))
        return binned.map(label_map)
    except ValueError:
        return series.astype(str)


def _distribution_quantize(
    series, dist_name, bins=3, labels=["Low", "Average", "High"]
):
    """Fit a named scipy distribution and bin the series using normal-equivalent deviates."""
    x = series.dropna()
    if len(x) == 0:
        return pd.Series([np.nan] * len(series), index=series.index)
    if dist_name in _DIST_NEEDS_POSITIVE and (x <= 0).all():
        return pd.Series([np.nan] * len(series), index=series.index)

    dist = getattr(stats, dist_name)
    fit_kwargs = {"floc": 0} if dist_name in _DIST_NEEDS_POSITIVE else {}
    params = dist.fit(x, **fit_kwargs)

    if dist_name == "norm":
        mu, sigma = params
        z = (series - mu) / sigma
    else:
        cdf = dist.cdf(series, *params)
        z = stats.norm.ppf(cdf)

    z_edges = np.linspace(_Z_EDGE_LOW, _Z_EDGE_HIGH, bins + 1)
    z_edges[0], z_edges[-1] = -np.inf, np.inf
    return pd.cut(z, bins=z_edges, labels=labels)


# Kept as thin wrappers for backwards compatibility
def gaussian_quantize(series, bins=3, labels=["Low", "Average", "High"]):
    return _distribution_quantize(series, "norm", bins=bins, labels=labels)


def lognorm_quantize(series, bins=3, labels=["Low", "Average", "High"]):
    return _distribution_quantize(series, "lognorm", bins=bins, labels=labels)


def expon_quantize(series, bins=3, labels=["Low", "Average", "High"]):
    return _distribution_quantize(series, "expon", bins=bins, labels=labels)


def gamma_quantize(series, bins=3, labels=["Low", "Average", "High"]):
    return _distribution_quantize(series, "gamma", bins=bins, labels=labels)
