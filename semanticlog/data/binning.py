import logging

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


def _bin_numeric_columns(df: pd.DataFrame, exclude_cols: set[str]) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and col not in exclude_cols:
            df[col] = bin_series(df[col])
    return df


def bin_attributes(data: OCEDStore) -> OCEDStore:
    """Bin all numeric attributes in an OCEDStore using distribution-fitted or equal binning."""
    for object_type in data.get_object_types():
        obj = data.get_object(object_type)
        data.set_object_data(object_type, _bin_numeric_columns(obj.data, {obj.id}))

    for activity in data.get_activities():
        table = data.get_events(activity)
        data.set_event_data(
            activity, _bin_numeric_columns(table.data, {table.id, table.time})
        )
    return data


_DIST_NEEDS_POSITIVE = {"lognorm", "gamma"}
_Z_EDGE_LOW = -3
_Z_EDGE_HIGH = 3
_DEFAULT_LABELS_5 = ["very_low", "low", "average", "high", "very_high"]
_DEFAULT_LABELS_3 = ["Low", "Average", "High"]


def bin_series(
    series,
    bins=5,
    labels=_DEFAULT_LABELS_5,
    min_values=10,
):
    """Bin a numeric series using distribution fitting, falling back to equal-frequency binning.

    For low-cardinality series (unique values ≤ max(bins, min_values)), converts to strings directly.
    """
    if series.nunique() <= max(bins, min_values):
        logging.warning(
            f"bin_series: low cardinality ({series.nunique()} unique values) — "
            "converting numeric column to string labels instead of binning."
        )
        return series.astype(str)
    best_fit = get_best_fit_distribution(series)[1]
    if best_fit in ("norm", "lognorm", "expon", "gamma"):
        return _dist_bin_series(series, best_fit, bins=bins, labels=labels)
    return equal_bin_series(series, bins=bins, labels=labels)


def get_best_fit_distribution(x, candidates=None, p=0.0):
    """Fit candidate distributions to a Series and return the best by AIC.

    Args:
        x: Numeric pandas Series to fit.
        candidates: Distribution names to try (default: ``["norm", "lognorm", "expon", "gamma"]``).
        p: Minimum KS-test p-value required to accept a fit. Distributions below
           this threshold are rejected. Default ``0.0`` accepts any fit.

    Returns:
        Tuple ``(aic, name, params, pval)`` for the best-fitting distribution,
        or ``(None, None, None, None)`` if the Series is empty or no distribution
        passes the p-value gate.
    """
    x = x.dropna().to_numpy()
    if candidates is None:
        candidates = ["norm", "lognorm", "expon", "gamma"]
    best = None
    for dist_name in candidates:
        dist = getattr(stats, dist_name)
        try:
            if len(x) == 0 or ((x <= 0).any() and dist_name in _DIST_NEEDS_POSITIVE):
                continue
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                params = dist.fit(x)
                loglik = np.sum(dist.logpdf(x, *params))
                aic = 2 * len(params) - 2 * loglik  # Akaike Information Criterion
                logging.debug(f"{dist_name} fits with aic {aic}")
                ks_stat, p_val = stats.kstest(x, dist_name, args=params)
                if best is None or aic < best[0]:
                    best = (aic, dist_name, params, p_val)
        except Exception as e:
            logging.debug(f"Distribution fit failed for {dist_name}: {e}")
            continue
    if best is None:
        return (None, None, None, None)
    if best[3] < p:
        best = (None, None, None, None)
    return best  # (aic, name, params, pval)


def _select_labels(labels: list, n: int) -> list:
    N = len(labels)
    if n >= N:
        return list(labels)
    center = N // 2
    k = n // 2
    if n % 2 == 1:
        return list(labels[center - k : center + k + 1])
    left = list(labels[center - k : center])
    if N % 2 == 1:
        right = list(labels[center + 1 : center + k + 1])
    else:
        right = list(labels[center : center + k])
    return left + right


def equal_bin_series(series, bins=3, labels=_DEFAULT_LABELS_3):
    """Bin a series into equal-frequency bins using pd.qcut."""
    n_unique = series.dropna().nunique()
    effective_bins = min(bins, n_unique)
    if effective_bins < 2:
        return series.astype(str)
    try:
        binned = pd.qcut(series, q=effective_bins, labels=False, duplicates="drop")
        actual_bins = int(binned.dropna().max()) + 1
        label_map = dict(enumerate(_select_labels(labels, actual_bins)))
        return binned.map(label_map)
    except ValueError:
        return series.astype(str)


def _dist_bin_series(series, dist_name, bins=3, labels=_DEFAULT_LABELS_3):
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
