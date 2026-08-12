"""Turning numeric columns into a handful of labelled bins.

A named individual per distinct number is rarely useful to reason over, so a
numeric attribute can be reduced to a few ordered classes instead. Two
strategies are available: equal-frequency bins, and bins cut from a fitted
distribution so that the labels mean the same thing across differently shaped
columns. Columns with too few distinct values are left as text, because
binning them would invent an order that the data does not support.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

_DISTRIBUTIONS = ("norm", "lognorm", "expon", "gamma")

_FIT_FAILURES = (stats.FitError, ValueError, FloatingPointError, RuntimeError)
_DIST_NEEDS_POSITIVE = frozenset({"lognorm", "gamma"})
_Z_EDGE_LOW = -3
_Z_EDGE_HIGH = 3

DEFAULT_BINS = 5
DEFAULT_BIN_LABELS = ("very_low", "low", "average", "high", "very_high")


def _check_labels(bins: int, labels) -> None:
    """Raise unless there is exactly one label per bin."""
    if len(labels) != bins:
        raise ValueError(
            f"bins={bins} needs exactly {bins} labels, got {len(labels)}: "
            f"{list(labels)}. Values in the unlabelled bins would be silently "
            "dropped."
        )


def dist_bin_series(
    series,
    bins=DEFAULT_BINS,
    labels=DEFAULT_BIN_LABELS,
    min_values=10,
):
    """Bin a numeric column using the distribution that fits it best.

    The candidate distributions are the normal, log-normal, exponential and
    gamma. When none of them fits, this falls back to equal-frequency bins.

    Args:
        series: Numeric column to bin.
        bins: Number of bins.
        labels: One label per bin, in order from low to high.
        min_values: Cardinality below which the column is left as text
            instead of being binned.

    Returns:
        A column of bin labels, or the values as text when the column has too
        few distinct values to bin.

    Raises:
        ValueError: If the number of labels does not match the number of bins.
    """
    _check_labels(bins, labels)
    if series.nunique() <= max(bins, min_values):
        logger.warning(
            f"dist_bin_series: low cardinality ({series.nunique()} unique values). "
            "Converting numeric column to string labels instead of binning."
        )
        return series.astype(str)
    best_fit = get_best_fit_distribution(series)[1]
    if best_fit in _DISTRIBUTIONS:
        return _cut_by_distribution(series, best_fit, bins=bins, labels=labels)
    return equal_bin_series(series, bins=bins, labels=labels)


def get_best_fit_distribution(x, candidates=None, p=0.0):
    """Find which candidate distribution fits a column best.

    Fits are compared by AIC, and a distribution that needs positive values is
    skipped for a column that has any that are not. A failed fit is skipped
    without raising, since a column may fit nothing at all.

    Args:
        x: Values to fit. Missing values are dropped.
        candidates: Distribution names to try, defaulting to the four this
            module supports.
        p: Minimum Kolmogorov-Smirnov p-value the winner must reach. Zero
            accepts any fit.

    Returns:
        A tuple of AIC, distribution name, fitted parameters and p-value.
        All four are None when nothing fit.
    """
    x = x.dropna().to_numpy()
    if candidates is None:
        candidates = list(_DISTRIBUTIONS)
    best = None
    for dist_name in candidates:
        dist = getattr(stats, dist_name)
        try:
            if len(x) == 0 or ((x <= 0).any() and dist_name in _DIST_NEEDS_POSITIVE):
                continue
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                params = dist.fit(x)
                loglik = np.sum(dist.logpdf(x, *params))
                aic = 2 * len(params) - 2 * loglik
                logger.debug(f"{dist_name} fits with aic {aic}")
                _, p_val = stats.kstest(x, dist_name, args=params)
                if best is None or aic < best[0]:
                    best = (aic, dist_name, params, p_val)
        except _FIT_FAILURES as exc:
            logger.debug(f"Distribution fit failed for {dist_name}: {exc}")
            continue
    if best is None:
        return (None, None, None, None)
    if best[3] < p:
        return (None, None, None, None)
    return best


def _select_labels(labels: list, n: int) -> list:
    """Take n labels from the middle outwards, for when fewer bins survive."""
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


def equal_bin_series(series, bins=DEFAULT_BINS, labels=DEFAULT_BIN_LABELS):
    """Bin a numeric column so that each bin holds roughly as many values.

    Repeated values can make fewer bins than requested, in which case labels
    are taken from the middle of the list outwards so that the extremes are
    not used for bins that are no longer extreme.

    Args:
        series: Numeric column to bin.
        bins: Number of bins.
        labels: One label per bin, in order from low to high.

    Returns:
        A column of bin labels, or the values as text when there are fewer
        than two distinct values or the cut fails.

    Raises:
        ValueError: If the number of labels does not match the number of bins.
    """
    _check_labels(bins, labels)
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


def _cut_by_distribution(
    series, dist_name, bins=DEFAULT_BINS, labels=DEFAULT_BIN_LABELS
):
    """Bin a column by cutting the fitted distribution at fixed z-scores.

    Values are mapped onto a standard normal scale first, so the same bin
    label means the same distance from the centre whatever the shape of the
    original column. The outermost edges are unbounded, so no value falls
    outside the bins.
    """
    _check_labels(bins, labels)
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
