"""Turning numeric columns into a handful of labelled bins.

A named individual per distinct number is rarely useful to reason over, so a
numeric attribute can be reduced to a few ordered classes instead. Two
strategies are available: equal-frequency bins, and bins cut from a fitted
distribution so that the labels mean the same thing across differently shaped
columns.

Fitting and applying are separate. `fit_equal_bins` and `fit_dist_bins` look
at every value an attribute takes and return a `Binning`, which then labels
any column of that attribute. Splitting them is what lets one attribute
carried by several tables be cut at one set of boundaries. A column with too
few distinct values to bin has no `Binning`, and the caller decides what to do
with it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

_DISTRIBUTIONS = ("norm", "lognorm", "expon", "gamma")

_FIT_FAILURES = (stats.FitError, ValueError, FloatingPointError, RuntimeError)
_DIST_NEEDS_POSITIVE = frozenset({"lognorm", "gamma", "expon"})
_Z_EDGE_LOW = -3
_Z_EDGE_HIGH = 3

DEFAULT_BINS = 5
DEFAULT_BIN_LABELS = ("very_low", "low", "average", "high", "very_high")

DEFAULT_MIN_VALUES = 10


def _check_labels(bins: int, labels) -> None:
    """Raise unless there is exactly one label per bin."""
    if len(labels) != bins:
        raise ValueError(
            f"bins={bins} needs exactly {bins} labels, got {len(labels)}: "
            f"{list(labels)}. Values in the unlabelled bins would be silently "
            "dropped."
        )


@dataclass(frozen=True)
class Binning:
    """The cut one attribute is binned at, fitted once and applied many times.

    Attributes:
        labels: Label per bin, in order from low to high.
        edges: The bin boundaries, in the space `apply` cuts in. Equal-frequency
            bins cut the values themselves, distribution bins cut z-scores.
        dist: Name of the fitted distribution, or None for equal frequency.
        params: The parameters that distribution was fitted with.
    """

    labels: tuple[str, ...]
    edges: tuple[float, ...]
    dist: str | None = None
    params: tuple[Any, ...] | None = None

    def apply(self, values: pd.Series) -> pd.Series:
        """Return the bin label of each value.

        Args:
            values: Numeric column of the attribute this was fitted for.

        Returns:
            One label per row, aligned with the input. A value outside every
            bin is missing.
        """
        cut_on = values if self.dist is None else self._z_scores(values)
        return pd.cut(cut_on, bins=list(self.edges), labels=list(self.labels))

    def _z_scores(self, values: pd.Series) -> pd.Series:
        """Map values onto a standard normal scale through the fitted shape."""
        params = self.params or ()
        if self.dist == "norm":
            mu, sigma = params
            return (values - mu) / sigma
        dist = getattr(stats, str(self.dist))
        return pd.Series(stats.norm.ppf(dist.cdf(values, *params)), index=values.index)


def fit_equal_bins(
    values: pd.Series,
    bins: int = DEFAULT_BINS,
    labels=DEFAULT_BIN_LABELS,
) -> Binning | None:
    """Fit bins that each hold roughly as many values.

    Repeated values can make fewer bins than requested, in which case the
    labels are taken symmetrically around the centre of the list.

    Args:
        values: Every value the attribute takes.
        bins: Number of bins.
        labels: One label per bin, in order from low to high.

    Returns:
        The fitted cut, or None when there are fewer than two distinct values
        to separate.

    Raises:
        ValueError: If the number of labels does not match the number of bins.
    """
    _check_labels(bins, labels)
    clean = values.dropna()
    if clean.nunique() < 2:
        return None
    try:
        edges = pd.qcut(clean, q=min(bins, clean.nunique()), duplicates="drop")
    except ValueError:
        return None
    boundaries = _unbounded(sorted({edge.left for edge in edges.cat.categories}))
    surviving = len(boundaries) - 1
    if surviving < 2:
        return None
    return Binning(
        labels=tuple(_select_labels(list(labels), surviving)), edges=boundaries
    )


def fit_dist_bins(
    values: pd.Series,
    bins: int = DEFAULT_BINS,
    labels=DEFAULT_BIN_LABELS,
    min_values: int = DEFAULT_MIN_VALUES,
) -> Binning | None:
    """Fit bins by cutting the distribution that fits the attribute best.

    Values are mapped onto a standard normal scale first, so the same bin
    label means the same distance from the centre whatever the shape of the
    original column. The candidate distributions are the normal, log-normal,
    exponential and gamma. When none of them fits, this falls back to equal
    frequency.

    Args:
        values: Every value the attribute takes.
        bins: Number of bins.
        labels: One label per bin, in order from low to high.
        min_values: Cardinality below which the attribute is not binned.

    Returns:
        The fitted cut, or None when the attribute has too few distinct values
        to bin.

    Raises:
        ValueError: If the number of labels does not match the number of bins.
    """
    _check_labels(bins, labels)
    if values.nunique() <= max(bins, min_values):
        return None
    dist_name, params = get_best_fit_distribution(values)[1:3]
    if dist_name not in _DISTRIBUTIONS:
        return fit_equal_bins(values, bins=bins, labels=labels)

    edges = list(np.linspace(_Z_EDGE_LOW, _Z_EDGE_HIGH, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return Binning(
        labels=tuple(labels), edges=tuple(edges), dist=dist_name, params=tuple(params)
    )


def _unbounded(left_edges: list[float]) -> tuple[float, ...]:
    """Return cut boundaries reaching to infinity on both sides."""
    return (-np.inf, *left_edges[1:], np.inf)


def _fit_kwargs(dist_name: str) -> dict[str, float]:
    """Return the constraints one distribution is always fitted under.

    A distribution with positive support is pinned at the origin. Left free,
    its location parameter settles on the smallest value in the column, which
    puts that value at the very bottom of the fitted distribution and leaves
    it outside every bin. The same constraint is used when the candidates are
    ranked, so the fit that wins is the fit the bins are cut from.
    """
    return {"floc": 0} if dist_name in _DIST_NEEDS_POSITIVE else {}


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
                params = dist.fit(x, **_fit_kwargs(dist_name))
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
    """Take n labels symmetrically around the centre of the list.

    An odd count keeps the centre label and expands outwards from it. An even
    count has no centre bin, so the centre-most labels are the ones dropped:
    two bins out of five labels become low and high, and four become the two
    low and the two high.
    """
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
