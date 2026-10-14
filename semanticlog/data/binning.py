"""Binning of numeric attributes into labeled classes.

`fit_equal_bins` cuts at equal frequency and `fit_dist_bins` cuts z-scores
under a fitted distribution. Each takes every value of an attribute and
returns a `Binning` that labels any column of it, or None when the attribute
has too few distinct values.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from semanticlog.core.errors import ConfigError

logger = logging.getLogger(__name__)

_DISTRIBUTIONS = ("norm", "lognorm", "expon", "gamma")

_FIT_FAILURES = (stats.FitError, ValueError, FloatingPointError, RuntimeError)
_DIST_NEEDS_POSITIVE = frozenset({"lognorm", "gamma", "expon"})
_Z_EDGE_LOW = -3
_Z_EDGE_HIGH = 3

DEFAULT_BINS = 5
DEFAULT_BIN_LABELS = ("very_low", "low", "average", "high", "very_high")

DEFAULT_MIN_VALUES = 10
DEFAULT_FIT_P = 0.05


def check_bins(bins: int, labels: Sequence[str]) -> None:
    """Raise `ConfigError` for under 2 bins, a wrong label count or a repeated label."""
    if bins < 2:
        raise ConfigError(f"bins must be at least 2, got {bins}.")
    if len(labels) != bins:
        raise ConfigError(
            f"bins={bins} needs exactly {bins} labels, got {len(labels)}: "
            f"{list(labels)}."
        )
    repeated = sorted({label for label, count in Counter(labels).items() if count > 1})
    if repeated:
        raise ConfigError(
            f"Bin labels must all differ, got {repeated} more than once: "
            f"{list(labels)}."
        )


@dataclass(frozen=True)
class Binning:
    """The bin edges and labels of one attribute.

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
        return pd.cut(
            cut_on,
            bins=list(self.edges),
            labels=list(self.labels),
            include_lowest=True,
        )

    def _z_scores(self, values: pd.Series) -> pd.Series:
        """Return the z-score of each value under the fitted distribution."""
        params = self.params or ()
        if self.dist == "norm":
            mu, sigma = params
            return (values - mu) / sigma
        dist = getattr(stats, str(self.dist))
        return pd.Series(stats.norm.ppf(dist.cdf(values, *params)), index=values.index)


def fit_equal_bins(
    values: pd.Series,
    bins: int = DEFAULT_BINS,
    labels: Sequence[str] = DEFAULT_BIN_LABELS,
    min_values: int = DEFAULT_MIN_VALUES,
) -> Binning | None:
    """Fit equal-frequency bins.

    Repeated values can make fewer bins than requested, in which case the
    labels are taken symmetrically around the center of the list.

    Args:
        values: Every value the attribute takes.
        labels: One label per bin, in order from low to high.
        min_values: Cardinality below which the attribute is not binned.

    Returns:
        The fitted cut, or None when the attribute has too few distinct values
        to bin, or fewer than two bins survive `qcut`.

    Raises:
        ConfigError: If `bins` is below two, the labels do not match it, or a
            label repeats.
    """
    check_bins(bins, labels)
    clean = values.dropna()
    if clean.nunique() <= max(bins, min_values):
        return None
    try:
        edges = pd.qcut(clean, q=bins, duplicates="drop")
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
    labels: Sequence[str] = DEFAULT_BIN_LABELS,
    min_values: int = DEFAULT_MIN_VALUES,
    p: float = DEFAULT_FIT_P,
) -> Binning | None:
    """Fit bins by cutting the distribution that fits the attribute best.

    Values are mapped to z-scores under the best-fitting distribution, so one
    label means one z-score range for every column. The candidate
    distributions are the normal, log-normal, exponential and gamma. When none
    of them fits closely enough, this falls back to equal frequency.

    Args:
        values: Every value the attribute takes.
        labels: One label per bin, in order from low to high.
        min_values: Cardinality below which the attribute is not binned.
        p: Minimum Kolmogorov-Smirnov p-value the winning fit must reach.

    Returns:
        The fitted cut, or None when the attribute has too few distinct values
        to bin. The equal-frequency fallback can also return None.

    Raises:
        ConfigError: If `bins` is below two, the labels do not match it, or a
            label repeats.
    """
    check_bins(bins, labels)
    if values.nunique() <= max(bins, min_values):
        return None
    fit = get_best_fit_distribution(values, p=p)
    if fit is None:
        return fit_equal_bins(values, bins=bins, labels=labels, min_values=min_values)

    edges = list(np.linspace(_Z_EDGE_LOW, _Z_EDGE_HIGH, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return Binning(
        labels=tuple(labels), edges=tuple(edges), dist=fit.name, params=fit.params
    )


def _unbounded(left_edges: list[float]) -> tuple[float, ...]:
    """Return cut boundaries reaching to infinity on both sides."""
    return (-np.inf, *left_edges[1:], np.inf)


def _fit_kwargs(dist_name: str) -> dict[str, float]:
    """Return the fixed parameters for one distribution."""
    # A free location parameter fits to the column minimum, which then falls
    # outside every bin.
    return {"floc": 0} if dist_name in _DIST_NEEDS_POSITIVE else {}


@dataclass(frozen=True)
class DistributionFit:
    """The winning fit of `get_best_fit_distribution`.

    Attributes:
        name: Name of the scipy distribution.
        params: The parameters it was fitted with.
        aic: AIC over the free parameters.
        p_value: Kolmogorov-Smirnov p-value of the fit.
    """

    name: str
    params: tuple[Any, ...]
    aic: float
    p_value: float


def get_best_fit_distribution(
    x: pd.Series,
    candidates: Sequence[str] | None = None,
    p: float = DEFAULT_FIT_P,
) -> DistributionFit | None:
    """Find which candidate distribution fits a column best.

    Fits are compared by AIC over the free parameters.

    Args:
        x: Values to fit. Missing values are dropped.
        candidates: scipy distribution names, defaulting to normal,
            log-normal, exponential and gamma.
        p: Minimum Kolmogorov-Smirnov p-value the winner must reach. Zero
            accepts any fit.

    Returns:
        The winning fit, or None when nothing fit or the winner is below `p`.
    """
    sample = x.dropna().to_numpy()
    if candidates is None:
        candidates = list(_DISTRIBUTIONS)
    best = None
    for dist_name in candidates:
        dist = getattr(stats, dist_name)
        try:
            if len(sample) == 0 or (
                (sample <= 0).any() and dist_name in _DIST_NEEDS_POSITIVE
            ):
                continue
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                constraints = _fit_kwargs(dist_name)
                params = dist.fit(sample, **constraints)
                loglik = np.sum(dist.logpdf(sample, *params))
                aic = 2 * (len(params) - len(constraints)) - 2 * loglik
                logger.debug("%s fits with aic %s", dist_name, aic)
                if not np.isfinite(aic):
                    logger.debug("%s has no finite aic, so it is skipped", dist_name)
                    continue
                if best is None or aic < best[0]:
                    best = (aic, dist_name, params)
        except _FIT_FAILURES as exc:
            logger.debug("Distribution fit failed for %s: %s", dist_name, exc)
            continue
    if best is None:
        return None
    aic, dist_name, params = best
    p_val = _goodness_of_fit(sample, dist_name, params)
    logger.debug("%s wins with aic %s and p %s", dist_name, aic, p_val)
    if np.isnan(p_val) or p_val < p:
        return None
    return DistributionFit(dist_name, tuple(params), float(aic), p_val)


def _goodness_of_fit(
    sample: np.ndarray, dist_name: str, params: Sequence[float]
) -> float:
    """Return the Kolmogorov-Smirnov p-value of one fit, NaN if it fails."""
    # The parameters come from the same sample, so the p-value is biased high.
    # It only filters out a distribution far from the column.
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            cdf = getattr(stats, dist_name)(*params).cdf
            return float(stats.kstest(sample, cdf)[1])
    except _FIT_FAILURES as exc:
        logger.debug("Goodness of fit failed for %s: %s", dist_name, exc)
        return float("nan")


def _select_labels(labels: Sequence[str], n: int) -> list[str]:
    """Take n labels symmetrically around the center of the list.

    Two of five labels become low and high, and four become the two low and
    the two high.
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
