"""Binning of numeric attributes into labeled classes.

`fit_equal_bins` cuts at equal frequency. `fit_dist_bins` cuts by distance
from the typical value, in standard deviations, after a Yeo-Johnson power
transform has made the column close to normal. Each takes every value of an
attribute and returns a `Binning` that labels any column of it, or None when
the attribute has too few distinct values.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from semanticlog.core.errors import ConfigError

logger = logging.getLogger(__name__)

DEFAULT_BINS = 5
DEFAULT_BIN_LABELS = ("very_low", "low", "average", "high", "very_high")

DEFAULT_MIN_VALUES = 10

# The interquartile range of a normal distribution in standard deviations.
_IQR_PER_SD = 1.349


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
        edges: The bin boundaries in the values of the attribute, from -inf
            to inf. A bin holds the values above its lower edge up to and
            including its upper edge.
    """

    labels: tuple[str, ...]
    edges: tuple[float, ...]

    def apply(self, values: pd.Series) -> pd.Series:
        """Return the bin label of each value.

        Args:
            values: Numeric column of the attribute this was fitted for.

        Returns:
            One label per row, aligned with the input. A missing value stays
            missing.
        """
        return pd.cut(
            values,
            bins=list(self.edges),
            labels=list(self.labels),
            include_lowest=True,
        )

    def bounds(self, label: str) -> tuple[float, float]:
        """Return the lower and upper edge of one bin, either of them infinite.

        Raises:
            ValueError: If no bin has that label.
        """
        i = self.labels.index(label)
        return self.edges[i], self.edges[i + 1]


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
) -> Binning | None:
    """Fit bins by distance from the typical value, after a power transform.

    The values are shifted to start at zero, scaled, made close to normal by
    a Yeo-Johnson transform and standardized by their median and
    interquartile range. The bins are one
    standard deviation wide and centered on the median: five bins cut at
    -1.5, -0.5, 0.5 and 1.5. The cuts are mapped back to the values.

    Args:
        values: Every value the attribute takes.
        labels: One label per bin, in order from low to high.
        min_values: Cardinality below which the attribute is not binned.

    Returns:
        The fitted cut, or None when the attribute has too few distinct values
        to bin. A bin the transform cannot reach on either end is dropped with
        its label, and None is returned when fewer than two remain.

    Raises:
        ConfigError: If `bins` is below two, the labels do not match it, or a
            label repeats.
    """
    check_bins(bins, labels)
    clean = values.dropna().astype(float)
    if clean.nunique() <= max(bins, min_values):
        return None

    x = clean.to_numpy()
    # Shifted to start at zero and scaled, so the transform acts on the
    # shape of the column, not its offset or magnitude.
    offset = float(x.min())
    scale = _robust_location(x)[1]
    transformed, lmbda = stats.yeojohnson((x - offset) / scale)
    y_center, y_spread = _robust_location(transformed)

    z = np.arange(1, bins) - bins / 2
    inner = _inverse_yeojohnson(y_center + z * y_spread, lmbda) * scale + offset

    low = int(np.sum(np.isneginf(inner)))
    high = int(np.sum(np.isposinf(inner)))
    kept = list(labels)[low : len(labels) - high]
    if len(kept) < 2:
        return None
    return Binning(
        labels=tuple(kept),
        edges=(-np.inf, *(float(e) for e in inner[low : len(inner) - high]), np.inf),
    )


def _robust_location(x: np.ndarray) -> tuple[float, float]:
    """Return the median and a robust standard deviation of the values.

    The deviation is the interquartile range scaled to a normal one, or the
    plain standard deviation when more than half of the values are equal.
    """
    q1, median, q3 = np.percentile(x, [25, 50, 75])
    spread = (q3 - q1) / _IQR_PER_SD
    if spread == 0:
        spread = float(np.std(x))
    return float(median), float(spread)


def _inverse_yeojohnson(y: np.ndarray, lmbda: float) -> np.ndarray:
    """Return the values a Yeo-Johnson transform with `lmbda` maps to `y`.

    A `y` outside what the transform can reach maps to -inf or inf.
    """
    x = np.empty_like(y, dtype=float)
    pos = y >= 0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if abs(lmbda) < 1e-12:
            x[pos] = np.expm1(y[pos])
        else:
            base = lmbda * y[pos] + 1
            x[pos] = np.where(base > 0, np.power(base, 1 / lmbda) - 1, np.inf)
        neg = ~pos
        if abs(lmbda - 2) < 1e-12:
            x[neg] = -np.expm1(-y[neg])
        else:
            base = 1 - (2 - lmbda) * y[neg]
            x[neg] = np.where(base > 0, 1 - np.power(base, 1 / (2 - lmbda)), -np.inf)
    return x


def _unbounded(left_edges: list[float]) -> tuple[float, ...]:
    """Return cut boundaries reaching to infinity on both sides."""
    return (-np.inf, *left_edges[1:], np.inf)


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
