"""Reducing numeric columns to labeled bins."""

import numpy as np
import pandas as pd
import pytest

from semanticlog.core.errors import ConfigError
from semanticlog.data.binning import (
    Binning,
    _select_labels,
    check_bins,
    fit_dist_bins,
    fit_equal_bins,
)

LABELS = ("very_low", "low", "average", "high", "very_high")

THREE = ["low", "average", "high"]


def _rng():
    return np.random.default_rng(0)


def labels_of(binning, values):
    return [None if pd.isna(v) else str(v) for v in binning.apply(values)]


# --------------------------------------------------------------------------
# Equal-frequency bins
# --------------------------------------------------------------------------


def test_equal_bins_use_every_label():
    series = pd.Series(range(20))

    binning = fit_equal_bins(series, bins=3, labels=THREE)

    assert set(labels_of(binning, series)) == set(THREE)


def test_equal_bins_reach_beyond_the_values_they_were_fitted_on():
    binning = fit_equal_bins(pd.Series(range(20)), bins=2, labels=["low", "high"])

    assert labels_of(binning, pd.Series([-500.0, 500.0])) == ["low", "high"]


def test_bins_lost_to_repeated_values_take_the_labels_around_the_center():
    series = pd.Series([1] * 30 + [2] * 30 + [3] * 30 + list(range(4, 15)))

    binning = fit_equal_bins(series, bins=5, labels=LABELS)

    assert binning.labels == ("low", "average", "high")


@pytest.mark.parametrize(
    ("labels", "surviving", "expected"),
    [
        (["a", "b", "c", "d", "e"], 2, ["b", "d"]),
        (["a", "b", "c", "d"], 2, ["b", "c"]),
    ],
)
def test_an_even_number_of_surviving_bins_drops_the_center_label(
    labels, surviving, expected
):
    assert _select_labels(labels, surviving) == expected


@pytest.mark.parametrize("fit", [fit_equal_bins, fit_dist_bins])
def test_a_column_with_too_few_distinct_values_is_not_binned(fit):
    assert fit(pd.Series([0, 1] * 50), bins=5, labels=LABELS) is None


# --------------------------------------------------------------------------
# Distribution bins
# --------------------------------------------------------------------------


def _shares(binning, series) -> list[float]:
    counts = binning.apply(series).value_counts(sort=False)
    return [counts[label] / len(series) for label in binning.labels]


def test_a_normal_column_puts_the_normal_share_in_each_bin():
    series = pd.Series(_rng().normal(50, 10, 20000))

    shares = _shares(fit_dist_bins(series, bins=5, labels=LABELS), series)

    # Cuts at -1.5, -0.5, 0.5 and 1.5 standard deviations.
    assert shares == pytest.approx([0.067, 0.242, 0.383, 0.242, 0.067], abs=0.02)


def test_a_skewed_column_is_cut_around_its_typical_value_not_by_frequency():
    series = pd.Series(_rng().lognormal(8, 0.8, 20000))

    by_distance = _shares(fit_dist_bins(series, bins=5, labels=LABELS), series)

    assert by_distance == pytest.approx([0.067, 0.242, 0.383, 0.242, 0.067], abs=0.03)
    assert fit_dist_bins(series).edges != fit_equal_bins(series).edges


def test_a_column_without_outliers_fills_no_outer_bin():
    series = pd.Series(_rng().uniform(0, 100, 5000))

    shares = _shares(fit_dist_bins(series, bins=5, labels=LABELS), series)

    assert (shares[0], shares[-1]) == (0, 0)


@pytest.mark.parametrize(
    "sample",
    [
        lambda rng: rng.normal(0, 1, 2000) ** 3,
        lambda rng: rng.lognormal(14, 1, 2000),
        lambda rng: -rng.lognormal(3, 1, 2000),
    ],
    ids=["negative and positive", "large", "negative"],
)
def test_the_cuts_rise_strictly_and_label_every_value(sample):
    series = pd.Series(sample(_rng()))

    binning = fit_dist_bins(series, bins=5, labels=LABELS)

    assert all(a < b for a, b in zip(binning.edges, binning.edges[1:], strict=False))
    assert int(binning.apply(series).isna().sum()) == 0


@pytest.mark.parametrize("bins", [2, 3, 4])
def test_the_bins_are_one_standard_deviation_wide_around_the_median(bins):
    series = pd.Series(_rng().normal(0, 1, 50000))

    inner = fit_dist_bins(series, bins=bins, labels=LABELS[:bins]).edges[1:-1]

    assert inner == pytest.approx(np.arange(1, bins) - bins / 2, abs=0.05)


def test_a_bin_reports_its_bounds():
    binning = Binning(labels=("low", "high"), edges=(-np.inf, 3.5, np.inf))

    assert binning.bounds("low") == (-np.inf, 3.5)
    assert binning.bounds("high") == (3.5, np.inf)


# --------------------------------------------------------------------------
# Bin settings
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bins", "labels", "message"),
    [
        (1, ["only"], "bins must be at least 2, got 1"),
        (3, ["low", "high"], r"bins=3 needs exactly 3 labels, got 2"),
        (3, ["low", "low", "high"], r"must all differ, got \['low'\] more than once"),
    ],
)
def test_a_bin_setting_that_cannot_label_every_bin_is_rejected(bins, labels, message):
    with pytest.raises(ConfigError, match=message):
        check_bins(bins, labels)
