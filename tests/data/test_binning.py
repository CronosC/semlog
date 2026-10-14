"""Reducing numeric columns to labeled bins."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from semanticlog.core.errors import ConfigError
from semanticlog.data.binning import (
    Binning,
    _select_labels,
    check_bins,
    fit_dist_bins,
    fit_equal_bins,
    get_best_fit_distribution,
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
    series = pd.Series([1] * 30 + [2] * 30 + [3] * 30 + [4, 5, 6])

    binning = fit_equal_bins(series, bins=5, labels=LABELS, min_values=0)

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


def test_a_close_fit_cuts_by_the_distribution_the_ranking_chose():
    series = pd.Series(_rng().lognormal(0, 1, 300))

    fit = get_best_fit_distribution(series)
    binning = fit_dist_bins(series, bins=3, labels=THREE)

    assert (binning.dist, binning.params) == ("lognorm", fit.params)


@pytest.mark.parametrize(
    "sample",
    [lambda rng: rng.normal(0, 1, 300), lambda rng: rng.gamma(2, 2, 300)],
    ids=["norm", "gamma"],
)
def test_every_value_of_a_fitted_column_and_zero_land_in_a_bin(sample):
    series = pd.Series(sample(_rng()))
    binning = fit_dist_bins(series, bins=3, labels=THREE)

    labeled = binning.apply(pd.concat([series, pd.Series([0.0])], ignore_index=True))

    assert int(labeled.isna().sum()) == 0


def test_a_value_beyond_either_tail_of_the_fitted_shape_lands_in_the_outer_bin():
    """The fitted cdf maps it to an infinite z-score, which the outer edges include."""
    binning = Binning(
        labels=("low", "high"),
        edges=(-np.inf, 0.0, np.inf),
        dist="gamma",
        params=(2.0, 0.0, 2.0),
    )

    assert labels_of(binning, pd.Series([0.0, 1e-320, np.inf])) == [
        "low",
        "low",
        "high",
    ]


def test_a_column_no_candidate_fits_closely_is_cut_by_equal_frequency():
    """A uniform column wins as a normal on AIC and loses on the p-value."""
    series = pd.Series(_rng().uniform(0, 100, 500))

    binning = fit_dist_bins(series, bins=5, labels=LABELS)

    assert get_best_fit_distribution(series, p=0.0).name == "norm"
    assert binning.dist is None
    assert set(labels_of(binning, series)) == set(LABELS)


# --------------------------------------------------------------------------
# Choosing the distribution
# --------------------------------------------------------------------------


def test_nothing_fits_an_empty_column():
    assert get_best_fit_distribution(pd.Series([], dtype=float)) is None


@pytest.mark.parametrize(("dist_name", "free_params"), [("norm", 2), ("gamma", 2)])
def test_the_aic_charges_only_the_parameters_the_fit_left_free(dist_name, free_params):
    series = pd.Series(_rng().gamma(2, 2, 300))

    fit = get_best_fit_distribution(series, candidates=[dist_name], p=0.0)
    loglik = np.sum(getattr(stats, dist_name).logpdf(series.to_numpy(), *fit.params))

    assert fit.aic == pytest.approx(2 * free_params - 2 * loglik)


def test_a_candidate_without_a_finite_aic_is_skipped(monkeypatch):
    monkeypatch.setattr(
        stats.norm, "logpdf", lambda x, *params: np.full(np.shape(x), np.nan)
    )
    series = pd.Series(_rng().exponential(1, 200))

    assert (
        get_best_fit_distribution(series, candidates=["norm", "expon"]).name == "expon"
    )


def test_a_positive_only_distribution_is_skipped_for_a_column_with_negatives():
    series = pd.Series(_rng().normal(0, 1, 300))

    assert get_best_fit_distribution(series, candidates=["lognorm", "gamma"]) is None


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
