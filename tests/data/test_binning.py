"""Reducing numeric columns to labeled bins.

Fitting and applying are separate, so these check the `Binning` a fit returns
and what it does to a column, rather than one call that did both.
"""

import numpy as np
import pandas as pd
import pytest

from semanticlog.data import binning as binning_module
from semanticlog.data.binning import (
    Binning,
    fit_dist_bins,
    fit_equal_bins,
    get_best_fit_distribution,
)

LABELS = ("very_low", "low", "average", "high", "very_high")


def _fits_nothing(*args, **kwargs):
    """Stand in for a fit that refused every candidate distribution."""
    return (None, None, None, None)


def labels_of(binning: Binning, values: pd.Series) -> list:
    """Return the label each value lands on, as plain strings."""
    return [None if pd.isna(v) else str(v) for v in binning.apply(values)]


# --------------------------------------------------------------------------
# Equal-frequency bins
# --------------------------------------------------------------------------


def test_equal_bins_use_every_label():
    series = pd.Series(range(20))

    binning = fit_equal_bins(series, bins=3, labels=["Low", "Average", "High"])

    assert set(labels_of(binning, series)) == {"Low", "Average", "High"}


def test_equal_bins_work_when_the_bin_count_equals_the_value_count():
    series = pd.Series([1, 1, 2, 2])

    binning = fit_equal_bins(series, bins=2, labels=["Low", "High"])

    assert labels_of(binning, series) == ["Low", "Low", "High", "High"]


def test_repeated_values_collapse_bins_and_relabel_them():
    # Five bins were asked for, but three distinct values leave three.
    series = pd.Series([1, 1, 1, 2, 2, 2, 3, 3, 3])

    binning = fit_equal_bins(series, bins=5, labels=LABELS)

    assert binning.labels == ("low", "average", "high")


def test_a_single_distinct_value_cannot_be_binned():
    assert (
        fit_equal_bins(pd.Series([7.0, 7.0, 7.0]), bins=3, labels=list("abc")) is None
    )


def test_an_empty_column_cannot_be_binned():
    assert fit_equal_bins(pd.Series([], dtype=float), bins=2, labels=["a", "b"]) is None


def test_equal_bins_reach_beyond_the_values_they_were_fitted_on():
    """The outermost edges are unbounded, so a later value cannot fall out."""
    binning = fit_equal_bins(pd.Series(range(20)), bins=2, labels=["Low", "High"])

    assert labels_of(binning, pd.Series([-500.0, 500.0])) == ["Low", "High"]


# --------------------------------------------------------------------------
# Distribution bins
# --------------------------------------------------------------------------


def test_dist_bins_only_use_the_labels_they_were_given():
    series = pd.Series(np.random.default_rng(0).normal(0, 1, 200))

    binning = fit_dist_bins(series, bins=5, labels=LABELS)

    assert set(labels_of(binning, series)) <= set(LABELS)


def test_dist_bins_use_every_label_for_normal_data():
    series = pd.Series(np.random.default_rng(0).normal(0, 1, 300))

    binning = fit_dist_bins(series, bins=3, labels=["Low", "Average", "High"])

    assert set(labels_of(binning, series)) == {"Low", "Average", "High"}


@pytest.mark.parametrize(
    ("dist_name", "sample"),
    [
        ("norm", lambda rng: rng.normal(0, 1, 300)),
        ("lognorm", lambda rng: rng.lognormal(0, 1, 300)),
        ("expon", lambda rng: rng.exponential(1, 300)),
        ("gamma", lambda rng: rng.gamma(2, 2, 300)),
    ],
)
def test_every_value_of_a_well_fitted_column_lands_in_a_bin(dist_name, sample):
    """A positive-support fit left free settles its location on the smallest
    value, which would put that value outside every bin."""
    series = pd.Series(sample(np.random.default_rng(0)))

    binning = fit_dist_bins(series, bins=3, labels=["Low", "Average", "High"])

    assert binning.apply(series).notna().all()


def test_a_low_cardinality_column_cannot_be_dist_binned():
    series = pd.Series([1, 2, 3, 1, 2, 3])

    assert fit_dist_bins(series, bins=5, labels=LABELS, min_values=10) is None


def test_a_column_that_fits_nothing_falls_back_to_equal_frequency(monkeypatch):
    """The normal fits almost anything, so this is reached only when every
    candidate is refused."""
    monkeypatch.setattr(binning_module, "get_best_fit_distribution", _fits_nothing)
    series = pd.Series(range(50))

    binning = fit_dist_bins(series, bins=3, labels=["Low", "Average", "High"])

    assert binning is not None
    assert binning.dist is None
    assert set(labels_of(binning, series)) == {"Low", "Average", "High"}


# --------------------------------------------------------------------------
# Choosing the distribution
# --------------------------------------------------------------------------


def test_the_best_fit_for_normal_data_is_the_normal():
    series = pd.Series(np.random.default_rng(42).normal(0, 1, 500))

    assert get_best_fit_distribution(series)[1] == "norm"


def test_nothing_fits_an_empty_column():
    assert get_best_fit_distribution(pd.Series([], dtype=float)) == (None,) * 4


def test_a_fit_below_the_p_value_threshold_is_rejected():
    series = pd.Series(np.random.default_rng(0).normal(0, 1, 100))

    assert get_best_fit_distribution(series, p=1.0) == (None,) * 4


def test_a_positive_only_distribution_is_skipped_for_a_column_with_negatives():
    series = pd.Series(np.random.default_rng(0).normal(0, 1, 300))

    _, dist_name, _, _ = get_best_fit_distribution(
        series, candidates=["lognorm", "gamma", "expon"]
    )

    assert dist_name is None


def test_the_ranking_and_the_cut_use_the_same_fit():
    """Ranking candidates under one constraint and cutting under another
    picks a distribution the bins are then not taken from."""
    series = pd.Series(np.random.default_rng(0).lognormal(0, 1, 300))

    _, dist_name, params, _ = get_best_fit_distribution(series)
    binning = fit_dist_bins(series, bins=3, labels=["Low", "Average", "High"])

    assert binning.dist == dist_name
    assert binning.params == tuple(params)


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("labels", "surviving", "expected"),
    [
        (["a", "b", "c", "d", "e"], 5, ["a", "b", "c", "d", "e"]),
        (["a", "b", "c", "d", "e"], 6, ["a", "b", "c", "d", "e"]),
        (["a", "b", "c", "d", "e"], 3, ["b", "c", "d"]),
        (["a", "b", "c", "d", "e"], 1, ["c"]),
        (["a", "b", "c", "d", "e"], 2, ["b", "d"]),
        (["a", "b", "c", "d", "e"], 4, ["a", "b", "d", "e"]),
        (["a", "b", "c", "d"], 2, ["b", "c"]),
    ],
)
def test_fewer_bins_take_their_labels_symmetrically(labels, surviving, expected):
    """An odd count keeps the center, an even one drops it."""
    from semanticlog.data.binning import _select_labels

    assert _select_labels(labels, surviving) == expected


@pytest.mark.parametrize("fit", [fit_equal_bins, fit_dist_bins])
def test_a_label_per_bin_is_required(fit):
    with pytest.raises(ValueError, match="needs exactly 3 labels"):
        fit(pd.Series(range(20)), bins=3, labels=["low", "high"])
