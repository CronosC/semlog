"""Reducing numeric columns to labelled bins."""

import numpy as np
import pandas as pd
import pytest

from semanticlog.data.binning import (
    _cut_by_distribution,
    dist_bin_series,
    equal_bin_series,
    get_best_fit_distribution,
)


def test_equal_bin_series_bins_into_expected_labels():
    series = pd.Series(range(20))
    labels = ["Low", "Average", "High"]
    result = equal_bin_series(series, bins=3, labels=labels)
    valid = set(result.dropna().astype(str))
    assert valid.issubset(set(labels))
    assert valid == set(labels)


def test_equal_bin_series_works_when_bins_equals_unique_value_count():
    series = pd.Series([1, 1, 2, 2])
    result = equal_bin_series(series, bins=2, labels=["Low", "High"])
    assert result.notna().all()
    assert set(result) == {"Low", "High"}
    assert set(result.dropna().astype(str)).issubset({"Low", "High"})


def test_bin_series_low_cardinality_returns_string_dtype():
    series = pd.Series([1, 2, 3, 1, 2, 3])
    result = dist_bin_series(series, bins=5, min_values=10)
    assert pd.api.types.is_object_dtype(result) or pd.api.types.is_string_dtype(result)
    assert result.tolist() == ["1", "2", "3", "1", "2", "3"]


def test_bin_series_high_cardinality_returns_categorical_labels():
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(0, 1, 200))
    labels = ["very_low", "low", "average", "high", "very_high"]
    result = dist_bin_series(series, bins=5, labels=labels)
    valid = set(result.dropna().astype(str))
    assert valid.issubset(set(labels))


def test_get_best_fit_distribution_returns_norm_for_normal_data():
    rng = np.random.default_rng(42)
    series = pd.Series(rng.normal(0, 1, 500))
    _, dist_name, _, _ = get_best_fit_distribution(series)
    assert dist_name == "norm"


def test_get_best_fit_distribution_returns_none_for_empty_series():
    result = get_best_fit_distribution(pd.Series([], dtype=float))
    assert result == (None, None, None, None)


def test_get_best_fit_distribution_returns_none_when_pvalue_threshold_not_met():
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(0, 1, 100))
    result = get_best_fit_distribution(series, p=1.0)
    assert result == (None, None, None, None)


@pytest.mark.parametrize(
    ("dist_name", "sample"),
    [
        ("norm", lambda rng: rng.normal(0, 1, 300)),
        ("lognorm", lambda rng: rng.lognormal(0, 1, 300)),
        ("expon", lambda rng: rng.exponential(1, 300)),
        ("gamma", lambda rng: rng.gamma(2, 2, 300)),
    ],
)
def test_dist_bin_series_output_contains_only_expected_labels(dist_name, sample):
    series = pd.Series(sample(np.random.default_rng(0)))
    labels = ["Low", "Average", "High"]
    result = _cut_by_distribution(series, dist_name, bins=3, labels=labels)
    assert set(result.dropna().astype(str)).issubset(set(labels))


def test_dist_bin_series_uses_every_label_for_normal_data():
    series = pd.Series(np.random.default_rng(0).normal(0, 1, 300))
    labels = ["Low", "Average", "High"]
    result = _cut_by_distribution(series, "norm", bins=3, labels=labels)
    assert set(result.dropna().astype(str)) == set(labels)
