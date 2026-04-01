from datetime import datetime

import numpy as np
import pandas as pd

from semanticlog.data.misc import (
    _distribution_quantize,
    expon_quantize,
    gamma_quantize,
    gaussian_quantize,
    get_best_fit_distribution,
    get_weighted_variant_mapping,
    lognorm_quantize,
    quantize,
    quantize_series,
    simple_quantize,
)
from semanticlog.data.oced import OCEDStore

# ---------------------------------------------------------------------------
# get_weighted_variant_mapping
# ---------------------------------------------------------------------------


def test_get_weighted_variant_mapping_counts_variants_correctly():
    data = [
        {
            "case:concept:name": "case1",
            "time:timestamp": datetime(2021, 1, 1, 8, 0),
            "concept:name": "A",
            "event": 1,
        },
        {
            "case:concept:name": "case1",
            "time:timestamp": datetime(2021, 1, 1, 9, 0),
            "concept:name": "B",
            "event": 2,
        },
        {
            "case:concept:name": "case2",
            "time:timestamp": datetime(2021, 1, 2, 10, 0),
            "concept:name": "A",
            "event": 3,
        },
        {
            "case:concept:name": "case2",
            "time:timestamp": datetime(2021, 1, 2, 11, 0),
            "concept:name": "B",
            "event": 4,
        },
        {
            "case:concept:name": "case2",
            "time:timestamp": datetime(2021, 1, 2, 12, 0),
            "concept:name": "C",
            "event": 5,
        },
        {
            "case:concept:name": "case3",
            "time:timestamp": datetime(2021, 1, 1, 8, 0),
            "concept:name": "A",
            "event": 6,
        },
        {
            "case:concept:name": "case3",
            "time:timestamp": datetime(2021, 1, 1, 9, 0),
            "concept:name": "B",
            "event": 7,
        },
    ]
    log = pd.DataFrame(data)
    result = get_weighted_variant_mapping(log)
    # case1 and case3 share variant (A,B); case2 has variant (A,B,C)
    assert result == {("A", "B"): 2, ("A", "B", "C"): 1}


def test_get_weighted_variant_mapping_empty_log_returns_empty_dict():
    log = pd.DataFrame(
        columns=["case:concept:name", "time:timestamp", "concept:name", "event"]
    )
    assert get_weighted_variant_mapping(log) == {}


# ---------------------------------------------------------------------------
# simple_quantize
# ---------------------------------------------------------------------------


def test_simple_quantize_bins_into_expected_labels():
    series = pd.Series(range(20))
    labels = ["Low", "Average", "High"]
    result = simple_quantize(series, bins=3, labels=labels)
    valid = set(result.dropna().astype(str))
    assert valid.issubset(set(labels))
    # All three labels should appear for 20 well-spread values
    assert valid == set(labels)


def test_simple_quantize_works_when_bins_equals_unique_value_count():
    # Exactly 2 unique values and 2 bins with 2 labels — no error, all values binned
    series = pd.Series([1, 1, 2, 2])
    result = simple_quantize(series, bins=2, labels=["Low", "High"])
    assert result.notna().any()
    assert set(result.dropna().astype(str)).issubset({"Low", "High"})


# ---------------------------------------------------------------------------
# quantize_series — low-cardinality fallback
# ---------------------------------------------------------------------------


def test_quantize_series_low_cardinality_returns_string_dtype():
    # 3 unique values <= max(5, 10) → string fallback
    series = pd.Series([1, 2, 3, 1, 2, 3])
    result = quantize_series(series, bins=5, min_values=10)
    assert pd.api.types.is_object_dtype(result) or pd.api.types.is_string_dtype(result)
    assert result.tolist() == ["1", "2", "3", "1", "2", "3"]


def test_quantize_series_high_cardinality_returns_categorical_labels():
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(0, 1, 200))
    labels = ["very_low", "low", "average", "high", "very_high"]
    result = quantize_series(series, bins=5, labels=labels)
    valid = set(result.dropna().astype(str))
    assert valid.issubset(set(labels))


# ---------------------------------------------------------------------------
# get_best_fit_distribution
# ---------------------------------------------------------------------------


def test_get_best_fit_distribution_returns_norm_for_normal_data():
    rng = np.random.default_rng(42)
    series = pd.Series(rng.normal(0, 1, 500))
    _, dist_name, _, _ = get_best_fit_distribution(series)
    assert dist_name == "norm"


def test_get_best_fit_distribution_returns_none_for_empty_series():
    result = get_best_fit_distribution(pd.Series([], dtype=float))
    assert result == (None, None, None, None)


def test_get_best_fit_distribution_returns_none_when_pvalue_threshold_not_met():
    # Setting p=1.0 means no distribution will ever pass the p-value gate
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(0, 1, 100))
    result = get_best_fit_distribution(series, p=1.0)
    assert result == (None, None, None, None)


# ---------------------------------------------------------------------------
# Distribution-specific quantize wrappers
# ---------------------------------------------------------------------------


def test_gaussian_quantize_output_contains_only_expected_labels():
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(0, 1, 300))
    labels = ["Low", "Average", "High"]
    result = gaussian_quantize(series, bins=3, labels=labels)
    valid = set(result.dropna().astype(str))
    assert valid.issubset(set(labels))
    assert valid == set(labels)


def test_lognorm_quantize_output_contains_only_expected_labels():
    rng = np.random.default_rng(1)
    series = pd.Series(rng.lognormal(0, 1, 300))
    labels = ["Low", "Average", "High"]
    result = lognorm_quantize(series, bins=3, labels=labels)
    valid = set(result.dropna().astype(str))
    assert valid.issubset(set(labels))


def test_expon_quantize_output_contains_only_expected_labels():
    rng = np.random.default_rng(2)
    series = pd.Series(rng.exponential(1, 300))
    labels = ["Low", "Average", "High"]
    result = expon_quantize(series, bins=3, labels=labels)
    valid = set(result.dropna().astype(str))
    assert valid.issubset(set(labels))


def test_gamma_quantize_output_contains_only_expected_labels():
    rng = np.random.default_rng(3)
    series = pd.Series(rng.gamma(2, 2, 300))
    labels = ["Low", "Average", "High"]
    result = gamma_quantize(series, bins=3, labels=labels)
    valid = set(result.dropna().astype(str))
    assert valid.issubset(set(labels))


def test_distribution_quantize_returns_all_nan_for_empty_series():
    result = _distribution_quantize(pd.Series([], dtype=float), "norm", bins=3)
    assert result.isna().all()


# ---------------------------------------------------------------------------
# quantize (full OCEDStore)
# ---------------------------------------------------------------------------


def test_quantize_applies_to_numeric_object_attributes():
    oced = OCEDStore()
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "oid": [f"o{i}" for i in range(50)],
            "score": rng.normal(0, 1, 50),
        }
    )
    oced.add_object("item", "oid", df)

    result = quantize(oced)

    obj_df = result.get_object("item")["data"]
    assert not pd.api.types.is_numeric_dtype(obj_df["score"])


def test_quantize_applies_to_numeric_event_attributes():
    oced = OCEDStore()
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "eid": [f"e{i}" for i in range(50)],
            "ts": pd.date_range("2022-01-01", periods=50),
            "value": rng.normal(5, 2, 50),
        }
    )
    oced.add_event("act", "eid", "ts", df)

    result = quantize(oced)

    ev_df = result.get_event("act")["data"]
    assert not pd.api.types.is_numeric_dtype(ev_df["value"])


def test_quantize_skips_id_and_time_columns():
    oced = OCEDStore()
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "eid": list(range(50)),  # numeric id column
            "ts": pd.date_range("2022-01-01", periods=50),
            "score": rng.normal(5, 2, 50),
        }
    )
    oced.add_event("act", "eid", "ts", df)

    result = quantize(oced)

    ev_df = result.get_event("act")["data"]
    # id column must stay numeric
    assert pd.api.types.is_numeric_dtype(ev_df["eid"])
    # attribute column gets binned
    assert not pd.api.types.is_numeric_dtype(ev_df["score"])
