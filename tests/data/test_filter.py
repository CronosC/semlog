import pandas as pd
import pm4py

from semanticlog.data.filtering import (
    filter_remove_months,
    filter_to_timeintervall,
    filter_variants_by_maximum_repetition,
    filter_variants_by_minimum_occurence,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pm4py_log(records: list[dict]) -> pd.DataFrame:
    """Build a pm4py-formatted DataFrame ready for variant filtering."""
    df = pd.DataFrame(records)
    return pm4py.format_dataframe(
        df,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )


# ---------------------------------------------------------------------------
# filter_to_timeintervall
# ---------------------------------------------------------------------------


def test_filter_to_timeintervall_timezone_aware_keeps_only_events_in_window():
    base = pd.Timestamp("2023-03-15 12:00:00", tz="UTC")
    log = pd.DataFrame(
        {
            "time:timestamp": pd.to_datetime(
                [
                    base - pd.Timedelta(minutes=30),
                    base + pd.Timedelta(minutes=30),
                    base + pd.Timedelta(hours=1),
                ],
                utc=True,
            ),
            "case:concept:name": ["case1", "case2", "case3"],
        }
    )
    mintime = int((base - pd.Timedelta(minutes=15)).timestamp())
    maxtime = int((base + pd.Timedelta(minutes=45)).timestamp())

    result = filter_to_timeintervall(log, mintime, maxtime)

    assert len(result) == 1
    assert result.iloc[0]["case:concept:name"] == "case2"


def test_filter_to_timeintervall_timezone_naive_keeps_only_events_in_window():
    base = pd.Timestamp("2023-03-15 12:00:00")
    log = pd.DataFrame(
        {
            "time:timestamp": pd.to_datetime(
                [
                    base - pd.Timedelta(minutes=30),
                    base + pd.Timedelta(minutes=30),
                    base + pd.Timedelta(hours=1),
                ]
            ),
            "case:concept:name": ["case1", "case2", "case3"],
        }
    )
    mintime = int((base - pd.Timedelta(minutes=15)).timestamp())
    maxtime = int((base + pd.Timedelta(minutes=45)).timestamp())

    result = filter_to_timeintervall(log, mintime, maxtime)

    assert len(result) == 1
    assert result.iloc[0]["case:concept:name"] == "case2"


def test_filter_to_timeintervall_returns_empty_when_nothing_in_range():
    base = pd.Timestamp("2023-03-15 12:00:00", tz="UTC")
    log = pd.DataFrame(
        {
            "time:timestamp": pd.to_datetime(
                [base - pd.Timedelta(hours=2), base + pd.Timedelta(hours=2)],
                utc=True,
            ),
            "case:concept:name": ["case1", "case2"],
        }
    )
    # Window: ±15 min around base — neither event falls inside
    mintime = int((base - pd.Timedelta(minutes=15)).timestamp())
    maxtime = int((base + pd.Timedelta(minutes=15)).timestamp())

    result = filter_to_timeintervall(log, mintime, maxtime)

    assert len(result) == 0


def test_filter_to_timeintervall_keeps_all_when_all_in_range():
    base = pd.Timestamp("2023-03-15 12:00:00", tz="UTC")
    log = pd.DataFrame(
        {
            "time:timestamp": pd.to_datetime(
                [
                    base,
                    base + pd.Timedelta(minutes=10),
                    base + pd.Timedelta(minutes=20),
                ],
                utc=True,
            ),
            "case:concept:name": ["c1", "c2", "c3"],
        }
    )
    mintime = int((base - pd.Timedelta(hours=1)).timestamp())
    maxtime = int((base + pd.Timedelta(hours=1)).timestamp())

    result = filter_to_timeintervall(log, mintime, maxtime)

    assert len(result) == 3


# ---------------------------------------------------------------------------
# filter_remove_months
# ---------------------------------------------------------------------------


def test_filter_remove_months_removes_cases_with_events_in_listed_months():
    dates = pd.to_datetime(["2023-01-15", "2023-02-20", "2023-03-10", "2023-04-05"])
    log = pd.DataFrame(
        {
            "time:timestamp": dates,
            "case:concept:name": ["case1", "case2", "case3", "case1"],
        }
    )
    # case1 has events in Jan and Apr; case3 has event in Mar
    result = filter_remove_months(log, months=[3, 4])

    remaining = set(result["case:concept:name"].unique())
    assert remaining == {"case2"}
    assert "case1" not in remaining
    assert "case3" not in remaining


def test_filter_remove_months_unchanged_when_no_match():
    dates = pd.to_datetime(["2023-01-15", "2023-02-20"])
    log = pd.DataFrame(
        {
            "time:timestamp": dates,
            "case:concept:name": ["case1", "case2"],
        }
    )
    # Remove months 6,7 which don't appear in the log
    result = filter_remove_months(log, months=[6, 7])

    assert len(result) == 2
    assert set(result["case:concept:name"].unique()) == {"case1", "case2"}


def test_filter_remove_months_removes_all_when_all_cases_touch_listed_months():
    dates = pd.to_datetime(["2023-03-01", "2023-03-15"])
    log = pd.DataFrame(
        {
            "time:timestamp": dates,
            "case:concept:name": ["case1", "case2"],
        }
    )
    result = filter_remove_months(log, months=[3])
    assert len(result) == 0


# ---------------------------------------------------------------------------
# filter_variants_by_minimum_occurence
# ---------------------------------------------------------------------------


def test_filter_variants_by_minimum_occurence_removes_rare_variants():
    # Variant A→B: 3 cases; Variant C: 1 case
    records = [
        {
            "case:concept:name": "c1",
            "concept:name": "A",
            "time:timestamp": pd.Timestamp("2022-01-01"),
        },
        {
            "case:concept:name": "c1",
            "concept:name": "B",
            "time:timestamp": pd.Timestamp("2022-01-02"),
        },
        {
            "case:concept:name": "c2",
            "concept:name": "A",
            "time:timestamp": pd.Timestamp("2022-01-03"),
        },
        {
            "case:concept:name": "c2",
            "concept:name": "B",
            "time:timestamp": pd.Timestamp("2022-01-04"),
        },
        {
            "case:concept:name": "c3",
            "concept:name": "A",
            "time:timestamp": pd.Timestamp("2022-01-05"),
        },
        {
            "case:concept:name": "c3",
            "concept:name": "B",
            "time:timestamp": pd.Timestamp("2022-01-06"),
        },
        {
            "case:concept:name": "c4",
            "concept:name": "C",
            "time:timestamp": pd.Timestamp("2022-01-07"),
        },
    ]
    log = _make_pm4py_log(records)

    result = filter_variants_by_minimum_occurence(log, minimum_occurences=2)

    # c4 (variant C, 1 occurrence) removed; c1, c2, c3 kept
    assert "c4" not in result["case:concept:name"].values
    assert "c1" in result["case:concept:name"].values
    assert "c2" in result["case:concept:name"].values
    assert "c3" in result["case:concept:name"].values


def test_filter_variants_by_minimum_occurence_keeps_all_when_all_meet_threshold():
    records = [
        {
            "case:concept:name": "c1",
            "concept:name": "A",
            "time:timestamp": pd.Timestamp("2022-01-01"),
        },
        {
            "case:concept:name": "c2",
            "concept:name": "A",
            "time:timestamp": pd.Timestamp("2022-01-02"),
        },
    ]
    log = _make_pm4py_log(records)

    result = filter_variants_by_minimum_occurence(log, minimum_occurences=1)

    assert set(result["case:concept:name"].unique()) == {"c1", "c2"}


# ---------------------------------------------------------------------------
# filter_variants_by_maximum_repetition
# ---------------------------------------------------------------------------


def test_filter_variants_by_maximum_repetition_downsamples_overrepresented_variants():
    # 3 cases all have variant A→B; max allowed is 2
    records = [
        {
            "case:concept:name": "c1",
            "concept:name": "A",
            "time:timestamp": pd.Timestamp("2022-01-01"),
        },
        {
            "case:concept:name": "c1",
            "concept:name": "B",
            "time:timestamp": pd.Timestamp("2022-01-02"),
        },
        {
            "case:concept:name": "c2",
            "concept:name": "A",
            "time:timestamp": pd.Timestamp("2022-01-03"),
        },
        {
            "case:concept:name": "c2",
            "concept:name": "B",
            "time:timestamp": pd.Timestamp("2022-01-04"),
        },
        {
            "case:concept:name": "c3",
            "concept:name": "A",
            "time:timestamp": pd.Timestamp("2022-01-05"),
        },
        {
            "case:concept:name": "c3",
            "concept:name": "B",
            "time:timestamp": pd.Timestamp("2022-01-06"),
        },
    ]
    log = _make_pm4py_log(records)

    result = filter_variants_by_maximum_repetition(log, maximum_repetition=2, seed=42)

    unique_cases = result["case:concept:name"].unique()
    assert len(unique_cases) == 2


def test_filter_variants_by_maximum_repetition_reproducible_with_seed():
    records = [
        {
            "case:concept:name": f"c{i}",
            "concept:name": "A",
            "time:timestamp": pd.Timestamp(f"2022-01-{i+1:02d}"),
        }
        for i in range(5)
    ]
    log = _make_pm4py_log(records)

    result1 = filter_variants_by_maximum_repetition(log, maximum_repetition=3, seed=99)
    result2 = filter_variants_by_maximum_repetition(log, maximum_repetition=3, seed=99)

    cases1 = sorted(result1["case:concept:name"].unique())
    cases2 = sorted(result2["case:concept:name"].unique())
    assert cases1 == cases2
