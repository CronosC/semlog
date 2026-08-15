"""Choosing a loader by file extension."""

import pytest

from semanticlog.data.loading import (
    ClassicalLoader,
    PM4PyLoader,
    loader_for,
)


def test_an_ocel_suffix_picks_the_ocel_loader(tmp_path):
    path = tmp_path / "log.jsonocel"
    path.touch()

    assert isinstance(loader_for(str(path), {}), PM4PyLoader)


@pytest.mark.parametrize("suffix", [".csv", ".xes"])
def test_a_flat_suffix_picks_the_classical_loader(tmp_path, suffix):
    path = tmp_path / f"log{suffix}"
    path.touch()

    assert isinstance(loader_for(str(path), {}), ClassicalLoader)


def test_the_suffix_is_read_without_regard_to_case(tmp_path):
    path = tmp_path / "log.CSV"
    path.touch()

    assert isinstance(loader_for(str(path), {}), ClassicalLoader)


def test_an_unknown_suffix_says_which_ones_are_read(tmp_path):
    path = tmp_path / "log.txt"
    path.touch()

    with pytest.raises(ValueError, match="no loader reads") as raised:
        loader_for(str(path), {})

    assert ".jsonocel" in str(raised.value)
    assert ".csv" in str(raised.value)


def test_the_classical_loader_takes_its_columns_from_the_config(tmp_path):
    path = tmp_path / "log.csv"
    path.write_text("case_id,activity,timestamp\nc1,A,2024-01-01 08:00:00\n")

    loader = loader_for(
        str(path),
        {"case_col": "case_id", "act_col": "activity", "time_col": "timestamp"},
    )

    assert isinstance(loader, ClassicalLoader)
    assert loader.case_col == "case_id"
    assert loader.act_col == "activity"
