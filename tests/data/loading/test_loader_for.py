"""Choosing a loader by file extension."""

import pytest

from semanticlog.core.errors import InputError
from semanticlog.data.loading import ClassicalLoader, PM4PyLoader, loader_for


@pytest.mark.parametrize(
    ("name", "loader"),
    [
        ("log.jsonocel", PM4PyLoader),
        ("log.xes", ClassicalLoader),
        ("log.CSV", ClassicalLoader),
    ],
)
def test_the_extension_picks_the_loader_in_any_case(tmp_path, name, loader):
    path = tmp_path / name
    path.touch()

    assert type(loader_for(str(path))) is loader


def test_an_unknown_extension_is_rejected_listing_the_ones_read(tmp_path):
    with pytest.raises(
        InputError,
        match=r"No loader reads '\.txt'\. Supported extensions: \.csv, .*\.xmlocel\.",
    ):
        loader_for(str(tmp_path / "log.txt"))


def test_a_flat_log_is_read_with_the_columns_the_config_names(tmp_path):
    path = tmp_path / "log.csv"
    path.touch()

    loader = loader_for(str(path), {"case_col": "case_id", "act_col": "activity"})

    assert (loader.columns.case_col, loader.columns.act_col) == ("case_id", "activity")
