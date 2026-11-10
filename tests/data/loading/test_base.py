"""The columns that carry the structure of a flat log."""

import pytest

from semlog.core.errors import ConfigError
from semlog.data.loading.base import LoaderColumns


def test_a_column_setting_no_loader_reads_is_rejected_listing_the_valid_ones():
    with pytest.raises(
        ConfigError,
        match="Unknown column setting\\(s\\): 'case_column'. Valid settings: case_col, act_col, time_col, object_cols.",
    ):
        LoaderColumns.from_mapping({"case_column": "id"})


def test_object_cols_given_as_one_string_is_rejected():
    with pytest.raises(
        ConfigError,
        match=r"object_cols must be a list, got the string 'resource'\. Pass \['resource'\]\.",
    ):
        LoaderColumns(object_cols="resource")
