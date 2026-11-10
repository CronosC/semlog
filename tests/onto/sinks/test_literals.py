"""Datatype resolution for literal values."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from semlog.core.enums import OWLProfile
from semlog.core.errors import InputError
from semlog.onto.sinks.literals import resolve_literal

DL = OWLProfile.OWL2DL
EL = OWLProfile.OWL2EL


def test_a_value_takes_the_xsd_datatype_of_its_type():
    values = {
        "boolean": True,
        "dateTime": datetime(2024, 1, 2, 9, 30),
        "date": date(2024, 1, 2),
        "integer": np.int64(3),
        "decimal": 2.5,
        "string": "web",
    }

    assert {k: resolve_literal(v, DL).datatype for k, v in values.items()} == {
        k: k for k in values
    }


def test_a_missing_value_resolves_to_nothing():
    assert [resolve_literal(v, DL) for v in (None, float("nan"), pd.NaT, pd.NA)] == [
        None
    ] * 4


@pytest.mark.parametrize("value", [float("inf"), np.float32("-inf")])
def test_an_infinite_number_is_rejected(value):
    with pytest.raises(InputError, match="cannot be written as an XSD numeric"):
        resolve_literal(value, DL)


@pytest.mark.parametrize("profile", [DL, EL])
def test_a_float_is_a_decimal_written_without_an_exponent(profile):
    literal = resolve_literal(1.0e20, profile)

    assert (literal.datatype, literal.text) == ("decimal", "100000000000000000000")


def test_a_control_character_in_a_string_becomes_a_replacement_character():
    literal = resolve_literal("a\x00b\x01c\x1fd", DL)

    assert (literal.value, literal.text) == ("a\ufffdb\ufffdc\ufffdd",) * 2
    assert literal.replaced_controls


def test_tab_line_feed_carriage_return_and_delete_stay_in_a_string():
    literal = resolve_literal("a\tb\nc\rd\x7fe\x85f", DL)

    assert literal.text == "a\tb\nc\rd\x7fe\x85f"
    assert not literal.replaced_controls
