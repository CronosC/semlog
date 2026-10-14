from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from semanticlog.core.enums import OWLProfile
from semanticlog.core.errors import InputError
from semanticlog.onto.sinks.literals import resolve_literal

DL = OWLProfile.OWL2DL
EL = OWLProfile.OWL2EL


def test_a_value_takes_the_xsd_datatype_of_its_type():
    values = {
        "boolean": True,
        "dateTime": datetime(2024, 1, 2, 9, 30),
        "date": date(2024, 1, 2),
        "integer": np.int64(3),
        "double": 2.5,
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


def test_under_el_a_float_is_a_decimal_written_without_an_exponent():
    literal = resolve_literal(1.0e20, EL)

    assert (literal.datatype, literal.text) == ("decimal", "100000000000000000000")
