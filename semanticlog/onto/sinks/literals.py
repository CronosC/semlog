"""Datatype resolution for literal values.

A pandas cell becomes a typed literal. Under OWL 2 EL, whose datatype map has
no `xsd:double`, `xsd:boolean` or `xsd:date`, those three are substituted.
Both sinks call `resolve_literal` and render the result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from semanticlog.core.enums import OWLProfile
from semanticlog.core.errors import InputError
from semanticlog.core.vocabulary import format_number

XSD = "http://www.w3.org/2001/XMLSchema#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"

BOOLEAN = "boolean"
DATE = "date"
DATE_TIME = "dateTime"
DECIMAL = "decimal"
DOUBLE = "double"
INTEGER = "integer"
STRING = "string"

OWL_DATATYPES: frozenset[str] = frozenset({INTEGER, DOUBLE, BOOLEAN, DATE_TIME})
"""The datatypes a data property may be declared to range over."""

_EL_SUBSTITUTES: dict[str, str] = {
    BOOLEAN: STRING,
    DATE: STRING,
    DOUBLE: DECIMAL,
}


def normalize_value(value: Any) -> Any:
    """Convert a pandas cell to a plain Python value.

    Every kind of missing value becomes None. numpy scalars become Python
    scalars.

    Args:
        value: A cell from a table.

    Returns:
        The plain Python value, or None when it was missing.

    Raises:
        InputError: If the value is infinite.
    """
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.to_pydatetime()
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if pd.isna(value):
            return None
        if not math.isfinite(value):
            raise InputError(
                f"{value!r} cannot be written as an XSD numeric literal. Remove "
                "it from the input or replace it with a finite number."
            )
    return value


@dataclass(frozen=True)
class Literal:
    """One value with the datatype it is emitted under.

    Attributes:
        value: The value to emit. Under EL a boolean or date is its text form,
            such as `true`.
        datatype: Local name of its XSD type, such as `integer`.
        text: The value as it is written inside the literal.
    """

    value: Any
    datatype: str
    text: str


def _plain_datatype(value: Any) -> tuple[str, str]:
    """Return the datatype and written form a value takes under OWL 2 DL."""
    if isinstance(value, bool):
        return BOOLEAN, str(value).lower()
    if isinstance(value, datetime):
        return DATE_TIME, value.isoformat()
    if isinstance(value, date):
        return DATE, value.isoformat()
    if isinstance(value, int):
        return INTEGER, str(value)
    if isinstance(value, float):
        return DOUBLE, str(value)
    return STRING, str(value)


def resolve_literal(value: Any, profile: OWLProfile) -> Literal | None:
    """Return the literal a value is emitted as, or None when it is missing.

    Args:
        value: A cell from a table.
        profile: Profile of the build. Under EL, boolean, date and double are
            substituted.

    Returns:
        The value with its datatype, or None if there was no value.
    """
    value = normalize_value(value)
    if value is None:
        return None

    datatype, text = _plain_datatype(value)
    if profile != OWLProfile.OWL2EL:
        return Literal(value, datatype, text)

    substitute = _EL_SUBSTITUTES.get(datatype)
    if substitute is None:
        return Literal(value, datatype, text)
    if substitute == DECIMAL:
        # `xsd:decimal` takes no exponent, so the float is formatted without
        # one. The value stays a float, which both backends write as a decimal.
        return Literal(value, substitute, format_number(value))
    return Literal(text, substitute, text)
