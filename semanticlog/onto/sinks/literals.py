"""Datatype resolution for literal values.

A pandas cell becomes a typed literal. A float is an `xsd:decimal` in every
profile, the type owlapy writes for it. Under OWL 2 EL, whose datatype map has
no `xsd:boolean` or `xsd:date`, those two are written as strings. A control
character that XML 1.0 forbids becomes U+FFFD in every text. Both sinks call
`resolve_literal` and render the result.
"""

from __future__ import annotations

import math
import re
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

OWL_DATATYPES: frozenset[str] = frozenset(
    {INTEGER, DECIMAL, DOUBLE, BOOLEAN, DATE_TIME, STRING}
)
"""The datatypes a data property may be declared to range over."""

REPLACEMENT_CHARACTER = "\ufffd"

# XML 1.0 forbids every C0 control character but tab, line feed and carriage
# return. It allows DEL and the C1 controls, so they stay.
_FORBIDDEN_CONTROLS = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f]")

_EL_SUBSTITUTES: dict[str, str] = {
    BOOLEAN: STRING,
    DATE: STRING,
}


def profile_datatype(datatype: str, profile: OWLProfile) -> str:
    """Return the datatype a literal of `datatype` is written under in a profile."""
    if profile == OWLProfile.OWL2EL:
        return _EL_SUBSTITUTES.get(datatype, datatype)
    return datatype


def replace_control_characters(text: str) -> tuple[str, bool]:
    """Return the text with each forbidden control character as U+FFFD.

    Returns:
        The text, and whether any character was replaced.
    """
    replaced, count = _FORBIDDEN_CONTROLS.subn(REPLACEMENT_CHARACTER, text)
    return replaced, count > 0


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
        replaced_controls: Whether a control character in the text was
            replaced by U+FFFD.
    """

    value: Any
    datatype: str
    text: str
    replaced_controls: bool = False


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
        # `xsd:decimal` takes no exponent, so the float is formatted without
        # one. The value stays a float, which owlapy writes as a decimal.
        return DECIMAL, format_number(value)
    return STRING, str(value)


def resolve_literal(value: Any, profile: OWLProfile) -> Literal | None:
    """Return the literal a value is emitted as, or None when it is missing.

    Args:
        value: A cell from a table.
        profile: Profile of the build. Under EL, a boolean or a date is
            written as a string.

    Returns:
        The value with its datatype, or None if there was no value. A string
        has its forbidden control characters replaced by U+FFFD.
    """
    value = normalize_value(value)
    if value is None:
        return None

    datatype, text = _plain_datatype(value)
    if datatype == STRING:
        text, replaced = replace_control_characters(text)
        return Literal(text, STRING, text, replaced)
    if profile != OWLProfile.OWL2EL:
        return Literal(value, datatype, text)

    substitute = _EL_SUBSTITUTES.get(datatype)
    if substitute is None:
        return Literal(value, datatype, text)
    return Literal(text, substitute, text)
