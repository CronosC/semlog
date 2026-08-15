"""What datatype a value carries in the ontology.

A value arrives from a pandas table and leaves as a typed literal. Which type
it leaves as depends on the OWL 2 profile the build names, because the EL
datatype map is smaller than the DL one and has no `xsd:double`, `xsd:boolean`
or `xsd:date`.

That decision is made once, here, so that both sinks agree on it and a backend
is left with nothing to do but render what it is given.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from semanticlog.core.enums import OWLProfile
from semanticlog.core.vocabulary import format_number

# The XSD types a build emits, as local names in the XSD namespace.
BOOLEAN = "boolean"
DATE = "date"
DATE_TIME = "dateTime"
DECIMAL = "decimal"
DOUBLE = "double"
INTEGER = "integer"
STRING = "string"

# What OWL 2 EL has instead, for the three types its datatype map leaves out.
_EL_SUBSTITUTES: dict[str, str] = {
    BOOLEAN: STRING,
    DATE: STRING,
    DOUBLE: DECIMAL,
}


def normalize_value(value: Any) -> Any:
    """Convert a pandas value into something a literal can be made from.

    Every flavor of missing value becomes None, and numpy scalars become their
    Python equivalents, so that nothing below has to handle either.

    Args:
        value: A cell from a table.

    Returns:
        The plain Python value, or None when it was missing.
    """
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.to_pydatetime()
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, np.generic):
        value = value.item()
        return None if isinstance(value, float) and pd.isna(value) else value
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


@dataclass(frozen=True)
class Literal:
    """One value with the datatype it is emitted under.

    Attributes:
        value: The value to emit. A profile that has no room for the type it
            arrived as replaces it, so a boolean under EL arrives here as the
            text `true`.
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
        profile: Profile the build stays inside, which decides what the three
            types EL has no room for become.

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
        # An exponent is legal in `xsd:double` and not in `xsd:decimal`. The
        # value stays a float, which is the one Python type both backends
        # already write as a decimal.
        return Literal(value, substitute, format_number(value))
    return Literal(text, substitute, text)
