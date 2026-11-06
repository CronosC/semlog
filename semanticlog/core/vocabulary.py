"""The two namespaces and every local name a build may emit.

Upper schema names are `Term`s in the TBox namespace, the same in every build.
Names derived from log content are plain strings under the log base and come
from the functions below. No other module formats an emitted name.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from decimal import Decimal
from typing import Any
from urllib.parse import quote

import pandas as pd

from semanticlog.core.errors import InputError

TBOX_IRI: str = "https://example.org/oced#"
TBOX_PREFIX = "oced"

_LOG_IRI_ROOT = "https://example.org/oced/"

DEFAULT_IRI: str = f"{_LOG_IRI_ROOT}log/"
"""Log base of a build that names none."""


def default_iri(run: str) -> str:
    """Return the log base of a run that sets no `iri`, one path per run stem."""
    return f"{_LOG_IRI_ROOT}{quote(run, safe='')}/"


class Term(str):
    """A name in the TBox namespace.

    It compares and hashes as its text, so it works wherever a name does. A
    sink writes it under `TBOX_IRI` and any plain string under the log base.
    """

    __slots__ = ()


def is_tbox(name: str) -> bool:
    """Whether a name belongs to the TBox namespace."""
    return isinstance(name, Term)


_NON_ALNUM = re.compile(r"[^A-Za-z0-9]")


# `+` and `~` are legal in an IRI but not in an XML element name, and RDF/XML
# writes a property as an element, so a name holding one cannot be serialized.
# `.` and `-` are allowed inside a name but not as its first character, and `.`
# not as its last, because a prefixed name in functional syntax ends there.
SAFE_LOCAL_NAME = re.compile(r"[A-Za-z0-9_](?:[A-Za-z0-9_.-]*[A-Za-z0-9_-])?")


class UnsafeLocalNameError(InputError):
    """A name cannot be used as the local part of an IRI."""


class NameCollisionError(InputError):
    """Two entities would be emitted under one name."""


def is_safe_local_name(name: str) -> bool:
    """Return whether the string form of a name can be the local part of an IRI."""
    return bool(SAFE_LOCAL_NAME.fullmatch(str(name)))


OBJECT = Term("Object")
EVENT = Term("Event")
ATTRIBUTE = Term("Attribute")
TIME = Term("Time")
OBJECT_ATTRIBUTE = Term("ObjectAttribute")
EVENT_ATTRIBUTE = Term("EventAttribute")
ATTRIBUTE_CHANGE = Term("AttributeChange")
DECLARE_CONSTRAINT = Term("DeclareConstraint")

OBSERVED_AT = Term("observed_at")
OBSERVES = Term("observes")
EVENT_OBJECT_RELATION = Term("event_object_relation")
OBJECT_OBJECT_RELATION = Term("object_object_relation")
HAS_ATTRIBUTE = Term("has_attribute")
ATTRIBUTE_OF = Term("attribute_of")
HAS_DATA_VALUE = Term("has_data_value")
TIMESTAMP = Term("timestamp")
HAS_ATTRIBUTE_CHANGE = Term("has_attribute_change")
ATTRIBUTE_CHANGE_VALUE = Term("attribute_change_value")
CHANGED_AT = Term("changed_at")
CAUSED_BY = Term("caused_by")
DF = Term("df")
EF = Term("ef")

ATTRIBUTE_CHANGE_RECORD = "AttrChange"

SCHEMA_CLASSES: frozenset[str] = frozenset(
    {
        OBJECT,
        EVENT,
        ATTRIBUTE,
        TIME,
        OBJECT_ATTRIBUTE,
        EVENT_ATTRIBUTE,
        ATTRIBUTE_CHANGE,
        DECLARE_CONSTRAINT,
    }
)

_INVERTIBLE_PROPERTIES: frozenset[str] = frozenset(
    {
        OBSERVED_AT,
        EVENT_OBJECT_RELATION,
        OBJECT_OBJECT_RELATION,
        HAS_ATTRIBUTE,
        HAS_ATTRIBUTE_CHANGE,
        ATTRIBUTE_CHANGE_VALUE,
        CHANGED_AT,
        CAUSED_BY,
        DF,
        EF,
    }
)

SCHEMA_PROPERTIES: frozenset[str] = _INVERTIBLE_PROPERTIES | {
    HAS_DATA_VALUE,
    TIMESTAMP,
}


INVERSE_PREFIX = "inv_"

_EXPLICIT_INVERSES: dict[Term, Term] = {
    HAS_ATTRIBUTE: ATTRIBUTE_OF,
    OBSERVED_AT: OBSERVES,
}

SCHEMA_NAMED_INVERSES: dict[Term, Term] = {
    **_EXPLICIT_INVERSES,
    **{inverse: prop for prop, inverse in _EXPLICIT_INVERSES.items()},
}


def inverse(prop: str, named: Mapping[str, str] | None = None) -> str:
    """Return the name of the inverse of a property, in its namespace.

    A TBox property in `SCHEMA_NAMED_INVERSES` takes its paired name. A
    property in `named` takes the name paired with it there. Everything else
    takes the `inv_` prefix.

    Args:
        named: Inverse pairs of log properties.
    """
    if isinstance(prop, Term):
        return SCHEMA_NAMED_INVERSES.get(prop, Term(INVERSE_PREFIX + prop))
    if named is not None and prop in named:
        return named[prop]
    return INVERSE_PREFIX + prop


TEMPLATE = Term("template")
ACTIVATION = Term("activation")
TARGET = Term("target")

SCHEMA_ANNOTATIONS: frozenset[str] = frozenset({TEMPLATE, ACTIVATION, TARGET})
"""The annotation properties a build may attach to a DECLARE constraint class."""

BUILD_TOOL = Term("buildTool")
BUILD_CONFIGURATION = Term("buildConfiguration")
SOURCE_LOG = Term("sourceLog")
SOURCE_LOG_SHA256 = Term("sourceLogSha256")
SOURCE_LOG_BYTES = Term("sourceLogBytes")
CASE_CONCEPTS = Term("caseConcepts")
CASE_CONCEPT_SOURCE = Term("caseConceptSource")
DECLARE_FILE = Term("declareFile")
ANNOTATION_DOCUMENT = Term("annotationDocument")
BUILD_COMMAND = Term("buildCommand")
BUILD_CONFIG_FILE = Term("buildConfigFile")
BUILD_INPUT_STATISTICS = Term("buildInputStatistics")
BUILD_OUTPUTS = Term("buildOutputs")
MATERIALIZE_REQUESTED = Term("materializeRequested")

PROVENANCE_ANNOTATIONS: frozenset[str] = frozenset(
    {
        BUILD_TOOL,
        BUILD_CONFIGURATION,
        SOURCE_LOG,
        SOURCE_LOG_SHA256,
        SOURCE_LOG_BYTES,
        CASE_CONCEPTS,
        CASE_CONCEPT_SOURCE,
        DECLARE_FILE,
        ANNOTATION_DOCUMENT,
        BUILD_COMMAND,
        BUILD_CONFIG_FILE,
        BUILD_INPUT_STATISTICS,
        BUILD_OUTPUTS,
        MATERIALIZE_REQUESTED,
    }
)


def start_class(case_concept: str) -> str:
    """Return the class name for events that open a trace of a case concept."""
    return f"StartEvent_{case_concept}"


def end_class(case_concept: str) -> str:
    """Return the class name for events that close a trace of a case concept."""
    return f"EndEvent_{case_concept}"


def df_property(case_concept: str) -> str:
    """Return the directly-follows property name for a case concept."""
    return f"{DF}_{case_concept}"


def ef_property(case_concept: str) -> str:
    """Return the eventually-follows property name for a case concept."""
    return f"{EF}_{case_concept}"


def has_attribute_property(attribute: str) -> str:
    """Return the property name linking an entity to a value of an attribute."""
    return f"has_{attribute}"


def change_value_property(attribute: str) -> str:
    """Return the property name linking a change record to the new value."""
    return f"change_value_{attribute}"


def value_property(attribute: str) -> str:
    """Return the data property name linking a value of an attribute to its literal."""
    return f"has_value_{attribute}"


def attribute_change_class(attribute: str) -> str:
    """Return the class name for changes of one attribute."""
    return f"{attribute}Change"


def bin_class(attribute: str, label: Any) -> str:
    """Return the class name for one bin of a binned numeric attribute."""
    return f"{attribute}_bin_{label}"


def declare_constraint(template: str, activities: Sequence[str]) -> str:
    """Return the class name of a DECLARE constraint, such as `Response__a__b__`."""
    return f"{template.replace(' ', '')}__{'__'.join(activities)}__"


def format_number(value: float) -> str:
    """Return a number as plain digits, without an exponent.

    Returns:
        The digits, with a decimal point when the number has a fraction.

    Raises:
        InputError: If the value is infinite or not a number.
    """
    # `str` writes an exponent outside roughly 1e-4 to 1e16, and neither an IRI
    # local name nor an `xsd:decimal` literal can hold the `+` or the `e`. A
    # numpy scalar is converted first because its `repr` holds the type name.
    number = float(value)
    if not math.isfinite(number):
        raise InputError(
            f"{number!r} cannot be written as a number. Remove the value from "
            "the input or replace it with a finite number."
        )
    return format(Decimal(repr(number)), "f")


def attribute_value(attribute: str, value: Any) -> str:
    """Return the individual name for one value of an attribute.

    Values are qualified by their attribute, so the same literal under two
    attributes stays two individuals. The name is the attribute and the value
    joined by a dot, so two attributes whose names and values differ only in
    where the dot falls name one individual.

    A float with no fraction names the same individual as the integer.

    Raises:
        InputError: If the value is missing (`NaN`, `pd.NaT` or `pd.NA`) or
            infinite, naming the attribute it came from.
    """
    if value is pd.NaT or value is pd.NA:
        raise InputError(
            f"Attribute {attribute!r}: {value!r} is a missing value and names no "
            "individual. Remove it from the input or replace it with a value."
        )
    if isinstance(value, pd.Timestamp):
        return f"{attribute}.{timestamp_id(value)}"
    if isinstance(value, float):
        try:
            text = format_number(value)
        except ValueError as exc:
            raise InputError(f"Attribute {attribute!r}: {exc}") from exc
        return f"{attribute}.{int(value) if value.is_integer() else text}"
    return f"{attribute}.{value}"


def attribute_change_ids(
    object_ids: pd.Series, attribute: str, timestamp_ids: pd.Series
) -> pd.Series:
    """Return individual names for attribute change records.

    A name is the object id, the attribute and the timestamp id. Repeats of
    one name are numbered from 2.

    Returns:
        One name per row, aligned with the inputs.
    """
    base = (
        ATTRIBUTE_CHANGE_RECORD
        + "_"
        + object_ids.astype(str)
        + "_"
        + attribute
        + "_"
        + timestamp_ids.astype(str)
    )
    repeat = base.groupby(base).cumcount()
    return base.where(repeat == 0, base + "_" + (repeat + 1).astype(str))


CASE_CONCEPT_CLASSES: tuple[Callable[[str], str], ...] = (start_class, end_class)
CASE_CONCEPT_PROPERTIES: tuple[Callable[[str], str], ...] = (df_property, ef_property)
ATTRIBUTE_CLASSES: tuple[Callable[[str], str], ...] = (attribute_change_class,)
ATTRIBUTE_PROPERTIES: tuple[Callable[[str], str], ...] = (
    has_attribute_property,
    change_value_property,
    value_property,
)
BIN_CLASSES: tuple[Callable[[str, Any], str], ...] = (bin_class,)


def _with_inverses(classes: set[str], properties: set[str]) -> frozenset[str]:
    """Return classes and properties together with the inverse of each property."""
    return frozenset(classes | properties | {inverse(p) for p in properties})


def derived_from_object_type(obj_type: str) -> frozenset[str]:
    """Return the names, inverses included, a build derives from an object type."""
    return _with_inverses(
        {build(obj_type) for build in CASE_CONCEPT_CLASSES},
        {build(obj_type) for build in CASE_CONCEPT_PROPERTIES},
    )


def derived_from_attribute(
    attribute: str, bin_labels: Iterable[Any] = ()
) -> frozenset[str]:
    """Return the names, inverses included, a build derives from an attribute."""
    classes = {build(attribute) for build in ATTRIBUTE_CLASSES}
    classes |= {
        build(attribute, label) for build in BIN_CLASSES for label in bin_labels
    }
    return _with_inverses(classes, {build(attribute) for build in ATTRIBUTE_PROPERTIES})


def derived_from_relation(relation: str) -> frozenset[str]:
    """Return the schema names a build derives from one relation, its inverse."""
    return frozenset({inverse(relation)})


TIMESTAMP_ID_FORMAT = "%Y%m%dT%H%M%S_%f"


def timestamp_id(timestamp: pd.Timestamp | str) -> str:
    """Return the individual name for one point in time.

    A timestamp with an offset is named by its UTC time, to the microsecond.
    Text has every non-alphanumeric character replaced.
    """
    if isinstance(timestamp, pd.Timestamp):
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp.strftime(TIMESTAMP_ID_FORMAT)
    return _NON_ALNUM.sub("_", str(timestamp))


def timestamp_ids(values: pd.Series) -> pd.Series:
    """Return individual names for a column of timestamps.

    Returns:
        One name per row, aligned with the input, as `timestamp_id` names it.
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        if values.dt.tz is not None:
            values = values.dt.tz_convert("UTC")
        return values.dt.strftime(TIMESTAMP_ID_FORMAT)
    return values.apply(timestamp_id)
