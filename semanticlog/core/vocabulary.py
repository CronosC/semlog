"""Every IRI a build may emit: the namespace, and every local name in it.

The upper schema names are constants, the names derived from log content are
built by the functions below, and nothing outside this module formats an
emitted name itself. Cleaning derives its reserved-name set from here, so a
name added here is defended against collisions everywhere else.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import pandas as pd

# The namespace a build puts its names in when it is given no other.
DEFAULT_IRI: str = "https://oced#"

_NON_ALNUM = re.compile(r"[^A-Za-z0-9]")


# `+` and `~` are legal in an IRI but not in an XML element name, and RDF/XML
# writes a property as an element, so a name holding one cannot be serialized.
SAFE_LOCAL_NAME = re.compile(r"[A-Za-z0-9_.-]+")


class UnsafeLocalNameError(ValueError):
    """A name cannot be used as the local part of an IRI."""


class NameCollisionError(ValueError):
    """Two different things would be emitted under one name."""


def is_safe_local_name(name: str) -> bool:
    """Report whether a name can be the local part of an IRI.

    Args:
        name: Candidate name. Non-string values are checked by their string
            form, as the emitted IRI would use.

    Returns:
        True when the whole name matches the safe character set.
    """
    return bool(SAFE_LOCAL_NAME.fullmatch(str(name)))


OBJECT = "Object"
EVENT = "Event"
ATTRIBUTE = "Attribute"
TIME = "Time"
OBJECT_ATTRIBUTE = "ObjectAttribute"
EVENT_ATTRIBUTE = "EventAttribute"
ATTRIBUTE_CHANGE = "AttributeChange"
DECLARE_CONSTRAINT = "DeclareConstraint"

OBSERVED_AT = "observed_at"
OBSERVES = "observes"
EVENT_OBJECT_RELATION = "event_object_relation"
OBJECT_OBJECT_RELATION = "object_object_relation"
HAS_ATTRIBUTE = "has_attribute"
ATTRIBUTE_OF = "attribute_of"
HAS_DATA_VALUE = "has_data_value"
HAS_ATTRIBUTE_CHANGE = "has_attribute_change"
ATTRIBUTE_CHANGE_VALUE = "attribute_change_value"
CHANGED_AT = "changed_at"
DF = "df"
EF = "ef"

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
        DF,
        EF,
    }
)

SCHEMA_PROPERTIES: frozenset[str] = _INVERTIBLE_PROPERTIES | {HAS_DATA_VALUE}


INVERSE_PREFIX = "inv_"

_EXPLICIT_INVERSES: dict[str, str] = {
    HAS_ATTRIBUTE: ATTRIBUTE_OF,
    OBSERVED_AT: OBSERVES,
}

SCHEMA_NAMED_INVERSES: dict[str, str] = {
    **_EXPLICIT_INVERSES,
    **{inverse: prop for prop, inverse in _EXPLICIT_INVERSES.items()},
}


def inverse(prop: str, named: dict[str, str] | None = None) -> str:
    """Return the name of the inverse of a property.

    A few schema properties have a readable inverse of their own, such as
    `observes` for `observed_at`. Everything else takes the `inv_` prefix.

    Args:
        prop: Property whose inverse is wanted.
        named: Known inverse pairs to consult instead of the schema ones. A
            build passes its own map here so that inverses named during the
            run are found as well.

    Returns:
        The inverse property name.
    """
    pairs = SCHEMA_NAMED_INVERSES if named is None else named
    return pairs.get(prop, INVERSE_PREFIX + prop)


RESERVED_NAMES: frozenset[str] = (
    SCHEMA_CLASSES
    | SCHEMA_PROPERTIES
    | frozenset(inverse(p) for p in _INVERTIBLE_PROPERTIES)
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


def attribute_change_class(attribute: str) -> str:
    """Return the class name for changes of one attribute."""
    return f"{attribute}Change"


def bin_class(attribute: str, label: Any) -> str:
    """Return the class name for one bin of a binned numeric attribute."""
    return f"{attribute}_bin_{label}"


def attribute_value(attribute: str, value: Any) -> str:
    """Return the individual name for one value of an attribute.

    Values are qualified by their attribute, so the same literal under two
    attributes stays two individuals.

    Args:
        attribute: Attribute the value belongs to.
        value: The value itself. Timestamps are converted to their stable id
            form first, because their text form is not a safe local name.

    Returns:
        The individual name.
    """
    if isinstance(value, pd.Timestamp):
        return f"{attribute}_{timestamp_id(value)}"
    return f"{attribute}_{value}"


def attribute_change_ids(
    object_ids: pd.Series, attribute: str, timestamp_ids: pd.Series
) -> pd.Series:
    """Return individual names for attribute change records.

    An object, an attribute and a moment name one change. A log that records
    two changes of one attribute at one instant would otherwise put both new
    values on one record, so the second and later of them are numbered.

    Args:
        object_ids: Object each change belongs to.
        attribute: Attribute that changed.
        timestamp_ids: Change times, already in their id form.

    Returns:
        One name per row, aligned with the inputs.
    """
    base = (
        "AttrChange_"
        + object_ids.astype(str)
        + "_"
        + attribute
        + "_"
        + timestamp_ids.astype(str)
    )
    repeat = base.groupby(base).cumcount()
    return base.where(repeat == 0, base + "_" + (repeat + 1).astype(str))


def derived_schema_names(
    object_types: Iterable[str],
    attributes: Iterable[str],
    bin_labels: Iterable[str] = (),
) -> frozenset[str]:
    """Return the schema names a build would derive from the given log content.

    Cleaning uses this to rename anything in the input that would otherwise
    collide with a name the build is about to generate for itself.

    Args:
        object_types: Object types that could serve as case concepts.
        attributes: Attribute names found on events and objects.
        bin_labels: Bin labels the build will use, if it bins numbers. The
            labels are not derivable from the log, so a caller that knows the
            settings passes them.

    Returns:
        Every derived class and property name, including their inverses.
    """
    classes: set[str] = set()
    properties: set[str] = set()
    labels = list(bin_labels)

    for obj_type in object_types:
        classes.update({start_class(obj_type), end_class(obj_type)})
        properties.update({df_property(obj_type), ef_property(obj_type)})

    for attribute in attributes:
        classes.add(attribute_change_class(attribute))
        classes.update(bin_class(attribute, label) for label in labels)
        properties.update(
            {has_attribute_property(attribute), change_value_property(attribute)}
        )

    return frozenset(classes | properties | {inverse(p) for p in properties})


TIMESTAMP_ID_FORMAT = "%Y%m%dT%H%M%S_%f"


def timestamp_id(timestamp: pd.Timestamp | str) -> str:
    """Return the individual name for one point in time.

    Args:
        timestamp: A pandas timestamp, or text that is already a timestamp in
            some other format.

    Returns:
        A name safe to use in an IRI. Text input keeps its content with every
        non-alphanumeric character replaced.
    """
    if isinstance(timestamp, pd.Timestamp):
        return timestamp.strftime(TIMESTAMP_ID_FORMAT)
    return _NON_ALNUM.sub("_", str(timestamp))


def timestamp_ids(values: pd.Series) -> pd.Series:
    """Return individual names for a column of timestamps.

    Args:
        values: Timestamps to name. A datetime column is formatted directly,
            anything else falls back to per-value conversion.

    Returns:
        One name per row, aligned with the input.
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        return values.dt.strftime(TIMESTAMP_ID_FORMAT)
    return values.apply(timestamp_id)
