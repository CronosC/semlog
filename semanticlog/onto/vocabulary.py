"""Every IRI local name the builders emit, in one place.

The ontology file is the product, so its naming scheme is a public contract that
downstream projects depend on.  Keeping the fixed schema names and the derived
name templates together means that contract can be read, reviewed and changed in
one file instead of being reconstructed from string concatenation spread across
the builders.

:data:`RESERVED_NAMES` is the set of fixed names a build can emit.  It is derived
from the constants below rather than restated, so
:func:`~semanticlog.data.cleaning.fix_duplicate_names` cannot drift out of sync
with what the builders actually write.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import pandas as pd

_NON_ALNUM = re.compile(r"[^A-Za-z0-9]")

# ----------------------------------------------------------------------
# Fixed schema names (upper ontology)
# ----------------------------------------------------------------------

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

# Schema object properties that are created with an inverse when
# ``InverseRepresentation`` is active.
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

# ----------------------------------------------------------------------
# Inverses
# ----------------------------------------------------------------------

INVERSE_PREFIX = "inv_"

# Schema properties whose inverse reads better under its own name than under the
# generated ``inv_{prop}`` form.
_EXPLICIT_INVERSES: dict[str, str] = {
    HAS_ATTRIBUTE: ATTRIBUTE_OF,
    OBSERVED_AT: OBSERVES,
}

#: Named inverse pairs of the upper ontology, in both directions.  Seeds
#: ``OCEDBaseBuilder._named_inverse_map`` so the TBox and ABox agree on the name
#: even when no ``InverseObjectProperties`` axiom is written to recover it from.
SCHEMA_NAMED_INVERSES: dict[str, str] = {
    **_EXPLICIT_INVERSES,
    **{inverse: prop for prop, inverse in _EXPLICIT_INVERSES.items()},
}


def inverse(prop: str, named: dict[str, str] | None = None) -> str:
    """Return the inverse property name for ``prop``.

    Args:
        prop: Property to invert.
        named: Known inverse pairs taking precedence over the generated form.
            Defaults to the upper ontology's pairs.

    Returns:
        The explicitly named inverse when one is known, otherwise ``inv_{prop}``.
    """
    pairs = SCHEMA_NAMED_INVERSES if named is None else named
    return pairs.get(prop, INVERSE_PREFIX + prop)


#: Fixed names a build can emit.  Data names colliding with these are renamed by
#: :func:`~semanticlog.data.cleaning.fix_duplicate_names`.
RESERVED_NAMES: frozenset[str] = (
    SCHEMA_CLASSES
    | SCHEMA_PROPERTIES
    | frozenset(inverse(p) for p in _INVERTIBLE_PROPERTIES)
)

# ----------------------------------------------------------------------
# Derived names
# ----------------------------------------------------------------------


def start_class(case_concept: str) -> str:
    """Return the class of events that open a trace of ``case_concept``."""
    return f"StartEvent_{case_concept}"


def end_class(case_concept: str) -> str:
    """Return the class of events that close a trace of ``case_concept``."""
    return f"EndEvent_{case_concept}"


def df_property(case_concept: str) -> str:
    """Return the directly-follows property for ``case_concept``."""
    return f"{DF}_{case_concept}"


def ef_property(case_concept: str) -> str:
    """Return the eventually-follows property for ``case_concept``."""
    return f"{EF}_{case_concept}"


def has_attribute_property(attribute: str) -> str:
    """Return the ``has_attribute`` sub-property for ``attribute``."""
    return f"has_{attribute}"


def change_value_property(attribute: str) -> str:
    """Return the ``attribute_change_value`` sub-property for ``attribute``."""
    return f"change_value_{attribute}"


def attribute_change_class(attribute: str) -> str:
    """Return the ``AttributeChange`` subclass recording changes to ``attribute``."""
    return f"{attribute}Change"


def bin_class(attribute: str, label: Any) -> str:
    """Return the subclass of ``attribute`` holding values in bin ``label``."""
    return f"{attribute}_{label}"


def attribute_value(attribute: str, value: Any) -> str:
    """Return the individual representing ``value`` of ``attribute``.

    Timestamps are rendered via :func:`timestamp_id`; every other value is
    interpolated as-is, so the caller is responsible for having cleaned it.
    """
    if isinstance(value, pd.Timestamp):
        return f"{attribute}_{timestamp_id(value)}"
    return f"{attribute}_{value}"


def attribute_change_ids(
    object_ids: pd.Series, attribute: str, timestamp_ids: pd.Series
) -> pd.Series:
    """Return one ``AttributeChange`` individual name per change record."""
    return (
        "AttrChange_"
        + object_ids.astype(str)
        + "_"
        + attribute
        + "_"
        + timestamp_ids.astype(str)
    )


def derived_schema_names(
    object_types: Iterable[str], attributes: Iterable[str]
) -> frozenset[str]:
    """Return every schema-level name a build can *derive* from these data names.

    :data:`RESERVED_NAMES` covers the names a build always emits.  These are the
    ones it emits only because of what the data is called, and they collide with
    data names just as readily: an activity called ``has_price`` claims the same
    IRI as the property derived from an attribute called ``price``.

    Case-concept names are derived from every object type rather than from the
    configured case concepts, because cleaning runs once per store while the same
    store may be built under several configurations.  Reserving the wider set
    keeps cleaned names independent of build settings.

    Args:
        object_types: Object type names, any of which may become a case concept.
        attributes: Attribute column names across all events and objects.

    Returns:
        The derived names, including the inverse of each derived property, since
        whether inverses are emitted is a build setting cleaning cannot see.
    """
    classes: set[str] = set()
    properties: set[str] = set()

    for obj_type in object_types:
        classes.update({start_class(obj_type), end_class(obj_type)})
        properties.update({df_property(obj_type), ef_property(obj_type)})

    for attribute in attributes:
        classes.add(attribute_change_class(attribute))
        properties.update(
            {has_attribute_property(attribute), change_value_property(attribute)}
        )

    return frozenset(classes | properties | {inverse(p) for p in properties})


# ----------------------------------------------------------------------
# Timestamps
# ----------------------------------------------------------------------

#: Microseconds are included so sub-second events keep distinct IRIs.
TIMESTAMP_ID_FORMAT = "%Y%m%dT%H%M%S_%f"


def timestamp_id(timestamp: pd.Timestamp | str) -> str:
    """Convert a single timestamp to an IRI-safe local name.

    Args:
        timestamp: A pandas ``Timestamp``, or any other value to be stringified.

    Returns:
        ``YYYYMMDDTHHmmSS_ffffff`` for a ``Timestamp``, otherwise the string form
        with every non-alphanumeric character replaced by ``_``.
    """
    if isinstance(timestamp, pd.Timestamp):
        return timestamp.strftime(TIMESTAMP_ID_FORMAT)
    return _NON_ALNUM.sub("_", str(timestamp))


def timestamp_ids(values: pd.Series) -> pd.Series:
    """Convert a timestamp column to IRI-safe local names."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return values.dt.strftime(TIMESTAMP_ID_FORMAT)
    return values.apply(timestamp_id)
