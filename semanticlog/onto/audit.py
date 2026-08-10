"""Pre-build checks on the names a build is about to turn into IRIs.

Two failures are caught here, both of which are otherwise silent.

**Collisions.** Local names are built by joining components with ``_``, and
cleaning maps every non-alphanumeric character to ``_`` while leaving existing
underscores alone. The separator is therefore also a legal character inside a
component, so distinct sources converge: attribute ``a_b`` with value ``c`` and
attribute ``a`` with value ``b_c`` both produce ``a_b_c``. owlready2 merges them
into one individual asserted as the value of both attributes.

**Unsafe characters.** The builders concatenate names into IRIs without checking
them. Skipping :func:`~semanticlog.data.cleaning.clean_oced_data` therefore does
not fail, it produces a file full of malformed IRIs.

Both run before any axiom is written, so a rejected build leaves no output.
"""

from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

from semanticlog.data.oced import OCEDStore, RelationKind
from semanticlog.onto import vocabulary as vocab
from semanticlog.onto.settings import DataRepresentation, OntoBuildSettings

#: Characters permitted in an IRI local name.
#:
#: Wider than the ``[A-Za-z0-9_]`` that :func:`clean_oced_data` guarantees for
#: *strings*, because numeric attribute values are interpolated unchanged and
#: legitimately produce ``price_10.5`` or ``price_-3.2e-05``. Those are valid IRI
#: references even though they are not valid XML QNames. What this set does
#: exclude is every character that actually breaks an IRI: whitespace, the
#: delimiters, and the QName-splitting colon.
_SAFE_LOCAL_NAME = re.compile(r"[A-Za-z0-9_.+~-]+")

#: Cap on entries listed per error, so a systematically broken log reports a
#: readable summary rather than thousands of lines.
_MAX_REPORTED = 20


class NameCollisionError(ValueError):
    """Two or more distinct sources produce the same IRI local name."""


class UnsafeLocalNameError(ValueError):
    """A name would produce a malformed IRI, usually because cleaning was skipped."""


class _Names:
    """Collects every local name a build will emit, with what claims it."""

    def __init__(self) -> None:
        self.claims: dict[str, set[str]] = defaultdict(set)
        self.unsafe: dict[str, str] = {}

    def claim(self, name: str, source: str) -> None:
        text = str(name)
        self.claims[text].add(source)
        self.check(text, source)

    def check(self, name: str, source: str) -> None:
        text = str(name)
        if not _SAFE_LOCAL_NAME.fullmatch(text) and text not in self.unsafe:
            self.unsafe[text] = source

    def collisions(self) -> dict[str, list[str]]:
        return {n: sorted(s) for n, s in self.claims.items() if len(s) > 1}


def audit_names(data: OCEDStore, settings: OntoBuildSettings) -> None:
    """Raise if the build would emit a malformed or ambiguous IRI local name.

    Args:
        data: The cleaned store the build will read.
        settings: Resolved build settings, which decide whether numeric
            attributes and their bin subclasses are emitted at all.

    Raises:
        UnsafeLocalNameError: If a name contains characters that break an IRI.
        NameCollisionError: If a name has more than one source. The message lists
            each colliding name with the sources that claim it.
    """
    names = _Names()

    _claim_schema_names(names, data, settings)
    _claim_attribute_names(names, data, settings)
    _check_entity_ids(names, data)

    if names.unsafe:
        raise UnsafeLocalNameError(_format_unsafe(names.unsafe))

    collisions = names.collisions()
    if collisions:
        raise NameCollisionError(_format_collisions(collisions))


def _claim_schema_names(
    names: _Names, data: OCEDStore, settings: OntoBuildSettings
) -> None:
    for activity in data.get_activities():
        names.claim(activity, f"activity {activity!r}")
    for obj_type in data.get_object_types():
        names.claim(obj_type, f"object type {obj_type!r}")
    for relation in data.get_relation_types(RelationKind.ALL):
        names.claim(relation, f"relation {relation!r}")

    for case_concept in settings.case_concepts:
        source = f"case concept {case_concept!r}"
        names.claim(vocab.start_class(case_concept), source)
        names.claim(vocab.end_class(case_concept), source)
        if settings.instantiate_df:
            names.claim(vocab.df_property(case_concept), source)
        if settings.instantiate_ef:
            names.claim(vocab.ef_property(case_concept), source)


def _claim_attribute_names(
    names: _Names, data: OCEDStore, settings: OntoBuildSettings
) -> None:
    for table, skip in _attribute_tables(data):
        for attribute in table.columns:
            if attribute in skip:
                continue
            column = table[attribute].dropna()
            if column.empty:
                continue
            numeric = pd.api.types.is_numeric_dtype(column)
            if numeric and not settings.data_representation:
                continue

            source = f"attribute {attribute!r}"
            names.claim(attribute, source)
            names.claim(vocab.has_attribute_property(attribute), source)
            _claim_values(names, attribute, column, numeric, settings)


def _claim_values(
    names: _Names,
    attribute: str,
    column: pd.Series,
    numeric: bool,
    settings: OntoBuildSettings,
) -> None:
    """Claim the value individuals and bin subclasses of one attribute column."""
    representations = set(settings.data_representation)
    binning = {DataRepresentation.EQUAL_BINS, DataRepresentation.DIST_BINS}

    # Numeric columns only produce value individuals under VALUES; under a
    # binning mode alone the individuals typed into the bins are the entities.
    if not numeric or DataRepresentation.VALUES in representations:
        for value in column.unique():
            names.claim(
                vocab.attribute_value(attribute, value),
                f"value {value!r} of attribute {attribute!r}",
            )

    if numeric and representations & binning:
        for label in settings.bin_labels:
            names.claim(
                vocab.bin_class(attribute, label),
                f"bin {label!r} of attribute {attribute!r}",
            )


def _check_entity_ids(names: _Names, data: OCEDStore) -> None:
    """Validate event and object identifiers without accumulating them.

    Identifiers are checked rather than claimed: duplicates across tables are
    already rejected by ``OCEDStore.update_used_ids``, and holding one set entry
    per event would cost more than the rest of the audit combined on a large log.
    Their overlap with schema names is checked directly instead.
    """
    schema_names = set(names.claims)
    for table, id_col, source in _identifier_columns(data):
        ids = table[id_col].dropna().astype(str)
        if ids.empty:
            continue

        unsafe = ids[~ids.str.fullmatch(_SAFE_LOCAL_NAME)]
        if not unsafe.empty:
            names.check(unsafe.iloc[0], source)

        for shared in schema_names.intersection(ids):
            names.claims[shared].add(source)


def _identifier_columns(data: OCEDStore) -> list[tuple[pd.DataFrame, str, str]]:
    columns = []
    for activity in data.get_activities():
        event = data.get_events(activity)
        columns.append((event.data, event.id, f"event id of activity {activity!r}"))
    for obj_type in data.get_object_types():
        obj = data.get_object(obj_type)
        columns.append((obj.data, obj.id, f"object id of type {obj_type!r}"))
    return columns


def _attribute_tables(data: OCEDStore) -> list[tuple[pd.DataFrame, set[str]]]:
    """Return each entity table with the columns that are not attributes."""
    tables = []
    for activity in data.get_activities():
        event = data.get_events(activity)
        tables.append((event.data, {event.id, event.time}))
    for obj_type in data.get_object_types():
        obj = data.get_object(obj_type)
        tables.append((obj.data, {obj.id}))
    return tables


def _format_collisions(collisions: dict[str, list[str]]) -> str:
    shown = sorted(collisions)[:_MAX_REPORTED]
    lines = [
        f"  {name!r} is claimed by: {', '.join(collisions[name])}" for name in shown
    ]
    if len(collisions) > _MAX_REPORTED:
        lines.append(f"  ... and {len(collisions) - _MAX_REPORTED} more")
    return (
        f"{len(collisions)} IRI local name(s) would be emitted for more than one "
        "thing, which silently merges them in the ontology:\n"
        + "\n".join(lines)
        + "\n\nRename the colliding source(s) in the input data before building."
    )


def _format_unsafe(unsafe: dict[str, str]) -> str:
    shown = sorted(unsafe)[:_MAX_REPORTED]
    lines = [f"  {name!r} from {unsafe[name]}" for name in shown]
    if len(unsafe) > _MAX_REPORTED:
        lines.append(f"  ... and {len(unsafe) - _MAX_REPORTED} more")
    return (
        f"{len(unsafe)} name(s) would produce a malformed IRI:\n"
        + "\n".join(lines)
        + "\n\nRun clean_oced_data(store) before building."
    )
