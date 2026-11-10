"""The labels and comments attached to an ontology.

`AnnotationWriter` writes the fixed text of the upper schema from
`tbox_annotations.json`, and fills the templates of `derived_annotations.json`
with the names the log used. It writes only when `derive_annotations` is on.
"""

from __future__ import annotations

import functools
import json
import math
import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from importlib import resources
from typing import Any, Protocol

import pandas as pd

from semlog.core import vocabulary as vocab
from semlog.core.enums import IndividualKind
from semlog.data.cleaning.renames import VALUE_KIND, RenameKind
from semlog.data.declare import template_name
from semlog.data.tables import EntityKind
from semlog.onto.settings import OntoBuildSettings

TBOX_ANNOTATION_DOCUMENT = "tbox_annotations.json"
DERIVED_ANNOTATION_DOCUMENT = "derived_annotations.json"

_FIELD = re.compile(r"\{(\w+)\}")

_TYPE_KIND = {
    EntityKind.EVENT: RenameKind.ACTIVITY,
    EntityKind.OBJECT: RenameKind.OBJECT_TYPE,
}


def _document(name: str) -> dict[str, dict[str, str]]:
    """Read one annotation document shipped with the package."""
    text = resources.files("semlog.onto").joinpath(name).read_text(encoding="utf-8")
    document: dict[str, dict[str, str]] = json.loads(text)
    return document


@functools.cache
def tbox_annotations() -> dict[str, dict[str, str]]:
    """Return the fixed labels and comments of the upper schema, keyed by name."""
    return _document(TBOX_ANNOTATION_DOCUMENT)


@functools.cache
def derived_annotations() -> dict[str, dict[str, str]]:
    """Return the templates for labels and comments derived from the log.

    Each entry holds a `label`, and a `comment` unless it is for individuals.
    An entry for a property with an inverse also holds `inverse_label` and
    `inverse_comment`.
    """
    return _document(DERIVED_ANNOTATION_DOCUMENT)


def _entry(template: str) -> dict[str, str]:
    """Return one template.

    Raises:
        ValueError: If no template of that name is in the document.
    """
    try:
        return derived_annotations()[template]
    except KeyError:
        raise ValueError(
            f"{template!r} is not an annotation template. "
            f"{DERIVED_ANNOTATION_DOCUMENT} holds "
            f"{', '.join(sorted(derived_annotations()))}."
        ) from None


def _field(fields: Mapping[str, Any], name: str, template: str) -> Any:
    """Return one field of a template.

    Raises:
        ValueError: If the caller supplied no value for it.
    """
    if name not in fields:
        raise ValueError(
            f"The {template!r} annotation template names {name!r}, which was "
            f"not supplied. It was given {sorted(fields)}."
        )
    return fields[name]


def _fill(text: str, template: str, fields: Mapping[str, Any]) -> str:
    """Fill one template with the names a caller supplied."""
    return _FIELD.sub(lambda m: str(_field(fields, m.group(1), template)), text)


def _fill_series(text: str, template: str, fields: Mapping[str, Any]) -> pd.Series:
    """Fill one template where a field holds a value per individual.

    Raises:
        ValueError: If no field holds a series.
    """
    pieces: list[Any] = []
    at = 0
    for match in _FIELD.finditer(text):
        pieces.append(text[at : match.start()])
        pieces.append(_field(fields, match.group(1), template))
        at = match.end()
    pieces.append(text[at:])

    length = next((len(p) for p in pieces if isinstance(p, pd.Series)), None)
    if length is None:
        raise ValueError(
            f"The {template!r} annotation template was given no series. Pass "
            "at least one field as a series."
        )
    labels = pd.Series([""] * length, dtype="object")
    for piece in pieces:
        if isinstance(piece, pd.Series):
            labels = labels + piece.astype(str).reset_index(drop=True)
        elif piece:
            labels = labels + piece
    return labels


_TIMESTAMP_LABEL_FORMAT = "%Y-%m-%d %H:%M:%S"
_SUB_SECOND_LABEL_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def timestamp_labels(values: pd.Series) -> pd.Series:
    """Return timestamps as display labels, which are not safe as names.

    A fraction of a second is kept where there is one, so two timestamps
    within one second get different labels.
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        whole = values.dt.microsecond == 0
        return values.dt.strftime(_SUB_SECOND_LABEL_FORMAT).where(
            ~whole, values.dt.strftime(_TIMESTAMP_LABEL_FORMAT)
        )
    return values.astype(str)


def _format_value(value: Any) -> str:
    """Return the label of one attribute value."""
    if isinstance(value, pd.Timestamp):
        if value.microsecond:
            return value.strftime(_SUB_SECOND_LABEL_FORMAT)
        return value.strftime(_TIMESTAMP_LABEL_FORMAT)
    return str(value)


def _joined(names: Sequence[str]) -> str:
    """Return the names as a list in prose: `a`, `a and b`, `a, b and c`."""
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _split_activities(rest: str, known: Collection[str]) -> list[str]:
    """Split the activity part of a mined constraint name.

    Activities are joined with `__`. A cleaned activity name can contain
    `__`, so known activity names are matched whole first, longest first. The
    separator is used only where no known name matches.

    Args:
        rest: Everything after the template name.
        known: Activity names the log holds, as cleaning left them.

    Returns:
        The activities, in the order they appear.
    """
    parts: list[str] = []
    while rest:
        match = max(
            (a for a in known if a and (rest == a or rest.startswith(a + "__"))),
            key=len,
            default="",
        )
        if match:
            parts.append(match)
            rest = rest[len(match) :].removeprefix("__")
            continue
        head, separator, rest = rest.partition("__")
        if head:
            parts.append(head)
        if not separator:
            break
    return parts


def split_constraint(constraint: str, known: Collection[str]) -> tuple[str, list[str]]:
    """Return the template of a mined constraint name and its activities.

    Mining writes `Response[a, b]` as `Response__a__b__`.

    Args:
        constraint: The constraint as it was mined.
        known: Activity names of the log, used to split the name.

    Returns:
        The template without spaces, and the activities in order.
    """
    template, _, rest = constraint.partition("__")
    return template, _split_activities(rest, known)


def _declare_label(
    constraint: str, original: Callable[[str], str], known: Collection[str]
) -> str:
    """Return a mined constraint name as its template and its activities.

    Args:
        constraint: The constraint as it was mined.
        original: Resolver for what the log called an activity.
        known: Activity names of the log, used to split the name.

    Returns:
        The constraint written out with the template as Declare4Py spells
        it, such as `Alternate Response (place order, pay)`.
    """
    compact, parts = split_constraint(constraint, known)
    template = template_name(compact)
    activities = [original(part) for part in parts]
    if not activities:
        return template
    return f"{template} ({', '.join(activities)})"


class AnnotationSink(Protocol):
    """The sink methods an annotation writer calls."""

    def annotate_entity(
        self, iri_fragment: str, label: str | None = ..., comment: str | None = ...
    ) -> None:
        """Attach a label and a comment to a declared class or property."""
        ...

    def annotate_individuals_bulk(
        self,
        pairs: list[tuple[str, str]],
        kind: IndividualKind,
        attribute: str | None = ...,
    ) -> None:
        """Label many individuals of one kind at once."""
        ...

    def inverse_name(self, prop: str) -> str:
        """Return the inverse of a property, including inverses named in this build."""
        ...

    def is_declared(self, name: str) -> bool:
        """Whether a class or property of this name has been declared."""
        ...


class AnnotationWriter:
    """Writes the labels and comments that go with the axioms.

    Every method does nothing unless `derive_annotations` is on.

    Args:
        sink: Where the annotations go.
        settings: Settings of the build.
        backmap: Cleaned name to the name in the log, one map per kind of
            name as `OCEDStore.backmap` holds them.
        document: Entity name to the label and comment that replace the
            derived ones.
    """

    def __init__(
        self,
        sink: AnnotationSink,
        settings: OntoBuildSettings,
        backmap: Mapping[str, Mapping[str, str]],
        document: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        self.sink = sink
        self.settings = settings
        self._backmap = backmap
        self._document = document or {}

    @property
    def enabled(self) -> bool:
        """Whether this build asked for derived annotations."""
        return self.settings.derive_annotations

    @property
    def has_inverses(self) -> bool:
        """Whether the build declares inverse properties."""
        return self.settings.has_inverses

    def original(self, name: str, kind: RenameKind) -> str:
        """Return a name of one kind as the log wrote it.

        A name cleaning did not change is returned as is.
        """
        return self._backmap.get(kind.value, {}).get(name, name)

    def _original_id(self, name: str) -> str:
        return self.original(name, RenameKind.ID)

    def _original_activity(self, name: str) -> str:
        return self.original(name, RenameKind.ACTIVITY)

    def write(self, template: str, name: str, **fields: Any) -> None:
        """Annotate one class or property, and its inverse where it has one.

        A label or comment the annotation document gives for `name` or for its
        inverse is written in place of the derived one.

        Args:
            template: Entry of `derived_annotations.json` to fill.
            name: Local name of the entity to annotate.
            **fields: A value for each `{}` field the entry names.

        Raises:
            ValueError: If the template is not in the document, or names a
                field that was not supplied.
        """
        if not self.enabled:
            return
        entry = _entry(template)
        self._annotate(name, entry["label"], entry["comment"], template, fields)
        if "inverse_label" not in entry or not self.has_inverses:
            return
        self._annotate(
            self.sink.inverse_name(name),
            entry["inverse_label"],
            entry["inverse_comment"],
            template,
            fields,
        )

    def _annotate(
        self,
        name: str,
        label: str,
        comment: str,
        template: str,
        fields: Mapping[str, Any],
    ) -> None:
        """Annotate one entity with the document's texts, or else the filled ones."""
        given = {} if vocab.is_tbox(name) else self._document.get(name, {})
        self.sink.annotate_entity(
            name,
            label=given.get("label") or _fill(label, template, fields),
            comment=given.get("comment") or _fill(comment, template, fields),
        )

    def label_series(
        self, template: str, names: pd.Series, kind: IndividualKind, **fields: Any
    ) -> None:
        """Label many individuals of one kind at once, from one template.

        Args:
            template: Entry to fill. Only its label is used.
            names: Local name of each individual.
            kind: The kind of the individuals.
            **fields: A string, or a series holding one value per individual.

        Raises:
            ValueError: If the template is not in the document, names a field
                that was not supplied, or was given no series.
        """
        if not self.enabled:
            return
        labels = _fill_series(_entry(template)["label"], template, fields)
        self.sink.annotate_individuals_bulk(list(zip(names, labels, strict=True)), kind)

    def tbox(self) -> None:
        """Annotate the parts of the TBox this build declared."""
        if not self.enabled:
            return
        for key, ann in tbox_annotations().items():
            name = vocab.Term(key)
            if not self.sink.is_declared(name):
                continue
            self.sink.annotate_entity(
                name, label=ann.get("label"), comment=ann.get("comment")
            )

    def entity_class(self, name: str, kind: EntityKind) -> None:
        """Label an activity or object type class with its name from the log."""
        original = self.original(name, _TYPE_KIND[kind])
        if kind is EntityKind.EVENT:
            self.write("entity.event", name, original=original)
        else:
            self.write("entity.object", name, original=original)

    def entity_individuals(
        self, entity_type: str, ids: Iterable[Any], kind: EntityKind
    ) -> None:
        """Label events or objects with their type and their id from the log.

        An entity with no id is skipped.
        """
        if not self.enabled:
            return
        kept = pd.Series([str(eid) for eid in ids if pd.notna(eid)], dtype="object")
        self.label_series(
            "entity.individual",
            kept,
            kind.individual,
            type=self.original(entity_type, _TYPE_KIND[kind]),
            id=kept.map(self._original_id),
        )

    def order_superproperties(self, include_df: bool, include_ef: bool) -> None:
        """Annotate the `df` and `ef` super-properties."""
        if include_df:
            self.write("order.df", vocab.DF)
        if include_ef:
            self.write("order.ef", vocab.EF)

    def trace_boundary_classes(
        self, case_concept: str, start_cls: str, end_cls: str
    ) -> None:
        """Annotate the start and end event classes of one case concept."""
        case = self.original(case_concept, RenameKind.OBJECT_TYPE)
        self.write("order.start", start_cls, case=case)
        self.write("order.end", end_cls, case=case)

    def df_property(self, case_concept: str, prop: str) -> None:
        """Annotate the directly-follows property of one case concept."""
        self.write(
            "order.df_case",
            prop,
            case=self.original(case_concept, RenameKind.OBJECT_TYPE),
        )

    def ef_property(self, case_concept: str, prop: str) -> None:
        """Annotate the eventually-follows property of one case concept."""
        self.write(
            "order.ef_case",
            prop,
            case=self.original(case_concept, RenameKind.OBJECT_TYPE),
        )

    def time_individuals(self, ids: pd.Series, labels: pd.Series) -> None:
        """Label time individuals with a readable timestamp."""
        if not self.enabled:
            return
        self.sink.annotate_individuals_bulk(
            list(zip(ids, labels, strict=True)), IndividualKind.TIME
        )

    def attribute(
        self, attr: str, prop: str, owners: Mapping[EntityKind, Sequence[str]]
    ) -> None:
        """Annotate an attribute class and its `has_` property.

        Args:
            attr: Attribute class name.
            prop: The `has_` property of the attribute.
            owners: The types carrying the attribute, per kind.
        """
        fields = self._attribute_fields(attr, owners)
        self.write("attribute.class", attr, **fields)
        self.write("attribute.property", prop, **fields)

    def attribute_value_property(
        self, attr: str, prop: str, owners: Mapping[EntityKind, Sequence[str]]
    ) -> None:
        """Annotate the data property carrying the literal of a value of `attr`."""
        self.write("attribute.value", prop, **self._attribute_fields(attr, owners))

    def bin_class(
        self,
        bin_cls: str,
        attr: str,
        owners: Mapping[EntityKind, Sequence[str]],
        label: str,
        bounds: tuple[float, float],
    ) -> None:
        """Annotate one bin class of a numeric attribute with the values it holds.

        Args:
            bin_cls: The bin class name.
            attr: Attribute class the bin is below.
            owners: The types carrying the attribute, per kind.
            label: The bin label, such as `very_high`.
            bounds: The lower and upper edge of the bin. The lowest bin has
                -inf below it and the highest inf above it.
        """
        low, high = bounds
        fields = {
            **self._attribute_fields(attr, owners),
            "bin": label.replace("_", " "),
            "low": repr(float(low)),
            "high": repr(float(high)),
        }
        if math.isinf(low):
            self.write("attribute.bin_lowest", bin_cls, **fields)
        elif math.isinf(high):
            self.write("attribute.bin_highest", bin_cls, **fields)
        else:
            self.write("attribute.bin", bin_cls, **fields)

    def _attribute_fields(
        self, attr: str, owners: Mapping[EntityKind, Sequence[str]]
    ) -> dict[str, str]:
        """Return the template fields that name an attribute and the types carrying it."""
        types = [
            self.original(t, _TYPE_KIND[kind])
            for kind in EntityKind
            for t in owners.get(kind, [])
        ]
        phrases = [
            f"{_joined([self.original(t, _TYPE_KIND[kind]) for t in owners[kind]])} "
            f"{kind.value}s"
            for kind in EntityKind
            if owners.get(kind)
        ]
        return {
            "original": self.original(attr, RenameKind.ATTRIBUTE),
            "types": ", ".join(types),
            "owners": " and ".join(phrases),
        }

    def attribute_values(self, attr: str, value_to_name: dict[Any, str]) -> None:
        """Label the value individuals of one attribute with the value as the log wrote it."""
        if not self.enabled:
            return
        values = self._backmap.get(VALUE_KIND, {})
        self.sink.annotate_individuals_bulk(
            [
                (name, values.get(name, _format_value(value)))
                for value, name in value_to_name.items()
            ],
            IndividualKind.VALUE,
            attr,
        )

    def relation(
        self,
        relation: str,
        from_types: Sequence[str],
        to_types: Sequence[str],
        source_kind: EntityKind = EntityKind.OBJECT,
    ) -> None:
        """Annotate a relation property with the types it connects.

        Args:
            relation: The relation property.
            from_types: The activities or object types of its domain.
            to_types: The object types of its range.
            source_kind: The kind of type `from_types` names.
        """
        self.write(
            "relation",
            relation,
            original=self.original(relation, RenameKind.RELATION),
            source=" or ".join(
                self.original(t, _TYPE_KIND[source_kind]) for t in from_types
            )
            or "event",
            target=" or ".join(
                self.original(t, RenameKind.OBJECT_TYPE) for t in to_types
            )
            or "object",
        )

    def declare_constraints(
        self,
        hierarchy: str,
        constraints: Sequence[str],
        activities: Collection[str] = (),
    ) -> None:
        """Annotate the constraint hierarchy and every constraint under it.

        Args:
            hierarchy: Class the mined constraints sit under.
            constraints: The mined constraint class names.
            activities: Activity names of the log, used to split each
                constraint name.
        """
        self.write("declare.hierarchy", hierarchy)
        self.write("declare.template", vocab.TEMPLATE)
        self.write("declare.activation", vocab.ACTIVATION)
        self.write("declare.target", vocab.TARGET)
        for constraint in constraints:
            self.write(
                "declare.constraint",
                constraint,
                constraint=_declare_label(
                    constraint, self._original_activity, activities
                ),
            )

    def attribute_change_schema(
        self, attr: str, change_class: str, change_prop: str | None = None
    ) -> None:
        """Annotate the change class of one attribute, and its change property if given."""
        original = self.original(attr, RenameKind.ATTRIBUTE)
        self.write("change.class", change_class, original=original)
        if change_prop is not None:
            self.write("change.property", change_prop, original=original)

    def attribute_change_individuals(
        self,
        change_ids: pd.Series,
        attr: str,
        obj_type: str,
        object_ids: pd.Series,
        timestamps: pd.Series,
    ) -> None:
        """Label change records with the attribute, the object and the timestamp."""
        self.label_series(
            "change.individual",
            change_ids,
            IndividualKind.CHANGE,
            attribute=self.original(attr, RenameKind.ATTRIBUTE),
            type=self.original(obj_type, RenameKind.OBJECT_TYPE),
            id=object_ids.astype(str).map(self._original_id),
            time=timestamp_labels(timestamps),
        )
