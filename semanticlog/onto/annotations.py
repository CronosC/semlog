"""The human-readable text attached to an ontology.

Cleaning makes names safe at the cost of making them uglier, and an ontology
of `place_order` and `AttrChange_o_1_status_20240101T000000_000000` is hard to
read. Labels and comments put the original wording back, without letting it
into an IRI.

Two sources feed this. The fixed text for the upper schema lives in
`tbox_annotations.json` beside this module, and everything else is derived
from the log through the store backmap. All of it is optional and only written
when a build asks for it.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Iterable, Sequence
from importlib import resources
from typing import Any, Protocol

import pandas as pd

from semanticlog.core import vocabulary as vocab
from semanticlog.data.tables import EntityKind
from semanticlog.onto.settings import OntoBuildSettings

TBOX_ANNOTATION_DOCUMENT = "tbox_annotations.json"


@functools.cache
def tbox_annotations() -> dict[str, dict[str, str]]:
    """Return the fixed labels and comments of the upper schema.

    Read once from the JSON document shipped with the package, so the wording
    can be reviewed and changed without touching Python.
    """
    text = (
        resources.files("semanticlog.onto")
        .joinpath(TBOX_ANNOTATION_DOCUMENT)
        .read_text(encoding="utf-8")
    )
    document: dict[str, dict[str, str]] = json.loads(text)
    return document


_TIMESTAMP_LABEL_FORMAT = "%Y-%m-%d %H:%M:%S"


def timestamp_labels(values: pd.Series) -> pd.Series:
    """Render timestamps for display. Not safe as part of a name."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return values.dt.strftime(_TIMESTAMP_LABEL_FORMAT)
    return values.astype(str)


def _format_value(value: Any) -> str:
    """Render one attribute value as its label."""
    if isinstance(value, pd.Timestamp):
        return value.strftime(_TIMESTAMP_LABEL_FORMAT)
    return str(value)


def _declare_label(constraint: str, original: Callable[[str], str]) -> str:
    """Render a mined constraint name as its template and its activities.

    Mining writes `Response[a, b]` as `Response__a__b__`, which is safe in an
    IRI and hard to read.

    Args:
        constraint: The constraint as it was mined.
        original: Resolver for what the log called an activity.

    Returns:
        The constraint written out, such as `Response (place order, ship)`.
    """
    template, _, rest = constraint.partition("__")
    activities = [original(part) for part in rest.split("__") if part]
    if not activities:
        return template
    return f"{template} ({', '.join(activities)})"


class AnnotationSink(Protocol):
    """The part of a sink an annotation writer needs.

    Stated as a protocol so the writer can be tested against anything that
    records these four calls.
    """

    def annotate_entity(
        self, iri_fragment: str, label: str | None = ..., comment: str | None = ...
    ) -> None:
        """Attach a label and a comment to a declared class or property."""
        ...

    def annotate_individuals_bulk(self, pairs: list[tuple[str, str]]) -> None:
        """Label many individuals at once."""
        ...

    def inverse_name(self, prop: str) -> str:
        """Return the name of the inverse of a property."""
        ...

    def is_declared(self, name: str) -> bool:
        """Report whether a class or property of this name has been declared."""
        ...


def _when_enabled(fn: Callable) -> Callable:
    """Make a method do nothing unless the build asked for annotations."""

    @functools.wraps(fn)
    def wrapper(self: AnnotationWriter, *args: Any, **kwargs: Any) -> None:
        if self.enabled:
            fn(self, *args, **kwargs)

    return wrapper


class AnnotationWriter:
    """Writes the labels and comments that go with the axioms.

    Every method here does nothing unless the build asked for derived
    annotations, so emitters can call them unconditionally.
    """

    def __init__(self, sink: AnnotationSink, settings: OntoBuildSettings) -> None:
        self.sink = sink
        self.settings = settings
        self._backmap: dict[str, str] = {}

    @property
    def enabled(self) -> bool:
        """Whether this build asked for derived annotations."""
        return self.settings.derive_annotations

    @property
    def has_inverses(self) -> bool:
        """Whether inverse properties exist and so need annotating too."""
        return self.settings.has_inverses

    def bind(self, backmap: dict[str, str]) -> None:
        """Supply the map from cleaned names back to what the log called them."""
        self._backmap = backmap

    def original(self, name: str) -> str:
        """Return what the log called a name, or the name if it never changed."""
        return self._backmap.get(name, name)

    @_when_enabled
    def tbox(self) -> None:
        """Annotate the parts of the TBox this build declared."""
        for name, ann in tbox_annotations().items():
            if not self.sink.is_declared(name):
                continue
            self.sink.annotate_entity(
                name, label=ann.get("label"), comment=ann.get("comment")
            )

    @_when_enabled
    def entity_class(self, name: str, kind: EntityKind) -> None:
        """Label an activity or object type class with its name from the log."""
        orig = self.original(name)
        noun = "Events" if kind is EntityKind.EVENT else "Objects"
        self.sink.annotate_entity(name, label=orig, comment=f"{noun} of type '{orig}'.")

    @_when_enabled
    def entity_individuals(self, entity_type: str, ids: Iterable[Any]) -> None:
        """Label events or objects with their type and their id from the log.

        An entity with no id is not asserted, so it is not labelled either.
        """
        orig_type = self.original(entity_type)
        self.sink.annotate_individuals_bulk(
            [
                (str(eid), f"{orig_type} ({self.original(str(eid))})")
                for eid in ids
                if pd.notna(eid)
            ]
        )

    @_when_enabled
    def order_superproperties(self, include_df: bool, include_ef: bool) -> None:
        """Annotate the order properties that every case concept sits below."""
        if include_df:
            self._pair(
                vocab.DF,
                "directly followed by",
                "Holds between two events when one immediately precedes the other "
                "within any case. Supertype of all case-specific directly-follows "
                "properties.",
                "directly preceded by",
                "Relates an event to the event that directly precedes it within any "
                "case. Inverse of 'directly followed by'.",
            )
        if include_ef:
            self._pair(
                vocab.EF,
                "eventually followed by",
                "Holds between two events whenever one comes at or before the other "
                "within any case, including itself. Supertype of all case-specific "
                "eventually-follows properties.",
                "eventually preceded by",
                "Relates an event to any event that comes at or before it within any "
                "case, including itself. Inverse of 'eventually followed by'.",
            )

    @_when_enabled
    def trace_boundary_classes(
        self, case_concept: str, start_cls: str, end_cls: str
    ) -> None:
        """Annotate the start and end event classes of one case concept."""
        cc = self.original(case_concept)
        self.sink.annotate_entity(
            start_cls,
            label=f"Start Event (via {cc})",
            comment=f"Events with no directly-follows predecessor in the '{cc}' case trace.",
        )
        self.sink.annotate_entity(
            end_cls,
            label=f"End Event (via {cc})",
            comment=f"Events with no directly-follows successor in the '{cc}' case trace.",
        )

    @_when_enabled
    def df_property(self, case_concept: str, prop: str) -> None:
        """Annotate the directly-follows property of one case concept."""
        cc = self.original(case_concept)
        self._pair(
            prop,
            f"directly followed (via {cc}) by",
            f"Holds between two events when one immediately precedes the other in the "
            f"same '{cc}' case trace.",
            f"directly preceded (via {cc}) by",
            f"Relates an event to its direct predecessor in the same '{cc}' case trace. "
            f"Inverse of 'directly followed (via {cc}) by'.",
        )

    @_when_enabled
    def ef_property(self, case_concept: str, prop: str) -> None:
        """Annotate the eventually-follows property of one case concept."""
        cc = self.original(case_concept)
        self._pair(
            prop,
            f"eventually followed (via {cc}) by",
            f"Holds between two events whenever one comes at or before the other in the "
            f"same '{cc}' case trace, including itself.",
            f"eventually preceded (via {cc}) by",
            f"Relates an event to any event that comes at or before it in the same "
            f"'{cc}' case trace, including itself. Inverse of 'eventually followed "
            f"(via {cc}) by'.",
        )

    @_when_enabled
    def time_individuals(self, ids: pd.Series, labels: pd.Series) -> None:
        """Label time individuals with a readable timestamp."""
        self.sink.annotate_individuals_bulk(list(zip(ids, labels, strict=True)))

    @_when_enabled
    def attribute(self, attr: str, prop: str, owner: str) -> None:
        """Annotate an attribute class and the property that reaches its values."""
        orig = self.original(attr)
        self.sink.annotate_entity(
            attr, label=orig, comment=f"Attribute '{orig}' of a {owner}."
        )
        self._pair(
            prop,
            f"has {orig}",
            f"Links a {owner} to its '{orig}' attribute value.",
            f"is {orig} of",
            f"Links a '{orig}' attribute value to its owning {owner}.",
        )

    @_when_enabled
    def attribute_values(self, value_to_name: dict[Any, str]) -> None:
        """Label value individuals with the value as the log wrote it."""
        self.sink.annotate_individuals_bulk(
            [
                (name, self._backmap.get(name, _format_value(value)))
                for value, name in value_to_name.items()
            ]
        )

    @_when_enabled
    def relation(
        self, relation: str, from_types: Sequence[str], to_types: Sequence[str]
    ) -> None:
        """Annotate a relation property with the types it connects."""
        orig = self.original(relation)
        source = " or ".join(self.original(t) for t in from_types) or "event"
        target = " or ".join(self.original(t) for t in to_types)
        self._pair(
            relation,
            orig,
            f"Relates '{source}' to '{target}' via the '{orig}' relationship.",
            f"inverse of {orig}",
            f"Inverse of '{orig}': relates '{target}' back to '{source}'.",
        )

    @_when_enabled
    def declare_constraints(self, hierarchy: str, constraints: Sequence[str]) -> None:
        """Annotate the constraint hierarchy and every constraint under it.

        Args:
            hierarchy: Class the mined constraints sit under.
            constraints: The mined constraint class names.
        """
        self.sink.annotate_entity(
            hierarchy,
            label="DECLARE constraint",
            comment=(
                "A rule about the order of activities within a trace. Its members "
                "are the traces that satisfy it."
            ),
        )
        for constraint in constraints:
            written = _declare_label(constraint, self.original)
            self.sink.annotate_entity(
                constraint,
                label=written,
                comment=f"Traces satisfying the DECLARE constraint '{written}'.",
            )

    @_when_enabled
    def attribute_change_schema(
        self, attr: str, change_class: str, change_prop: str
    ) -> None:
        """Annotate the change class and change property of one attribute."""
        orig = self.original(attr)
        self.sink.annotate_entity(
            change_class,
            label=f"{orig} Change",
            comment=f"A timestamped record of the attribute '{orig}' taking a new value.",
        )
        self._pair(
            change_prop,
            f"changes {orig} to",
            f"Links an attribute change record to the new value of '{orig}'.",
            f"is new {orig} in change",
            f"Links a '{orig}' value to the attribute change record in which it became "
            f"the new value.",
        )

    @_when_enabled
    def attribute_change_individuals(
        self,
        change_ids: pd.Series,
        attr: str,
        obj_type: str,
        object_ids: pd.Series,
        timestamps: pd.Series,
    ) -> None:
        """Label change records with the attribute, the object and the moment."""
        orig_attr = self.original(attr)
        orig_type = self.original(obj_type)
        owners = orig_type + " (" + object_ids.astype(str) + ")"
        labels = (
            f"{orig_attr} Change for " + owners + " at " + timestamp_labels(timestamps)
        )
        self.sink.annotate_individuals_bulk(list(zip(change_ids, labels, strict=True)))

    def _pair(
        self,
        prop: str,
        label: str,
        comment: str,
        inverse_label: str,
        inverse_comment: str,
    ) -> None:
        """Annotate a property and its inverse, when the build has inverses."""
        self.sink.annotate_entity(prop, label=label, comment=comment)
        if self.has_inverses:
            self.sink.annotate_entity(
                self.sink.inverse_name(prop),
                label=inverse_label,
                comment=inverse_comment,
            )
