"""All human-facing ``rdfs:label`` and ``rdfs:comment`` text emitted by a build.

The builders call an :class:`AnnotationWriter` unconditionally; the writer itself
decides whether anything is emitted, so axiom construction never branches on
``settings.derive_annotations``.  Keeping every template in one module means the
wording of the ontology can be reviewed without reading the build logic.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol

import pandas as pd

from semanticlog.onto.settings import OntoBuildSettings

# Static schema-level annotations for the upper ontology (TBox).
TBOX_ANNOTATIONS: dict[str, dict[str, str]] = {
    "Object": {
        "label": "Object",
        "comment": "An entity that participates in the process, such as an order, "
        "invoice, or product. Each object is of exactly one object type "
        "and may carry attributes whose values can change over time.",
    },
    "Event": {
        "label": "Event",
        "comment": "An atomic, timestamped occurrence of an activity in the process, "
        "such as creating an order or inserting a payment. Each event is of "
        "exactly one event type and may relate to any number of objects.",
    },
    "Attribute": {
        "label": "Attribute",
        "comment": "A named, typed value associated with an event or object.",
    },
    "Time": {
        "label": "Time point",
        "comment": "A point in time, used as an event timestamp or to date an object "
        "attribute value.",
    },
    "ObjectAttribute": {
        "label": "Object attribute",
        "comment": "An attribute defined for an object type. Its value is recorded "
        "with a timestamp and may change over the object's lifetime.",
    },
    "EventAttribute": {
        "label": "Event attribute",
        "comment": "An attribute defined for an event type, carrying a single value "
        "fixed at the time of the event.",
    },
    "AttributeChange": {
        "label": "Object attribute change",
        "comment": "A timestamped record of an object attribute taking a new value, "
        "capturing the history of dynamic object attribute values.",
    },
    "observed_at": {
        "label": "observed at",
        "comment": "Relates an event to the time point of its timestamp.",
    },
    "observes": {
        "label": "observes",
        "comment": "Inverse of observed at: relates a time point to the events "
        "timestamped at it.",
    },
    "event_object_relation": {
        "label": "involves object",
        "comment": "A qualified event-to-object relationship, associating an event "
        "with an object that takes part in it.",
    },
    "object_object_relation": {
        "label": "is related to object",
        "comment": "A qualified object-to-object relationship, associating an object "
        "with another object independently of any event.",
    },
    "has_attribute": {
        "label": "has attribute",
        "comment": "Relates an event or object to one of its attributes.",
    },
    "attribute_of": {
        "label": "is attribute of",
        "comment": "Inverse of has attribute: relates an attribute to its owning "
        "event or object.",
    },
    "has_data_value": {
        "label": "has data value",
        "comment": "Attaches a literal value to an attribute, attribute change, or "
        "time individual.",
    },
    "inv_event_object_relation": {
        "label": "is object involved in",
        "comment": "Inverse of involves object: relates an object to the events it "
        "participates in.",
    },
    "inv_object_object_relation": {
        "label": "is related from object",
        "comment": "Inverse of is related to object: the symmetric counterpart of an "
        "object-to-object relationship.",
    },
    "has_attribute_change": {
        "label": "has attribute change",
        "comment": "Relates an object to a record of one of its attribute changes.",
    },
    "inv_has_attribute_change": {
        "label": "is attribute change of",
        "comment": "Inverse of has attribute change: relates an attribute change record "
        "to the object whose attribute it tracks.",
    },
    "attribute_change_value": {
        "label": "changes value to",
        "comment": "Relates an attribute change record to the value the attribute "
        "took at that change.",
    },
    "inv_attribute_change_value": {
        "label": "is new value in change",
        "comment": "Inverse of changes value to: relates an attribute value to the "
        "change record in which it became the new value.",
    },
    "changed_at": {
        "label": "changed at",
        "comment": "Relates an attribute change record to the time point at which the "
        "change took effect.",
    },
    "inv_changed_at": {
        "label": "is change time of",
        "comment": "Inverse of changed at: relates a time point to the attribute change "
        "records that took effect at it.",
    },
}

_TIMESTAMP_LABEL_FORMAT = "%Y-%m-%d %H:%M:%S"


def timestamp_labels(values: pd.Series) -> pd.Series:
    """Format a timestamp column as human-readable labels, falling back to str()."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return values.dt.strftime(_TIMESTAMP_LABEL_FORMAT)
    return values.astype(str)


def _format_value(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime(_TIMESTAMP_LABEL_FORMAT)
    return str(value)


class AnnotationSink(Protocol):
    """The two axiom-emitting calls an annotation writer needs from a builder."""

    def annotate_entity(
        self, iri_fragment: str, label: str | None = ..., comment: str | None = ...
    ) -> None: ...

    def annotate_individuals_bulk(self, pairs: list[tuple[str, str]]) -> None: ...

    def generate_inverse_property_name(self, prop: str) -> str: ...


def _when_enabled(fn: Callable) -> Callable:
    """Skip the decorated method (and the label building it does) when disabled."""

    @functools.wraps(fn)
    def wrapper(self: AnnotationWriter, *args: Any, **kwargs: Any) -> None:
        if self.enabled:
            fn(self, *args, **kwargs)

    return wrapper


class AnnotationWriter:
    """Writes derived labels and comments for the entities an ABox build creates.

    Args:
        sink: The builder emitting the annotation axioms.
        settings: Build settings; ``derive_annotations`` and ``has_inverses`` are read
            live, so mutating them between build steps takes effect immediately.
    """

    def __init__(self, sink: AnnotationSink, settings: OntoBuildSettings) -> None:
        self.sink = sink
        self.settings = settings
        self._backmap: dict[str, str] = {}

    @property
    def enabled(self) -> bool:
        """True if derived annotations should be emitted at all."""
        return self.settings.derive_annotations

    @property
    def has_inverses(self) -> bool:
        """True if named inverse properties exist and should be annotated too."""
        return self.settings.has_inverses

    def bind(self, backmap: dict[str, str]) -> None:
        """Point the writer at the cleaned-to-original name map of the store being built."""
        self._backmap = backmap

    def original(self, name: str) -> str:
        """Return the pre-cleaning form of ``name``, or ``name`` itself if unchanged."""
        return self._backmap.get(name, name)

    # ------------------------------------------------------------------
    # Schema (TBox)
    # ------------------------------------------------------------------

    @_when_enabled
    def upper_ontology(self) -> None:
        for name, ann in TBOX_ANNOTATIONS.items():
            self.sink.annotate_entity(
                name, label=ann.get("label"), comment=ann.get("comment")
            )

    # ------------------------------------------------------------------
    # Events and objects
    # ------------------------------------------------------------------

    @_when_enabled
    def activity_class(self, activity: str) -> None:
        orig = self.original(activity)
        self.sink.annotate_entity(
            activity, label=orig, comment=f"Events of type '{orig}'."
        )

    @_when_enabled
    def object_class(self, obj_type: str) -> None:
        orig = self.original(obj_type)
        self.sink.annotate_entity(
            obj_type, label=orig, comment=f"Objects of type '{orig}'."
        )

    @_when_enabled
    def entity_individuals(self, entity_type: str, ids: Iterable[Any]) -> None:
        """Label each event/object individual as ``Type (original id)``."""
        orig_type = self.original(entity_type)
        self.sink.annotate_individuals_bulk(
            [(str(eid), f"{orig_type} ({self.original(str(eid))})") for eid in ids]
        )

    # ------------------------------------------------------------------
    # Event order
    # ------------------------------------------------------------------

    @_when_enabled
    def order_superproperties(self, include_ef: bool) -> None:
        self._pair(
            "df",
            "directly followed by",
            "Holds between two events when one immediately precedes the other within "
            "any case. Supertype of all case-specific directly-follows properties.",
            "directly preceded by",
            "Relates an event to the event that directly precedes it within any case. "
            "Inverse of 'directly followed by'.",
        )
        if include_ef:
            self._pair(
                "ef",
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

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------

    @_when_enabled
    def time_individuals(self, ids: pd.Series, labels: pd.Series) -> None:
        self.sink.annotate_individuals_bulk(list(zip(ids, labels)))

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    @_when_enabled
    def attribute(self, attr: str, prop: str, owner: str) -> None:
        """Annotate an attribute class and its ``has_attribute`` sub-property.

        Args:
            attr: Attribute class name.
            prop: The ``has_{attr}`` property name.
            owner: ``"event"`` or ``"object"``, used in the comment text.
        """
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
        """Label attribute-value individuals with their original (pre-clean) text."""
        self.sink.annotate_individuals_bulk(
            [
                (name, self._backmap.get(name, _format_value(value)))
                for value, name in value_to_name.items()
            ]
        )

    # ------------------------------------------------------------------
    # Relations
    # ------------------------------------------------------------------

    @_when_enabled
    def relation(
        self, relation: str, from_types: Sequence[str], to_types: Sequence[str]
    ) -> None:
        orig = self.original(relation)
        # e2o relations carry no explicit source type; the domain is always an event.
        source = " or ".join(self.original(t) for t in from_types) or "event"
        target = " or ".join(self.original(t) for t in to_types)
        self._pair(
            relation,
            orig,
            f"Relates '{source}' to '{target}' via the '{orig}' relationship.",
            f"inverse of {orig}",
            f"Inverse of '{orig}': relates '{target}' back to '{source}'.",
        )

    # ------------------------------------------------------------------
    # Attribute change history
    # ------------------------------------------------------------------

    @_when_enabled
    def attribute_change_schema(
        self, attr: str, change_class: str, change_prop: str
    ) -> None:
        orig = self.original(attr)
        self.sink.annotate_entity(
            attr,
            label=orig,
            comment=f"Attribute '{orig}' of an object, tracking changes over time.",
        )
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
        owners = obj_type + " (" + object_ids.astype(str) + ")"
        labels = f"{attr} Change for " + owners + " at " + timestamp_labels(timestamps)
        self.sink.annotate_individuals_bulk(list(zip(change_ids, labels)))

    # ------------------------------------------------------------------

    def _pair(
        self,
        prop: str,
        label: str,
        comment: str,
        inverse_label: str,
        inverse_comment: str,
    ) -> None:
        """Annotate a property and, when inverses are active, its named inverse."""
        self.sink.annotate_entity(prop, label=label, comment=comment)
        if self.has_inverses:
            self.sink.annotate_entity(
                self.sink.generate_inverse_property_name(prop),
                label=inverse_label,
                comment=inverse_comment,
            )
