from __future__ import annotations

from semanticlog.onto.builder import OCEDBaseBuilder
from semanticlog.onto.semlog import SemanticLog
from semanticlog.onto.settings import OcelExtension, OwlProfile

_TBOX_ANNOTATIONS: dict[str, dict[str, str]] = {
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


class OCEDUpperOntoBuilder(OCEDBaseBuilder):
    """Builds the upper ontology (TBox schema) for an OCED-to-OWL ontology.

    Defines the core class hierarchy (Object, Event, Attribute, Time) and the
    shared object/data properties.  Called internally by
    :class:`~semanticlog.onto.instantiate.OCEDOntoBuilder`; use directly only
    for inspection or debugging.
    """

    def construct_upper_ontology(self, path: str | None = None) -> SemanticLog:
        """Build the upper ontology (TBox) schema and return it as a SemanticLog.

        Args:
            path: If provided, also save the OWL file to this path (e.g. for
                inspection or debugging). Optional.

        Returns:
            SemanticLog containing the built upper ontology.
        """
        has_inverses = bool(self.settings.inverse_representation)
        self.add_classes(["Object", "Event", "Attribute", "Time"])
        self.add_classes(["ObjectAttribute", "EventAttribute"], "Attribute")

        self.add_property(
            "observed_at",
            domain_class="Event",
            range_class="Time",
            create_inverse=has_inverses,
            inverse_name="observes" if has_inverses else None,
        )
        self.add_property(
            "event_object_relation",
            domain_class="Event",
            range_class="Object",
            create_inverse=has_inverses,
        )
        self.add_property(
            "object_object_relation",
            domain_class="Object",
            range_class="Object",
            create_inverse=has_inverses,
        )
        has_attribute_domain = (
            "Object or Event" if self.settings.profile == OwlProfile.OWL2DL else None
        )
        self.add_property(
            "has_attribute",
            domain_class=has_attribute_domain,
            range_class="Attribute",
            create_inverse=has_inverses,
            inverse_name="attribute_of" if has_inverses else None,
        )
        has_data_value_domain = (
            "Attribute or Time" if self.settings.profile == OwlProfile.OWL2DL else None
        )
        self.add_property(
            "has_data_value",
            domain_class=has_data_value_domain,
            data_prop=True,
        )

        if OcelExtension.ATTRIBUTE_HISTORY in self.settings.ocel_extensions:
            self._add_attribute_history_schema()

        if self.settings.derive_annotations:
            self._annotate_upper_ontology()

        if path is not None:
            self.save_onto(path)

        return SemanticLog(onto=self.onto, settings=self.settings, iri=self.iri)

    def _annotate_upper_ontology(self) -> None:
        for name, ann in _TBOX_ANNOTATIONS.items():
            self.annotate_entity(
                name, label=ann.get("label"), comment=ann.get("comment")
            )

    def _add_attribute_history_schema(self) -> None:
        has_inverses = bool(self.settings.inverse_representation)
        # AttributeChange is a top-level class (not a subclass of Attribute).
        # Each individual represents one timestamped change to an object attribute.
        self.add_class("AttributeChange")
        self.add_property(
            "has_attribute_change",
            domain_class="Object",
            range_class="AttributeChange",
            create_inverse=has_inverses,
        )
        self.add_property(
            "attribute_change_value",
            domain_class="AttributeChange",
            range_class="Attribute",
            create_inverse=has_inverses,
        )
        self.add_property(
            "changed_at",
            domain_class="AttributeChange",
            range_class="Time",
            create_inverse=has_inverses,
        )
