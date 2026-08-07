from __future__ import annotations

from semanticlog.onto.builder import OCEDBaseBuilder
from semanticlog.onto.semlog import SemanticLog
from semanticlog.onto.settings import OwlProfile


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
        has_inverses = self.settings.has_inverses
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

        if self.settings.instantiate_attribute_history:
            self._add_attribute_history_schema()

        self.annotations.upper_ontology()

        if path is not None:
            self.save_onto(path)

        return SemanticLog(onto=self.onto, settings=self.settings, iri=self.iri)

    def _add_attribute_history_schema(self) -> None:
        has_inverses = self.settings.has_inverses
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
