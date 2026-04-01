from semanticlog.onto.builder import OCEDBaseBuilder
from semanticlog.onto.settings import OcelExtension, OntoBuildSettings


class OCEDUpperOntoBuilder(OCEDBaseBuilder):

    def __init__(self, iri, settings: OntoBuildSettings):
        super().__init__(iri, settings)

    def construct_upper_ontology(self, path: str) -> None:
        self.add_classes(["Object", "Event", "Attribute", "Time"])
        self.add_classes(["ObjectAttribute", "EventAttribute"], "Attribute")

        self.add_property(
            "observed_at",
            domain_class="Event",
            range_class="Time",
            create_inverse=self.settings.inverse_properties,
            inverse_name="observes" if self.settings.inverse_properties else None,
        )
        self.add_property(
            "event_object_relation",
            domain_class="Event",
            range_class="Object",
            create_inverse=self.settings.inverse_properties,
        )
        self.add_property(
            "object_object_relation",
            domain_class="Object",
            range_class="Object",
            create_inverse=self.settings.inverse_properties,
        )
        self.add_property(
            "has_attribute",
            domain_class="Object or Event",
            range_class="Attribute",
            create_inverse=self.settings.inverse_properties,
            inverse_name="attribute_of" if self.settings.inverse_properties else None,
        )
        self.add_property(
            "has_data_value",
            domain_class="Attribute or Time",
            data_prop=True,
        )

        if OcelExtension.ATTRIBUTE_HISTORY in self.settings.ocel_extensions:
            self._add_attribute_history_schema()

        if OcelExtension.RELATION_TIMESTAMPS in self.settings.ocel_extensions:
            self._add_relation_timestamps_schema()

        self.save_onto(path)

    def _add_attribute_history_schema(self) -> None:
        # AttributeChange is a top-level class (not a subclass of Attribute).
        # Each individual represents one timestamped change to an object attribute.
        self.add_class("AttributeChange")
        self.add_property(
            "has_attribute_change",
            domain_class="Object",
            range_class="AttributeChange",
            create_inverse=self.settings.inverse_properties,
        )
        self.add_property(
            "attribute_change_value",
            domain_class="AttributeChange",
            range_class="Attribute",
            create_inverse=self.settings.inverse_properties,
        )
        self.add_property(
            "changed_at",
            domain_class="AttributeChange",
            range_class="Time",
            create_inverse=self.settings.inverse_properties,
        )

    def _add_relation_timestamps_schema(self) -> None:
        # OCEL 2.0 e2o relations carry a timestamp recording when each event-object
        # relationship was observed. OWL object properties cannot carry data directly,
        # so a n-ary reification pattern is used: each timed observation becomes a
        # RelationObservation individual that links event, object, and timestamp.
        # The existing direct event_object_relation properties are preserved for
        # backward-compatible queries.
        self.add_class("RelationObservation")
        self.add_property(
            "has_relation_observation",
            domain_class="Event",
            range_class="RelationObservation",
            create_inverse=self.settings.inverse_properties,
        )
        self.add_property(
            "relation_involves_object",
            domain_class="RelationObservation",
            range_class="Object",
            create_inverse=self.settings.inverse_properties,
        )
        self.add_property(
            "relation_observed_at",
            domain_class="RelationObservation",
            range_class="Time",
            create_inverse=self.settings.inverse_properties,
        )
