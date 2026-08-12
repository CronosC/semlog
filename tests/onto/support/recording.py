"""A sink that records calls instead of writing axioms.

Lets an emitter be tested for what it asks for, with no ontology and no
backend involved.
"""

from __future__ import annotations

import inspect

from semanticlog.onto.settings import DEFAULT_IRI, OntoBuildSettings
from semanticlog.onto.sinks import AxiomSink


class RecordingSink(AxiomSink):
    def __init__(self, settings: OntoBuildSettings, iri: str = DEFAULT_IRI) -> None:
        super().__init__(iri, settings)
        self.calls: list[tuple] = []

    def _record(self, *call) -> None:
        self.calls.append(call)

    def write_declaration(self, name, kind):
        self._record("declare", name, kind)

    def write_sub_class_of(self, sub, superclass):
        self._record("sub_class", sub, superclass)

    def write_sub_property_of(self, sub, parent, data_prop):
        self._record("sub_property", sub, parent)

    def write_domain(self, prop, expression, data_prop):
        self._record("domain", prop, expression)

    def write_range(self, prop, expression, data_prop):
        self._record("range", prop, expression)

    def write_inverse_properties(self, prop, inverse):
        self._record("inverse", prop, inverse)

    def write_transitive(self, prop):
        self._record("transitive", prop)

    def write_class_assertions(self, class_expression, individuals):
        for individual in individuals:
            self._record("individual", str(individual), class_expression)

    def write_object_assertions(self, prop, pairs):
        for subject, obj in pairs:
            self._record("relation", prop, str(subject), str(obj))

    def write_data_assertions(self, prop, pairs):
        for subject, value in pairs:
            self._record("data", prop, str(subject), value)

    def write_annotations(self, annotation, pairs):
        for name, text in pairs:
            self._record("annotate", str(name), annotation, text)

    def write_ontology_annotation(self, property_iri, value):
        self._record("ontology_annotation", property_iri, value)

    def save(self, path):
        raise NotImplementedError("RecordingSink holds no ontology to save")

    def of_kind(self, kind: str) -> list[tuple]:
        return [c for c in self.calls if c[0] == kind]

    def declaration_order_violations(self) -> list[tuple[str, str]]:
        declared: set[str] = set()
        seen_as_individual: dict[str, str] = {}
        violations: list[tuple[str, str]] = []
        for call in self.calls:
            if call[0] == "declare":
                if call[1] in seen_as_individual:
                    violations.append((call[1], seen_as_individual[call[1]]))
                declared.add(call[1])
            elif call[0] == "individual":
                seen_as_individual.setdefault(call[1], str(call[2]))
        return violations


def make_emitter(cls, sink: AxiomSink, settings: OntoBuildSettings):
    from semanticlog.onto.annotations import AnnotationWriter
    from semanticlog.onto.emitters.attributes import AttributeEmitter

    annotations = AnnotationWriter(sink, settings)
    parameters = inspect.signature(cls.__init__).parameters
    if "attributes" in parameters:
        return cls(
            sink, settings, annotations, AttributeEmitter(sink, settings, annotations)
        )
    return cls(sink, settings, annotations)
