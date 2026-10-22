"""A sink that records each call."""

from __future__ import annotations

from typing import Any

from semanticlog.core.enums import OWLProfile
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.onto.sinks import AxiomSink


class RecordingSink(AxiomSink):
    """Records every `write_*` call as a tuple headed by its kind.

    Attributes:
        calls: The calls in order. A data assertion carries the resolved
            literal's value.
    """

    def __init__(
        self, profile: OWLProfile = OWLProfile.OWL2DL, iri: str = DEFAULT_IRI
    ) -> None:
        super().__init__(iri, profile)
        self.calls: list[tuple[Any, ...]] = []

    def _record(self, *call: Any) -> None:
        self._begin_write()
        self.calls.append(call)

    def write_declaration(self, name, kind):
        self._record("declare", name, kind)

    def write_sub_class_of(self, sub, superclass):
        self._record("sub_class", sub, superclass)

    def write_sub_property_of(self, sub, parent, data_prop):
        self._record("sub_property", sub, parent, data_prop)

    def write_domain(self, prop, expression, data_prop):
        self._record("domain", prop, expression, data_prop)

    def write_range(self, prop, expression):
        self._record("range", prop, expression, False)

    def write_data_range(self, prop, datatype):
        self._record("range", prop, [datatype], True)

    def write_inverse_properties(self, prop, inverse):
        self._record("inverse", prop, inverse)

    def write_transitive(self, prop):
        self._record("transitive", prop)

    def write_bin_definition(
        self, bin_class, attribute_class, value_prop, lower, upper
    ):
        self._record(
            "bin_definition", bin_class, attribute_class, value_prop, lower, upper
        )

    def write_functional(self, prop, data_prop):
        self._record("functional", prop, data_prop)

    def write_inverse_functional(self, prop):
        self._record("inverse_functional", prop)

    def write_disjoint_classes(self, classes):
        self._record("disjoint", list(classes))

    def write_class_assertions(self, class_expression, individuals):
        for individual in individuals:
            self._record("individual", str(individual), class_expression)

    def write_object_assertions(self, prop, pairs):
        for subject, obj in pairs:
            self._record("relation", prop, str(subject), str(obj))

    def write_data_assertions(self, prop, pairs):
        for subject, literal in pairs:
            self._record("data", prop, str(subject), literal.value)

    def write_annotations(self, annotation, pairs):
        for name, text in pairs:
            self._record("annotate", str(name), annotation, text)

    def write_ontology_header(self, annotations):
        for property_iri, value in annotations.items():
            self._record("ontology_annotation", property_iri, value)

    def save(self, path):
        raise NotImplementedError("RecordingSink holds no ontology to save")

    def of_kind(self, kind: str) -> list[tuple[Any, ...]]:
        """Return the recorded calls of one kind, in order."""
        return [c for c in self.calls if c[0] == kind]

    def declaration_order_violations(self) -> list[tuple[str, str]]:
        """Return each name asserted as an individual before its declaration.

        Returns:
            Pairs of the name and the class it was asserted into.
        """
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
