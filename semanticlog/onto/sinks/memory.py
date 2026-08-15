"""Sink that builds an ontology object in memory, through owlapy.

Every axiom is kept as an object, so the result can be inspected or reasoned
over without being written out first. A log too large to hold in memory needs
`StreamingSink` instead, which is what the CLI selects unless `streaming` is
turned off. A builder given no sink of its own uses this one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from owlapy.class_expression import OWLClass, OWLObjectUnionOf
from owlapy.iri import IRI
from owlapy.owl_axiom import (
    OWLAnnotation,
    OWLAnnotationAssertionAxiom,
    OWLAnnotationProperty,
    OWLClassAssertionAxiom,
    OWLDataPropertyAssertionAxiom,
    OWLDataPropertyDomainAxiom,
    OWLDataPropertyRangeAxiom,
    OWLDeclarationAxiom,
    OWLInverseObjectPropertiesAxiom,
    OWLObjectPropertyAssertionAxiom,
    OWLObjectPropertyDomainAxiom,
    OWLObjectPropertyRangeAxiom,
    OWLSubClassOfAxiom,
    OWLSubDataPropertyOfAxiom,
    OWLSubObjectPropertyOfAxiom,
    OWLTransitiveObjectPropertyAxiom,
)
from owlapy.owl_datatype import OWLDatatype
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_literal import OWLLiteral
from owlapy.owl_ontology import Ontology
from owlapy.owl_property import OWLDataProperty, OWLObjectProperty

from semanticlog.core.enums import OWLProfile
from semanticlog.onto.sinks.base import (
    _RDFS,
    _XSD,
    OWL_DATATYPES,
    PARTIAL_SUFFIX,
    AxiomSink,
)
from semanticlog.onto.sinks.literals import resolve_literal

_RDFS_LABEL = OWLAnnotationProperty(IRI(_RDFS, "label"))
_RDFS_COMMENT = OWLAnnotationProperty(IRI(_RDFS, "comment"))


def owl_literal(value: Any, datatype: Any = None) -> Any:
    """Return the owlapy literal for a plain Python value.

    `OWLLiteral` is declared abstract and picks the concrete subclass for a
    value in `__new__`, which a type checker cannot follow.

    Args:
        value: The value to carry.
        datatype: Type to declare it under, instead of the one owlapy would
            infer from the Python type.
    """
    return OWLLiteral(value, datatype)  # type: ignore[abstract]


class InMemorySink(AxiomSink):
    """Collects axioms in an owlapy ontology.

    Attributes:
        onto: The ontology being built. It is available while the build runs,
            and is what the build result carries afterwards.
    """

    def __init__(self, iri: str, profile: OWLProfile):
        """Create an empty ontology to collect into."""
        super().__init__(iri, profile)
        self.onto = Ontology(IRI.create(iri), load=False)
        self._pending_inverses: list[tuple[str, str]] = []
        self._axiom_count = 0

    @property
    def axiom_count(self) -> int:
        """How many axioms have been added, counted as they arrive."""
        return self._axiom_count

    def _add(self, axioms: Any) -> None:
        """Add one axiom or a list of them, and count what went in."""
        self._begin_write()
        self._axiom_count += len(axioms) if isinstance(axioms, list) else 1
        self.onto.add_axiom(axioms)

    def _entity(self, name: str, kind: str) -> Any:
        """Return the owlapy entity for a local name of the given kind."""
        if kind == "class":
            return OWLClass(IRI.create(self.iri + name))
        if kind == "data":
            return OWLDataProperty(self.iri + name)
        if kind == "annotation":
            return OWLAnnotationProperty(IRI.create(self.iri + name))
        return OWLObjectProperty(self.iri + name)

    def _class_expression(self, disjuncts: list[str]) -> Any:
        """Return one class, or their union when there are several."""
        classes = [OWLClass(IRI.create(self.iri + name)) for name in disjuncts]
        return classes[0] if len(classes) == 1 else OWLObjectUnionOf(classes)

    def write_declaration(self, name: str, kind: str) -> None:
        self._add(OWLDeclarationAxiom(self._entity(name, kind)))

    def write_sub_class_of(self, sub: str, superclass: list[str]) -> None:
        self._add(
            OWLSubClassOfAxiom(
                self._entity(sub, "class"), self._class_expression(superclass)
            )
        )

    def write_sub_property_of(self, sub: str, parent: str, data_prop: bool) -> None:
        axiom = OWLSubDataPropertyOfAxiom if data_prop else OWLSubObjectPropertyOfAxiom
        kind = "data" if data_prop else "object"
        self._add(axiom(self._entity(sub, kind), self._entity(parent, kind)))

    def write_domain(self, prop: str, expression: list[str], data_prop: bool) -> None:
        axiom = (
            OWLDataPropertyDomainAxiom if data_prop else OWLObjectPropertyDomainAxiom
        )
        kind = "data" if data_prop else "object"
        self._add(axiom(self._entity(prop, kind), self._class_expression(expression)))

    def write_range(self, prop: str, expression: list[str], data_prop: bool) -> None:
        axiom = OWLDataPropertyRangeAxiom if data_prop else OWLObjectPropertyRangeAxiom
        kind = "data" if data_prop else "object"
        if len(expression) == 1 and expression[0] in OWL_DATATYPES:
            owl_range: Any = OWLDatatype(IRI(_XSD, expression[0]))
        else:
            owl_range = self._class_expression(expression)
        self._add(axiom(self._entity(prop, kind), owl_range))

    def write_inverse_properties(self, prop: str, inverse: str) -> None:
        """Hold the axiom back until the build has written its assertions.

        The backend links a pair of inverse properties the moment the axiom
        arrives, and from then on keeps one direction of every assertion
        between them. Assertions written before the axiom keep both, so these
        are collected here and written by `close`.
        """
        self._pending_inverses.append((prop, inverse))

    def _finish(self) -> None:
        """Write the inverse axioms held back during the build.

        Anything asserted after this loses one direction of an inverse pair,
        which is what the base class closing once is there to prevent.
        """
        for prop, inverse in self._pending_inverses:
            self._add(
                OWLInverseObjectPropertiesAxiom(
                    OWLObjectProperty(self.iri + prop),
                    OWLObjectProperty(self.iri + inverse),
                )
            )

    def write_transitive(self, prop: str) -> None:
        self._add(OWLTransitiveObjectPropertyAxiom(OWLObjectProperty(self.iri + prop)))

    def write_class_assertions(
        self, class_name: str, individuals: Iterable[str]
    ) -> None:
        owl_class = self._entity(class_name, "class")
        iri = self.iri
        axioms = [
            OWLClassAssertionAxiom(OWLNamedIndividual(f"{iri}{individual}"), owl_class)
            for individual in individuals
        ]
        if axioms:
            self._add(axioms)

    def write_object_assertions(
        self, prop: str, pairs: Iterable[tuple[str, str]]
    ) -> None:
        iri = self.iri
        owl_prop = OWLObjectProperty(f"{iri}{prop}")
        axioms = [
            OWLObjectPropertyAssertionAxiom(
                OWLNamedIndividual(f"{iri}{s}"),
                owl_prop,
                OWLNamedIndividual(f"{iri}{o}"),
            )
            for s, o in pairs
        ]
        if axioms:
            self._add(axioms)

    def write_data_assertions(
        self, prop: str, pairs: Iterable[tuple[str, Any]]
    ) -> None:
        iri = self.iri
        owl_prop = OWLDataProperty(f"{iri}{prop}")
        axioms = []
        for subject, value in pairs:
            literal = self._literal(value)
            if literal is None:
                continue
            axioms.append(
                OWLDataPropertyAssertionAxiom(
                    OWLNamedIndividual(f"{iri}{subject}"), owl_prop, literal
                )
            )
        if axioms:
            self._add(axioms)

    def _literal(self, value: Any) -> Any:
        """Return the owlapy literal for a value, or None when it is missing.

        The value is the one `resolve_literal` settled for this profile, so an
        EL build hands owlapy the text of a boolean or a date and gets the
        `xsd:string` that profile has room for.
        """
        literal = resolve_literal(value, self.profile)
        if literal is None:
            return None
        return owl_literal(literal.value)

    def write_annotations(
        self, annotation: str, pairs: Iterable[tuple[str, str]]
    ) -> None:
        """Annotate entities, turning a backend refusal into a clear failure.

        Raises:
            ValueError: If the backend rejects the annotations, which means a
                build step annotated something before declaring it.
        """
        prop = _RDFS_LABEL if annotation == "label" else _RDFS_COMMENT
        pairs = list(pairs)
        axioms = [
            OWLAnnotationAssertionAxiom(
                IRI.create(f"{self.iri}{name}"),
                OWLAnnotation(prop, owl_literal(text)),
            )
            for name, text in pairs
        ]
        if not axioms:
            return
        try:
            self._add(axioms)
        except AssertionError as exc:
            names = sorted({str(name) for name, _ in pairs})
            raise ValueError(
                f"The backend refused {len(axioms)} {annotation} annotation(s) on "
                f"{', '.join(names[:5])}. Every annotated entity should already be "
                "declared, so this is a build-order problem rather than bad input."
            ) from exc

    def write_ontology_header(self, annotations: Mapping[str, str]) -> None:
        """Annotate the ontology itself, which is what carries the record.

        Nothing has to be opened here. The ontology object exists from the
        start and takes an annotation whenever one arrives, which is why the
        rule about the record coming first lives in the base class rather
        than in this backend.
        """
        for property_iri, value in annotations.items():
            namespace, _, local = property_iri.rpartition("#")
            self._add(
                OWLAnnotationAssertionAxiom(
                    IRI.create(self.iri.rstrip("#")),
                    OWLAnnotation(
                        OWLAnnotationProperty(IRI(namespace + "#", local)),
                        owl_literal(value),
                    ),
                )
            )

    def save(self, path: str) -> None:
        """Finish the ontology and write it out, through a partial file.

        A save that fails part-way then leaves whatever was at the destination
        untouched, rather than a truncated file where a finished one used to
        be.
        """
        self.close()
        partial = Path(path + PARTIAL_SUFFIX)
        try:
            self.onto.save(str(partial))
            partial.replace(path)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

    def result_target(self) -> tuple[Any, str | None]:
        """Hand back the ontology, which this sink held in memory."""
        return self.onto, None
