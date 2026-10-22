"""Sink that builds an owlapy ontology object in memory.

`InMemorySink` keeps the ontology for inspection or reasoning without writing
a file. `StreamingSink` writes axioms to disk as they arrive. `OCEDOntoBuilder`
uses `InMemorySink` when given no sink.
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

from semanticlog.core.enums import OWLProfile, SinkEntity
from semanticlog.onto.sinks.base import PARTIAL_SUFFIX, AxiomSink
from semanticlog.onto.sinks.literals import RDFS, XSD, Literal

_RDFS_LABEL = OWLAnnotationProperty(IRI(RDFS, "label"))
_RDFS_COMMENT = OWLAnnotationProperty(IRI(RDFS, "comment"))


def owl_literal(value: Any, datatype: Any = None) -> Any:
    """Return the owlapy literal for a plain Python value.

    `OWLLiteral` is declared abstract and picks the concrete subclass for a
    value in `__new__`, which a type checker cannot follow.

    Args:
        value: The value to carry.
        datatype: Datatype override. Default: inferred from the Python type.
    """
    return OWLLiteral(value, datatype)  # type: ignore[abstract]


class InMemorySink(AxiomSink):
    """Collects axioms in an owlapy ontology.

    Args:
        iri: Base IRI of the ontology.
        profile: Profile of the build.

    Attributes:
        onto: The ontology being built.
    """

    def __init__(self, iri: str, profile: OWLProfile) -> None:
        super().__init__(iri, profile)
        self.onto = Ontology(IRI.create(self.ontology_iri), load=False)
        self._pending_inverses: list[tuple[str, str]] = []

    def _add(self, axioms: Any) -> None:
        """Add one axiom or a list, and count them."""
        self._begin_write(len(axioms) if isinstance(axioms, list) else 1)
        self.onto.add_axiom(axioms)

    def _entity(self, name: str, kind: SinkEntity) -> Any:
        """Return the owlapy entity for a local name of the given kind.

        Raises:
            ValueError: If `kind` is a `SinkEntity` member this method does
                not handle.
        """
        if kind is SinkEntity.CLASS:
            return OWLClass(IRI.create(self.iri_of(name)))
        if kind is SinkEntity.OBJECT_PROPERTY:
            return OWLObjectProperty(self.iri_of(name))
        if kind is SinkEntity.DATA_PROPERTY:
            return OWLDataProperty(self.iri_of(name))
        if kind is SinkEntity.ANNOTATION_PROPERTY:
            return OWLAnnotationProperty(IRI.create(self.iri_of(name)))
        raise ValueError(f"{type(self).__name__} has no entity for {kind!r}.")

    def _class_expression(self, disjuncts: list[str]) -> Any:
        """Return one class, or their union when there are several."""
        classes = [OWLClass(IRI.create(self.iri_of(name))) for name in disjuncts]
        return classes[0] if len(classes) == 1 else OWLObjectUnionOf(classes)

    def write_declaration(self, name: str, kind: SinkEntity) -> None:
        self._add(OWLDeclarationAxiom(self._entity(name, kind)))

    def write_sub_class_of(self, sub: str, superclass: list[str]) -> None:
        self._add(
            OWLSubClassOfAxiom(
                self._entity(sub, SinkEntity.CLASS), self._class_expression(superclass)
            )
        )

    def write_sub_property_of(self, sub: str, parent: str, data_prop: bool) -> None:
        axiom = OWLSubDataPropertyOfAxiom if data_prop else OWLSubObjectPropertyOfAxiom
        kind = SinkEntity.property_for(data_prop)
        self._add(axiom(self._entity(sub, kind), self._entity(parent, kind)))

    def write_domain(self, prop: str, expression: list[str], data_prop: bool) -> None:
        axiom = (
            OWLDataPropertyDomainAxiom if data_prop else OWLObjectPropertyDomainAxiom
        )
        kind = SinkEntity.property_for(data_prop)
        self._add(axiom(self._entity(prop, kind), self._class_expression(expression)))

    def write_range(self, prop: str, expression: list[str]) -> None:
        self._add(
            OWLObjectPropertyRangeAxiom(
                self._entity(prop, SinkEntity.OBJECT_PROPERTY),
                self._class_expression(expression),
            )
        )

    def write_data_range(self, prop: str, datatype: str) -> None:
        self._add(
            OWLDataPropertyRangeAxiom(
                self._entity(prop, SinkEntity.DATA_PROPERTY),
                OWLDatatype(IRI(XSD, datatype)),
            )
        )

    def write_inverse_properties(self, prop: str, inverse: str) -> None:
        """Queue the axiom for `_finish`.

        owlapy keeps one direction of every assertion between a pair once it
        holds their inverse axiom. Assertions written before the axiom keep
        both.
        """
        self._pending_inverses.append((prop, inverse))

    def _finish(self) -> None:
        """Write the queued inverse axioms."""
        for prop, inverse in self._pending_inverses:
            self._add(
                OWLInverseObjectPropertiesAxiom(
                    OWLObjectProperty(self.iri_of(prop)),
                    OWLObjectProperty(self.iri_of(inverse)),
                )
            )

    def write_transitive(self, prop: str) -> None:
        self._add(
            OWLTransitiveObjectPropertyAxiom(OWLObjectProperty(self.iri_of(prop)))
        )

    def write_class_assertions(
        self, class_name: str, individuals: Iterable[str]
    ) -> None:
        owl_class = self._entity(class_name, SinkEntity.CLASS)
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
        owl_prop = OWLObjectProperty(self.iri_of(prop))
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
        self, prop: str, pairs: Iterable[tuple[str, Literal]]
    ) -> None:
        """Assert each literal.

        Under EL, a boolean or date arrives as text and owlapy writes it as
        `xsd:string`.
        """
        iri = self.iri
        owl_prop = OWLDataProperty(self.iri_of(prop))
        axioms = [
            OWLDataPropertyAssertionAxiom(
                OWLNamedIndividual(f"{iri}{subject}"),
                owl_prop,
                owl_literal(literal.value),
            )
            for subject, literal in pairs
        ]
        if axioms:
            self._add(axioms)

    def write_annotations(
        self, annotation: str, pairs: Iterable[tuple[str, str]]
    ) -> None:
        """Annotate entities.

        Raises:
            ValueError: If owlapy rejects an annotation on an undeclared
                entity.
        """
        prop = _RDFS_LABEL if annotation == "label" else _RDFS_COMMENT
        pairs = list(pairs)
        axioms = [
            OWLAnnotationAssertionAxiom(
                IRI.create(self.iri_of(name)),
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
                f"owlapy rejected {len(axioms)} {annotation} annotation(s) on "
                f"{', '.join(names[:5])}. Declare each entity before annotating it."
            ) from exc

    def write_ontology_header(self, annotations: Mapping[str, str]) -> None:
        """Write the build record as annotations on the ontology."""
        for property_iri, value in annotations.items():
            self._add(
                OWLAnnotationAssertionAxiom(
                    IRI.create(self.ontology_iri),
                    OWLAnnotation(
                        OWLAnnotationProperty(IRI.create(property_iri)),
                        owl_literal(value),
                    ),
                )
            )

    def save(self, path: str) -> None:
        """Close the sink and write the ontology.

        The file is written under `PARTIAL_SUFFIX` and then moved to `path`.
        """
        self.close()
        partial = Path(path + PARTIAL_SUFFIX)
        try:
            self.onto.save(str(partial))
            partial.replace(path)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

    def result_target(self) -> tuple[Ontology | None, str | None]:
        """Return the ontology and no path."""
        return self.onto, None
