"""Sink that builds an owlapy ontology object in memory.

`InMemorySink` keeps the ontology for inspection or reasoning without writing
a file. `StreamingSink` writes axioms to disk as they arrive. `OCEDOntoBuilder`
uses `InMemorySink` when given no sink.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from owlapy.class_expression import (
    OWLClass,
    OWLDataSomeValuesFrom,
    OWLDatatypeRestriction,
    OWLFacetRestriction,
    OWLObjectIntersectionOf,
    OWLObjectUnionOf,
)
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
    OWLDisjointClassesAxiom,
    OWLEquivalentClassesAxiom,
    OWLFunctionalDataPropertyAxiom,
    OWLFunctionalObjectPropertyAxiom,
    OWLInverseFunctionalObjectPropertyAxiom,
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
from owlapy.owl_literal import DoubleOWLDatatype, OWLLiteral
from owlapy.owl_ontology import Ontology
from owlapy.owl_property import OWLDataProperty, OWLObjectProperty
from owlapy.vocab import OWLFacet

from semanticlog.core.enums import IndividualKind, OWLProfile, SinkEntity
from semanticlog.onto.sinks.base import PARTIAL_SUFFIX, AxiomSink
from semanticlog.onto.sinks.literals import DECIMAL, RDFS, XSD, Literal

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
        """Write the range, giving `xsd:decimal` as `xsd:double`.

        owlapy has no mapping for `xsd:decimal`. It maps `xsd:double` to Python
        floats, which owlready2 writes as `xsd:decimal`.
        """
        owl_datatype = (
            DoubleOWLDatatype
            if datatype == DECIMAL
            else OWLDatatype(IRI(XSD, datatype))
        )
        self._add(
            OWLDataPropertyRangeAxiom(
                self._entity(prop, SinkEntity.DATA_PROPERTY), owl_datatype
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

    def write_bin_definition(
        self,
        bin_class: str,
        attribute_class: str,
        value_prop: str,
        lower: float | None,
        upper: float | None,
    ) -> None:
        """Write the definition over `xsd:decimal`.

        owlapy maps a restriction on `xsd:double` to Python floats, which
        owlready2 writes as `xsd:decimal`. It has no mapping for `xsd:decimal`
        itself.
        """
        facets = []
        if lower is not None:
            facets.append(
                OWLFacetRestriction(OWLFacet.MIN_EXCLUSIVE, owl_literal(lower))
            )
        if upper is not None:
            facets.append(
                OWLFacetRestriction(OWLFacet.MAX_INCLUSIVE, owl_literal(upper))
            )
        in_range = OWLDataSomeValuesFrom(
            OWLDataProperty(self.iri_of(value_prop)),
            OWLDatatypeRestriction(DoubleOWLDatatype, facets),
        )
        self._add(
            OWLEquivalentClassesAxiom(
                [
                    self._entity(bin_class, SinkEntity.CLASS),
                    OWLObjectIntersectionOf(
                        [self._entity(attribute_class, SinkEntity.CLASS), in_range]
                    ),
                ]
            )
        )

    def write_functional(self, prop: str, data_prop: bool) -> None:
        if data_prop:
            self._add(
                OWLFunctionalDataPropertyAxiom(OWLDataProperty(self.iri_of(prop)))
            )
        else:
            self._add(
                OWLFunctionalObjectPropertyAxiom(OWLObjectProperty(self.iri_of(prop)))
            )

    def write_inverse_functional(self, prop: str) -> None:
        self._add(
            OWLInverseFunctionalObjectPropertyAxiom(
                OWLObjectProperty(self.iri_of(prop))
            )
        )

    def write_disjoint_classes(self, classes: list[str]) -> None:
        self._add(
            OWLDisjointClassesAxiom(
                [self._entity(name, SinkEntity.CLASS) for name in classes]
            )
        )

    def _individual(self, name: str, kind: IndividualKind) -> OWLNamedIndividual:
        return OWLNamedIndividual(self.iri_of(name, kind))

    def write_class_assertions(
        self, class_name: str, individuals: Iterable[str], kind: IndividualKind
    ) -> None:
        owl_class = self._entity(class_name, SinkEntity.CLASS)
        axioms = [
            OWLClassAssertionAxiom(self._individual(individual, kind), owl_class)
            for individual in individuals
        ]
        if axioms:
            self._add(axioms)

    def write_object_assertions(
        self,
        prop: str,
        pairs: Iterable[tuple[str, str]],
        kinds: tuple[IndividualKind, IndividualKind],
    ) -> None:
        owl_prop = OWLObjectProperty(self.iri_of(prop))
        subject_kind, target_kind = kinds
        axioms = [
            OWLObjectPropertyAssertionAxiom(
                self._individual(s, subject_kind),
                owl_prop,
                self._individual(o, target_kind),
            )
            for s, o in pairs
        ]
        if axioms:
            self._add(axioms)

    def write_data_assertions(
        self, prop: str, pairs: Iterable[tuple[str, Literal]], kind: IndividualKind
    ) -> None:
        """Assert each literal.

        Under EL, a boolean or date arrives as text and owlapy writes it as
        `xsd:string`.
        """
        owl_prop = OWLDataProperty(self.iri_of(prop))
        axioms = [
            OWLDataPropertyAssertionAxiom(
                self._individual(subject, kind),
                owl_prop,
                owl_literal(literal.value),
            )
            for subject, literal in pairs
        ]
        if axioms:
            self._add(axioms)

    def write_annotations(
        self,
        annotation: str,
        pairs: Iterable[tuple[str, str]],
        kind: IndividualKind | None = None,
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
                IRI.create(self.iri_of(name, kind)),
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

    def write_entity_annotation(
        self, prop: str, name: str, value: str, iri: bool
    ) -> None:
        annotation_value = IRI.create(self.iri_of(value)) if iri else owl_literal(value)
        self._add(
            OWLAnnotationAssertionAxiom(
                IRI.create(self.iri_of(name)),
                OWLAnnotation(
                    OWLAnnotationProperty(IRI.create(self.iri_of(prop))),
                    annotation_value,
                ),
            )
        )

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
