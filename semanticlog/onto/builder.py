import logging
import re
from typing import Any

import pandas as pd
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
)
from owlapy.owl_datatype import OWLDatatype
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_literal import OWLLiteral
from owlapy.owl_ontology import Ontology
from owlapy.owl_property import OWLDataProperty, OWLObjectProperty

from semanticlog.onto.settings import InverseRepresentation, OntoBuildSettings
from semanticlog.onto.transform import parse_manchester

_OWL_DATATYPES: frozenset[str] = frozenset({"integer", "double", "boolean", "dateTime"})

# Explicitly named inverses from the upper ontology schema.  Without AXIOMATIC
# representation no OWLInverseObjectPropertiesAxiom is written, so these pairs
# cannot be recovered from the saved TBox via prop.inverse_property.  Seeding
# them here ensures both the TBox and ABox builders always use the right name.
# Must be kept in sync with upper.py add_property calls that pass inverse_name.
_SCHEMA_NAMED_INVERSES: dict[str, str] = {
    "has_attribute": "attribute_of",
    "attribute_of": "has_attribute",
    "observed_at": "observes",
    "observes": "observed_at",
}

_RDFS_LABEL = OWLAnnotationProperty(
    IRI("http://www.w3.org/2000/01/rdf-schema#", "label")
)
_RDFS_COMMENT = OWLAnnotationProperty(
    IRI("http://www.w3.org/2000/01/rdf-schema#", "comment")
)


class OCEDBaseBuilder:
    """Low-level OWL axiom builder wrapping owlapy.

    Manages an owlapy ``Ontology`` and provides methods for declaring classes,
    properties, and ABox assertions.  Handles named-inverse tracking, annotation
    documents, and the ``InverseRepresentation`` strategy.

    Subclassed by :class:`~semanticlog.onto.upper.OCEDUpperOntoBuilder` (TBox)
    and :class:`~semanticlog.onto.instantiate.OCEDOntoBuilder` (full pipeline).
    """

    def __init__(
        self,
        iri: str,
        settings: OntoBuildSettings,
        _onto: Ontology | None = None,
    ):
        if _onto is not None:
            self.onto = _onto
            self.iri = iri
        else:
            try:
                self.onto = Ontology(IRI.create(iri), load=True)
                self.iri = self.onto.get_ontology_id().get_ontology_iri().str  # type: ignore[union-attr]
            except Exception as e:
                # Ontology could not be loaded (file not found, unsupported kwarg, etc.)
                # — fall back to creating a fresh empty ontology.
                logging.warning(
                    f"Could not load ontology from IRI '{iri}': {e}. Creating empty ontology."
                )
                self.onto = Ontology(IRI.create(iri), load=False)
                self.iri = iri
        if not self.iri.endswith("#"):
            self.iri = self.iri + "#"
        self.settings = settings
        self._named_inverse_map: dict[str, str] = dict(_SCHEMA_NAMED_INVERSES)
        for prop in self.onto._onto.object_properties():
            if prop.inverse_property is not None:
                self._named_inverse_map[prop.name] = prop.inverse_property.name
        self._annotation_doc: dict[str, dict[str, str]] = settings.annotation_data or {}

    def annotate_entity(
        self,
        iri_fragment: str,
        label: str | None = None,
        comment: str | None = None,
    ) -> None:
        """Add rdfs:label and/or rdfs:comment to a class or property by IRI fragment."""
        subject = IRI.create(f"{self.iri}{iri_fragment}")
        try:
            if label:
                self.onto.add_axiom(
                    OWLAnnotationAssertionAxiom(
                        subject,
                        OWLAnnotation(_RDFS_LABEL, OWLLiteral(label)),  # type: ignore[abstract]
                    )
                )
            if comment:
                self.onto.add_axiom(
                    OWLAnnotationAssertionAxiom(
                        subject,
                        OWLAnnotation(_RDFS_COMMENT, OWLLiteral(comment)),  # type: ignore[abstract]
                    )
                )
        except AssertionError:
            logging.warning(
                f"Skipping annotation for '{iri_fragment}': entity not found in ontology."
            )

    def annotate_individuals_bulk(self, pairs: list[tuple[str, str]]) -> None:
        """Emit rdfs:label for each (iri_fragment, label) pair in one batch."""
        axioms = [
            OWLAnnotationAssertionAxiom(
                IRI.create(f"{self.iri}{name}"),
                OWLAnnotation(_RDFS_LABEL, OWLLiteral(label)),  # type: ignore[abstract]
            )
            for name, label in pairs
        ]
        if axioms:
            self.onto.add_axiom(axioms)

    def apply_annotation_document(self) -> None:
        """Annotate all entities listed in the annotation document that exist in this ontology."""
        for name, ann in self._annotation_doc.items():
            self.annotate_entity(
                name, label=ann.get("label"), comment=ann.get("comment")
            )

    def add_classes(
        self, class_list: list[str], superclasses: list[str] | str | None = None
    ) -> None:
        """Add multiple OWL classes, optionally with a shared or per-class superclass.

        Args:
            class_list: Names of the classes to declare.
            superclasses: A single superclass name (applied to all), a list of
                superclass names (one per class), or ``None`` for no superclass.

        Raises:
            ValueError: If ``superclasses`` is a list of a different length than ``class_list``.
        """
        logging.info(f"  adding {len(class_list)} classes..")
        if isinstance(superclasses, str) or superclasses is None:
            for class_name in class_list:
                self.add_class(class_name, superclasses)
        elif isinstance(superclasses, list) and len(class_list) == len(superclasses):
            for class_name, superclass in zip(class_list, superclasses):
                self.add_class(class_name, superclass)
        else:
            raise ValueError(
                "superclasses must be a list of str with the same length as class_list, a str, or None"
            )

    def add_class(self, class_name: str, superclass: str | None = None) -> None:
        """Declare a single OWL class, optionally as a subclass of another.

        Args:
            class_name: Name of the class to declare.
            superclass: Name of the parent class, or ``None`` for a root class.
        """
        logging.info(f"  adding class {class_name} as subclass of {superclass}..")

        owl_class = parse_manchester(self.iri, class_name)

        self.onto.add_axiom(OWLDeclarationAxiom(owl_class))  # type: ignore[arg-type]

        if superclass is not None:
            owl_super = parse_manchester(self.iri, superclass)
            self.onto.add_axiom(OWLSubClassOfAxiom(owl_class, owl_super))

    def _validate_property_args(
        self,
        range_class: str | None,
        data_prop: bool,
        create_inverse: bool,
        inverse_name: str | None,
    ) -> None:
        if data_prop and create_inverse:
            raise ValueError(
                "Data properties cannot have inverses; set create_inverse to False for data properties."
            )
        if inverse_name is not None and not create_inverse:
            raise ValueError(
                "When adding an inverse via inverse_name create_inverse must be set to true!"
            )
        if data_prop and range_class is not None and range_class not in _OWL_DATATYPES:
            raise ValueError(f"Unknown range for a data property: {range_class}")
        if not data_prop and range_class in _OWL_DATATYPES:
            raise ValueError("Can not create an object property with a data as range!")

    def add_property(
        self,
        prop_name: str,
        domain_class: str | None = None,
        range_class: str | None = None,
        superprop: str | None = None,
        create_inverse: bool = False,
        inverse_name: str | None = None,
        data_prop: bool = False,
    ) -> None:
        """Add a property with optional domain, range, super-property, and inverse.

        Raises:
            ValueError: For invalid combinations (data property with inverse, unknown
                data-property range, object property with datatype range).
        """
        logging.info(f"  adding property {prop_name} ..")
        self._validate_property_args(
            range_class, data_prop, create_inverse, inverse_name
        )

        if data_prop:
            prop_class: Any = OWLDataProperty
            super_axiom: Any = OWLSubDataPropertyOfAxiom
            domain_axiom: Any = OWLDataPropertyDomainAxiom
            range_axiom: Any = OWLDataPropertyRangeAxiom
        else:
            prop_class = OWLObjectProperty
            super_axiom = OWLSubObjectPropertyOfAxiom
            domain_axiom = OWLObjectPropertyDomainAxiom
            range_axiom = OWLObjectPropertyRangeAxiom

        owl_prop = prop_class(self.iri + prop_name)
        self.onto.add_axiom(OWLDeclarationAxiom(owl_prop))

        if superprop is not None:
            owl_super = prop_class(self.iri + superprop)
            self.onto.add_axiom(super_axiom(owl_prop, owl_super))

        if domain_class is not None:
            owl_domain = parse_manchester(self.iri, domain_class)
            self.onto.add_axiom(domain_axiom(owl_prop, owl_domain))

        if range_class is not None:
            if range_class in _OWL_DATATYPES:
                owl_range = OWLDatatype(
                    IRI("http://www.w3.org/2001/XMLSchema#", range_class)
                )
            else:
                owl_range = parse_manchester(self.iri, range_class)  # type: ignore[assignment]
            self.onto.add_axiom(range_axiom(owl_prop, owl_range))

        if create_inverse:
            if inverse_name is None:
                inverse_name = self.generate_inverse_property_name(prop_name)
            self._named_inverse_map[prop_name] = inverse_name
            self._named_inverse_map[inverse_name] = prop_name
            owl_inverse = prop_class(self.iri + inverse_name)
            self.onto.add_axiom(OWLDeclarationAxiom(owl_inverse))
            if InverseRepresentation.AXIOMATIC in self.settings.inverse_representation:
                self.onto.add_axiom(
                    OWLInverseObjectPropertiesAxiom(owl_prop, owl_inverse)
                )
            if superprop is not None:
                super_inv_name = self.generate_inverse_property_name(superprop)
                owl_super_inv = prop_class(self.iri + super_inv_name)
                self.onto.add_axiom(
                    OWLSubObjectPropertyOfAxiom(owl_inverse, owl_super_inv)
                )

    def assert_individual(self, individual: str, class_name: str) -> None:
        """Assert that a single named individual belongs to a class.

        Args:
            individual: IRI local name of the individual.
            class_name: Name of the OWL class.
        """
        owl_class = parse_manchester(self.iri, class_name)
        owl_individual = OWLNamedIndividual(f"{self.iri}{individual}")
        self.onto.add_axiom(OWLClassAssertionAxiom(owl_individual, owl_class))

    def assert_individuals(self, individuals: list[str], class_name: str) -> None:
        """Assert that a list of named individuals all belong to a class.

        Duplicates in ``individuals`` are silently removed before asserting.

        Args:
            individuals: IRI local names of the individuals.
            class_name: Name of the OWL class.
        """
        # dict.fromkeys preserves insertion order while deduplicating
        individuals = list(dict.fromkeys(individuals))
        logging.info(
            f"  asserting {len(individuals)} individuals of class {class_name}.."
        )
        owl_class = parse_manchester(self.iri, class_name)
        iri = self.iri
        axioms = [
            OWLClassAssertionAxiom(OWLNamedIndividual(f"{iri}{individual}"), owl_class)
            for individual in individuals
        ]
        self.onto.add_axiom(axioms)

    def assert_relation(
        self,
        prop_name: str,
        from_name: str,
        to: str | int | bool | float,
        data_prop: bool = False,
        inverse_name: str | None = None,
    ) -> None:
        """Assert a single property between two individuals or an individual and a literal.

        Args:
            prop_name: Name of the property to assert.
            from_name: IRI local name of the subject individual.
            to: Object individual name (str, for object properties) or literal value
                (int/bool/float, for data properties).
            data_prop: If True, assert as a data property; otherwise as an object property.
            inverse_name: Explicit inverse property name. If ``None`` and
                ``IMPLICIT`` inverse representation is active, ``inv_{prop_name}`` is used.

        Raises:
            ValueError: For type mismatches between ``to`` and ``data_prop``.
        """
        if not isinstance(to, str) and not data_prop:
            raise ValueError(
                "When creating an object property the type of the to must be str."
            )
        if isinstance(to, str) and data_prop:
            raise ValueError(
                "When creating an data property the type of the to must not be str."
            )

        if data_prop:
            owl_prop = OWLDataProperty(self.iri + prop_name)
            owl_from = OWLNamedIndividual(self.iri + from_name)
            owl_to = OWLLiteral(to)  # type: ignore[abstract]
            self.onto.add_axiom(
                OWLDataPropertyAssertionAxiom(owl_from, owl_prop, owl_to)
            )
        else:
            owl_prop = OWLObjectProperty(self.iri + prop_name)  # type: ignore[assignment]
            owl_from = OWLNamedIndividual(self.iri + from_name)
            owl_to = OWLNamedIndividual(self.iri + to)  # type: ignore[assignment, operator]
            self.onto.add_axiom(
                OWLObjectPropertyAssertionAxiom(owl_from, owl_prop, owl_to)  # type: ignore[arg-type]
            )

            if InverseRepresentation.IMPLICIT in self.settings.inverse_representation:
                inv = inverse_name if inverse_name is not None else "inv_" + prop_name
                inv_prop = OWLObjectProperty(self.iri + inv)
                self.onto.add_axiom(
                    OWLObjectPropertyAssertionAxiom(owl_to, inv_prop, owl_from)  # type: ignore[arg-type]
                )

    def assert_relations(
        self,
        df: pd.DataFrame,
        prop: str,
        from_key: str,
        to_key: str,
        data_prop: bool = False,
        inverse_name: str | None = None,
    ) -> None:
        """Assert property relations from a DataFrame, dropping nulls and duplicates.

        Args:
            df: DataFrame containing at least ``from_key`` and ``to_key`` columns.
            prop: Property name to assert.
            from_key: Column name for the subject IRI local names.
            to_key: Column name for the object IRI local names or literal values.
            data_prop: If True, assert as data property assertions.
            inverse_name: Passed through to :meth:`assert_object_relations`.
        """
        df = df[[from_key, to_key]].copy()
        df = df[df[from_key].notna() & df[to_key].notna()]
        df = df.drop_duplicates([from_key, to_key])

        if data_prop:
            self.assert_data_relations(df, prop, from_key, to_key)
        else:
            self.assert_object_relations(df, prop, from_key, to_key, inverse_name)

    def assert_object_relations(
        self,
        df: pd.DataFrame,
        prop: str,
        from_key: str,
        to_key: str,
        inverse_name: str | None = None,
    ) -> None:
        """Assert object property axioms from every row of a DataFrame.

        If ``IMPLICIT`` inverse representation is active, the inverse relation is
        also asserted for each pair.

        Args:
            df: DataFrame with subject/object columns (already deduplicated).
            prop: Object property name.
            from_key: Subject column name (IRI local names).
            to_key: Object column name (IRI local names).
            inverse_name: Explicit inverse property name. Defaults to ``inv_{prop}``.
        """
        logging.info(f"  asserting {len(df)} relations of object property {prop}..")
        iri = self.iri
        owl_prop = OWLObjectProperty(f"{iri}{prop}")
        axioms = [
            OWLObjectPropertyAssertionAxiom(
                OWLNamedIndividual(f"{iri}{s}"),
                owl_prop,
                OWLNamedIndividual(f"{iri}{o}"),
            )
            for s, o in df.itertuples(index=False, name=None)
        ]
        if axioms:
            self.onto.add_axiom(axioms)

        if InverseRepresentation.IMPLICIT in self.settings.inverse_representation:
            inv = inverse_name if inverse_name is not None else "inv_" + prop
            inv_prop = OWLObjectProperty(f"{iri}{inv}")
            inverse_axioms = [
                OWLObjectPropertyAssertionAxiom(
                    OWLNamedIndividual(f"{iri}{o}"),
                    inv_prop,
                    OWLNamedIndividual(f"{iri}{s}"),
                )
                for s, o in df.itertuples(index=False, name=None)
            ]
            if inverse_axioms:
                self.onto.add_axiom(inverse_axioms)

    def assert_data_relations(
        self, df: pd.DataFrame, prop: str, from_key: str, to_key: str
    ) -> None:
        """Assert data property axioms from every row of a DataFrame.

        Datetime columns are converted to Python ``datetime`` objects before
        wrapping in ``OWLLiteral``.

        Args:
            df: DataFrame with subject/literal columns (already deduplicated).
            prop: Data property name.
            from_key: Subject column name (IRI local names).
            to_key: Literal value column name.
        """
        logging.info(f"  asserting {len(df)} relations of data property {prop}..")
        iri = self.iri
        owl_prop = OWLDataProperty(f"{iri}{prop}")

        if pd.api.types.is_datetime64_any_dtype(df[to_key]):
            axioms = []
            for s, v in df.itertuples(index=False, name=None):
                if pd.isna(v):
                    continue
                # v may be numpy.datetime64[us] or Timestamp; convert to Python datetime for OWLLiteral
                py_dt = (
                    v.to_pydatetime()
                    if hasattr(v, "to_pydatetime")
                    else pd.Timestamp(v).to_pydatetime()
                )
                axioms.append(
                    OWLDataPropertyAssertionAxiom(
                        OWLNamedIndividual(f"{iri}{s}"),
                        owl_prop,
                        OWLLiteral(py_dt),  # type: ignore[abstract]
                    )
                )
        else:
            axioms = [
                OWLDataPropertyAssertionAxiom(
                    OWLNamedIndividual(f"{iri}{s}"),
                    owl_prop,
                    OWLLiteral(v),  # type: ignore[abstract]
                )
                for s, v in df.itertuples(index=False, name=None)
                if not (isinstance(v, float) and pd.isna(v))
            ]

        if axioms:
            self.onto.add_axiom(axioms)

    def generate_inverse_property_name(self, prop: str) -> str:
        """Return the inverse property name for ``prop``.

        Uses the named-inverse map (seeded from the upper ontology and extended
        during ``add_property`` calls) when available; falls back to ``inv_{prop}``.
        """
        return self._named_inverse_map.get(prop, "inv_" + prop)

    def generate_specialized_has_attribute_name(self, attribute: str) -> str:
        """Return the ``has_attribute`` sub-property name for ``attribute``."""
        return "has_" + attribute

    def generate_attr_change_value_prop_name(self, attribute: str) -> str:
        """Return the ``attribute_change_value`` sub-property name for ``attribute``."""
        return "change_value_" + attribute

    def generate_timestamp_string(self, timestamp: pd.Timestamp | str) -> str:
        """Convert a timestamp to an IRI-safe string.

        Args:
            timestamp: A pandas ``Timestamp`` or an arbitrary string representation.

        Returns:
            ``YYYYMMDDTHHmmSS_ffffff`` for Timestamps (microseconds included to
            avoid IRI collisions for sub-second events), or the input with all
            non-alphanumeric characters replaced by ``_``.
        """
        if isinstance(timestamp, pd.Timestamp):
            # Include microseconds to avoid IRI collisions for sub-second events.
            return timestamp.strftime("%Y%m%dT%H%M%S_%f")
        return re.sub(r"[^A-Za-z0-9]", "_", str(timestamp))

    def save_onto(self, path: str) -> None:
        """Save the current ontology to an OWL file.

        Args:
            path: Destination file path.
        """
        self.onto.save(path)
