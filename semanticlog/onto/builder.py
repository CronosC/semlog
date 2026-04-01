import logging
import os
import re

import pandas as pd
from owlapy.iri import IRI
from owlapy.owl_axiom import (
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

from semanticlog.onto.settings import OntoBuildSettings, OwlProfile
from semanticlog.onto.transform import parse_manchester


class OCEDBaseBuilder:

    def __init__(self, iri: str, settings: OntoBuildSettings):
        try:
            self.onto = Ontology(IRI.create(iri), load=True)
            self.iri = self.onto.get_ontology_id().get_ontology_iri().str
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
        self.inverse_dict: dict[str, str] = {}

    def add_classes(
        self, class_list: list[str], superclasses: list[str] | str | None = None
    ) -> None:
        logging.info(f"  adding {len(class_list)} classes..")
        if isinstance(superclasses, str) or superclasses is None:
            for class_name in class_list:
                self.add_class(class_name, superclasses)
        elif isinstance(superclasses, list) and len(class_list) == len(superclasses):
            for class_name, superclass in zip(class_list, superclasses):
                self.add_class(class_name, superclass)
        else:
            raise ValueError(
                "Invalid argument for superclasses! Must be list of str with the same length as the class_list , str or None"
            )

    def add_class(self, class_name: str, superclass: str | None = None) -> None:
        logging.info(f"  adding class {class_name} as subclass of {superclass}..")

        owl_class = parse_manchester(self.iri, class_name)

        self.onto.add_axiom(OWLDeclarationAxiom(owl_class))

        if superclass is not None:
            owl_super = parse_manchester(self.iri, superclass)
            self.onto.add_axiom(OWLSubClassOfAxiom(owl_class, owl_super))

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
        """
        Add a property to the ontology with the specified axioms.

        Parameters:
            prop_name (str): The name of the property.
            domain (Optional[str]): The domain class for the property.
            range (Optional[str]): The range class for the property.
            superprop (Optional[str]): The name of a super-property.
            create_inverse (bool): If True, an inverse property will be created (object properties only).
            data_prop (bool): If True, the property is a data property; otherwise, it's an object property.

        Raises:
            ValueError: If a data property is specified with create_inverse=True.
        """
        logging.info(f"  adding property {prop_name} ..")
        datatypes = ["integer", "double", "boolean", "dateTime"]
        if data_prop and create_inverse:
            raise ValueError(
                "Data properties cannot have inverses; set create_inverse to False for data properties."
            )
        if inverse_name is not None and not create_inverse:
            raise ValueError(
                "When adding an inverse via inverse_name create_inverse must be set to true!"
            )
        if data_prop and range_class not in datatypes and range_class is not None:
            raise ValueError(f"Unknown range for a data property: {range_class}")
        if not data_prop and range_class in datatypes:
            raise ValueError("Can not create an object property with a data as range!")

        # Map property type to the associated functions.
        property_funcs = {
            True: {
                "creation": OWLDataProperty,
                "super_axiom": OWLSubDataPropertyOfAxiom,
                "domain_axiom": OWLDataPropertyDomainAxiom,
                "range_axiom": OWLDataPropertyRangeAxiom,
            },
            False: {
                "creation": OWLObjectProperty,
                "super_axiom": OWLSubObjectPropertyOfAxiom,
                "domain_axiom": OWLObjectPropertyDomainAxiom,
                "range_axiom": OWLObjectPropertyRangeAxiom,
            },
        }[data_prop]

        prop_class = property_funcs["creation"]
        owl_prop = prop_class(self.iri + prop_name)
        self.onto.add_axiom(OWLDeclarationAxiom(owl_prop))

        if superprop is not None:
            owl_super = prop_class(self.iri + superprop)
            self.onto.add_axiom(property_funcs["super_axiom"](owl_prop, owl_super))

        if domain_class is not None:
            owl_domain = parse_manchester(self.iri, domain_class)
            self.onto.add_axiom(property_funcs["domain_axiom"](owl_prop, owl_domain))

        if range_class is not None:
            if range_class in datatypes:
                owl_range = OWLDatatype(
                    IRI("http://www.w3.org/2001/XMLSchema#", range_class)
                )
            else:
                owl_range = parse_manchester(self.iri, range_class)
            self.onto.add_axiom(property_funcs["range_axiom"](owl_prop, owl_range))

        if create_inverse:
            if inverse_name is None:
                inverse_name = self.generate_inverse_property_name(prop_name)
                self.inverse_dict[prop_name] = inverse_name
            owl_inverse = prop_class(self.iri + inverse_name)
            self.onto.add_axiom(OWLDeclarationAxiom(owl_inverse))
            if self.settings.profile == OwlProfile.OWL2DL:
                self.onto.add_axiom(
                    OWLInverseObjectPropertiesAxiom(owl_prop, owl_inverse)
                )

    def assert_individual(self, individual: str, class_name: str) -> None:
        owl_class = parse_manchester(self.iri, class_name)
        owl_individual = OWLNamedIndividual(f"{self.iri}{individual}")
        self.onto.add_axiom(OWLClassAssertionAxiom(owl_individual, owl_class))

    def assert_individuals(self, individuals: list[str], class_name: str) -> None:
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
        data_prop=False,
    ) -> None:
        if not isinstance(to, str) and not data_prop:
            raise ValueError(
                "When creating an object property the type of the to must be str."
            )
        if isinstance(to, str) and data_prop:
            raise ValueError(
                "When creating an data property the type of the to must not be str."
            )
        funcs = {
            True: {
                "prop": OWLDataProperty,
                "assert": OWLDataPropertyAssertionAxiom,
            },
            False: {
                "prop": OWLObjectProperty,
                "assert": OWLObjectPropertyAssertionAxiom,
            },
        }[data_prop]

        owl_prop = funcs["prop"](self.iri + prop_name)
        owl_from = OWLNamedIndividual(self.iri + from_name)
        if data_prop:
            owl_to = OWLLiteral(to)
        else:
            owl_to = OWLNamedIndividual(self.iri + to)
        self.onto.add_axiom(funcs["assert"](owl_from, owl_prop, owl_to))

        if self.settings.make_inverses_explicit and not data_prop:
            inverse_prop = self.inverse_dict.get(prop_name, None)
            if inverse_prop is not None:
                inverse_prop = funcs["prop"](self.iri + inverse_prop)
                self.onto.add_axiom(funcs["assert"](owl_to, inverse_prop, owl_from))

    def assert_relations(
        self,
        df: pd.DataFrame,
        prop: str,
        from_key: str,
        to_key: str,
        data_prop: bool = False,
    ) -> None:
        df = df[[from_key, to_key]].copy()
        df = df[df[from_key].notna() & df[to_key].notna()]
        df = df.drop_duplicates([from_key, to_key])

        if data_prop:
            self.assert_data_relations(df, prop, from_key, to_key)
        else:
            self.assert_object_relations(df, prop, from_key, to_key)

    def assert_object_relations(
        self, df: pd.DataFrame, prop: str, from_key: str, to_key: str
    ) -> None:
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

        if self.settings.make_inverses_explicit:
            inverse_prop = self.inverse_dict.get(prop, None)
            if inverse_prop is not None:
                inverse_prop = OWLObjectProperty(f"{iri}{inverse_prop}")
                inverse_axioms = [
                    OWLObjectPropertyAssertionAxiom(
                        OWLNamedIndividual(f"{iri}{o}"),
                        inverse_prop,
                        OWLNamedIndividual(f"{iri}{s}"),
                    )
                    for s, o in df.itertuples(index=False, name=None)
                ]
                if inverse_axioms:
                    self.onto.add_axiom(inverse_axioms)

    def assert_data_relations(
        self, df: pd.DataFrame, prop: str, from_key: str, to_key: str
    ) -> None:
        logging.info(f"  asserting {len(df)} relations of data property {prop}..")
        iri = self.iri
        owl_prop = OWLDataProperty(f"{iri}{prop}")

        rel_df = df[[from_key, to_key]].copy()

        if pd.api.types.is_datetime64_any_dtype(rel_df[to_key]):
            axioms = []
            for s, v in rel_df.itertuples(index=False, name=None):
                if pd.isna(v):
                    continue
                # v may be numpy.datetime64[us] or Timestamp; convert to Python datetime for OWLLIteral class
                py_dt = (
                    v.to_pydatetime()
                    if hasattr(v, "to_pydatetime")
                    else pd.Timestamp(v).to_pydatetime()
                )
                axioms.append(
                    OWLDataPropertyAssertionAxiom(
                        OWLNamedIndividual(f"{iri}{s}"),
                        owl_prop,
                        OWLLiteral(py_dt),
                    )
                )
        else:
            axioms = [
                OWLDataPropertyAssertionAxiom(
                    OWLNamedIndividual(f"{iri}{s}"),
                    owl_prop,
                    OWLLiteral(v),
                )
                for s, v in rel_df.itertuples(index=False, name=None)
                if not (isinstance(v, float) and pd.isna(v))
            ]

        if axioms:
            self.onto.add_axiom(axioms)

    # HELPER

    def generate_inverse_property_name(self, prop: str) -> str:
        return "inv_" + prop

    def generate_spezialized_has_attribute_name(self, attribute: str) -> str:
        return "has_" + attribute

    def generate_timestamp_string(self, timestamp: pd.Timestamp | str) -> str:
        if isinstance(timestamp, pd.Timestamp):
            # Include microseconds to avoid IRI collisions for sub-second events.
            # Format: 20251013T114530_123456
            return timestamp.strftime("%Y%m%dT%H%M%S_%f")
        elif isinstance(timestamp, str):
            # sanitize any string timestamp (remove special chars)
            return re.sub(r"[^A-Za-z0-9]", "_", timestamp)
        else:
            # fallback for unexpected types
            return re.sub(r"[^A-Za-z0-9]", "_", str(timestamp))

    def save_onto(self, path):
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write("")
        self.onto.save(path)
