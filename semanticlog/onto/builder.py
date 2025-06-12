import datetime
import os
import re
import pandas as pd
import owlapy as owl
from owlapy.iri import IRI
from owlapy.owl_axiom import (
    OWLDeclarationAxiom,
    OWLSubClassOfAxiom,
    OWLSubObjectPropertyOfAxiom,
    OWLSubDataPropertyOfAxiom,
    OWLObjectPropertyRangeAxiom,
    OWLDataPropertyRangeAxiom,
    OWLObjectPropertyDomainAxiom,
    OWLDataPropertyDomainAxiom,
    OWLInverseObjectPropertiesAxiom,
    OWLClassAssertionAxiom,
    OWLObjectPropertyAssertionAxiom,
    OWLDataPropertyAssertionAxiom,
)
from owlapy.owl_literal import (
    OWLLiteral,
    IntegerOWLDatatype,
    FloatOWLDatatype,
    BooleanOWLDatatype,
)
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_ontology_manager import OntologyManager
from owlapy.class_expression import OWLClass
from owlapy.owl_property import OWLObjectProperty, OWLDataProperty
from owlapy.owl_datatype import OWLDatatype
import logging
import owlready2

from semanticlog.onto.transform import parse_manchester


class OCEDBaseBuilder:

    def __init__(self, iri):
        self.manager = OntologyManager()
        try:
            self.onto = self.manager.load_ontology(iri)
            self.iri = self.onto.get_ontology_id().get_ontology_iri().str
        except:
            self.onto = self.manager.create_ontology(iri)
            self.iri = iri
        if not self.iri.endswith("#"):
            self.iri = self.iri + "#"

    def add_classes(
        self, class_list: list[str], superclasses: list[str] | str | None = None
    ) -> None:
        logging.info(f"  adding {len(class_list)} classes..")
        if type(superclasses) == str or superclasses is None:
            for class_name in class_list:
                self.add_class(class_name, superclasses)
        elif type(superclasses) == list and len(class_list) == len(superclasses):
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
        datatypes = ["integer", "double", "boolean"]  # TODO datetime?
        if data_prop and create_inverse:
            raise ValueError(
                "Data properties cannot have inverses; set create_inverse to False for data properties."
            )
        if inverse_name != None and not create_inverse:
            raise ValueError(
                "When adding an inverse via inverse_name create_inverse must be set to true!"
            )
        if data_prop and range_class not in datatypes and range_class != None:
            raise ValueError("Unknown range for a data property:", range_class)
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
            if inverse_name == None:
                inverse_name = self.generate_inverse_property_name(prop_name)
            owl_inverse = prop_class(self.iri + inverse_name)
            self.onto.add_axiom(OWLDeclarationAxiom(owl_inverse))
            self.onto.add_axiom(OWLInverseObjectPropertiesAxiom(owl_prop, owl_inverse))

    def assert_individual(self, individual: str, class_name: str) -> None:
        owl_class = parse_manchester(self.iri, class_name)
        owl_individual = OWLNamedIndividual(f"{self.iri}{individual}")
        self.onto.add_axiom(OWLClassAssertionAxiom(owl_individual, owl_class))

    def assert_individuals(self, individuals: list[str], class_name: str) -> None:
        logging.info(f"  asserting {len(individuals)} individuals of class {class_name}..")
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
        if type(to) != str and not data_prop:
            raise ValueError(
                "When creating an object property the type of the to must be str."
            )
        if type(to) == str and data_prop:
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

    def assert_relations(
        self,
        dataframe: pd.DataFrame,
        prop: str,
        from_key: str,
        to_key: str,
        data_prop=False,
    ) -> None:
        logging.info(f"  asserting {len(dataframe)} relations of property {prop}..")
        iri = self.iri
        
        dataframe = dataframe[~(dataframe[from_key].isin([None, "", pd.NA]) | dataframe[to_key].isin([None, "", pd.NA]))]
        
        if data_prop:
            owl_prop = OWLDataProperty(f"{iri}{prop}")
            dataframe[to_key] = dataframe[to_key].dropna()
            axioms = [
                OWLDataPropertyAssertionAxiom(
                    OWLNamedIndividual(f"{iri}{from_name}"),
                    owl_prop,
                    OWLLiteral(to),
                )
                for from_name, to in zip(dataframe[from_key], dataframe[to_key])
            ]
        else:
            owl_prop = OWLObjectProperty(f"{iri}{prop}")
            axioms = [
                OWLObjectPropertyAssertionAxiom(
                    OWLNamedIndividual(f"{iri}{from_name}"),
                    owl_prop,
                    OWLNamedIndividual(f"{iri}{to}"),
                )
                for from_name, to in zip(dataframe[from_key], dataframe[to_key])
            ]
        self.onto.add_axiom(axioms)

    #### HELPER ####

    def generate_inverse_property_name(self, prop: str) -> str:
        return "inv_" + prop
    
    def generate_spezialized_has_attribute_name(self, attribute: str) -> str:
        return "has_" + attribute
    
    def generate_timestamp_string(self, timestamp: pd.Timestamp | str) -> str:
        if type(timestamp) != str:
            timestamp =  str(timestamp)
        return re.sub(r'[^A-Za-z0-9]', '_',  timestamp)

    def save_onto(self, path):
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write("")
        self.onto.save(path)
