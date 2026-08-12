"""Running a reasoner over a built ontology.

Reasoning is a separate step from building. It reads an `.owl` file through
the Java OWL API, which needs a JVM, and can either report whether the file is
consistent or write back everything the reasoner entails.
"""

import logging

from owlapy.owl_axiom import OWLObjectPropertyAssertionAxiom
from owlapy.owl_ontology import SyncOntology
from owlapy.owl_reasoner import SyncReasoner

from semanticlog.onto.settings import OWLProfile

logger = logging.getLogger(__name__)

_GENERATORS_ALL = [
    "InferredClassAssertionAxiomGenerator",
    "InferredSubClassAxiomGenerator",
    "InferredDisjointClassesAxiomGenerator",
    "InferredEquivalentClassAxiomGenerator",
    "InferredEquivalentDataPropertiesAxiomGenerator",
    "InferredEquivalentObjectPropertyAxiomGenerator",
    "InferredInverseObjectPropertiesAxiomGenerator",
    "InferredSubDataPropertyAxiomGenerator",
    "InferredSubObjectPropertyAxiomGenerator",
]

_GENERATORS_DL_ONLY = [
    "InferredDataPropertyCharacteristicAxiomGenerator",
    "InferredObjectPropertyCharacteristicAxiomGenerator",
]


class Reasoner:
    """A reasoner bound to one ontology file.

    Attributes:
        path: File the ontology was loaded from.
        ontology: The loaded ontology.
        reasoner: The backend doing the work.
        possible_generators: Which sorts of entailment to ask for.
    """

    def __init__(
        self,
        path: str,
        backend: str = "HermiT",
        profile: OWLProfile = OWLProfile.OWL2DL,
    ):
        """Load an ontology and prepare a reasoner over it.

        Args:
            path: Ontology file to read.
            backend: One of HermiT, Pellet or ELK.
            profile: Profile the ontology was built for. Under OWL 2 EL the
                property characteristic entailments are not asked for, since
                that profile has no axioms for them.

        Raises:
            ValueError: If the backend is not one of the three supported.
        """
        if backend not in ["HermiT", "Pellet", "ELK"]:
            raise ValueError(f"Unsupported reasoner backend: {backend}")

        self.path = path
        self.ontology = SyncOntology(self.path)
        self.reasoner = SyncReasoner(self.ontology, reasoner=backend)
        self.possible_generators = list(_GENERATORS_ALL)
        if profile != OWLProfile.OWL2EL:
            self.possible_generators.extend(_GENERATORS_DL_ONLY)

    def materialize(
        self, overwrite_old: bool = True, property_assertions: bool = True
    ) -> None:
        """Add everything the reasoner entails to the ontology.

        Args:
            overwrite_old: Save the result back over the file it came from.
            property_assertions: Also add the entailed relations between
                individuals, so a transitive property such as
                eventually-follows can be read without a reasoner. This is the
                expensive part.
        """
        inferred = list(
            self.reasoner.infer_axioms(inference_types=self.possible_generators)
        )
        if property_assertions:
            inferred.extend(self.inferred_property_assertions())
        self.ontology.add_axiom(inferred)
        if overwrite_old:
            self.save()

    def inferred_property_assertions(self) -> list[OWLObjectPropertyAssertionAxiom]:
        """Ask the reasoner for every entailed relation between individuals.

        Every individual is queried against every object property, so the cost
        grows with the product of the two.

        Returns:
            The entailed assertions.
        """
        individuals = list(self.ontology.individuals_in_signature())
        properties = list(self.ontology.object_properties_in_signature())
        logger.info(
            f"  querying {len(individuals)} individuals x {len(properties)} "
            "object properties for entailed assertions.."
        )
        axioms = [
            OWLObjectPropertyAssertionAxiom(individual, prop, value)
            for individual in individuals
            for prop in properties
            for value in self.reasoner.object_property_values(individual, prop)
        ]
        logger.info(f"  {len(axioms)} entailed object property assertion(s)")
        return axioms

    def check(self) -> bool:
        """Report whether the ontology is consistent."""
        return self.reasoner.has_consistent_ontology()

    def save(self, save_path: str | None = None) -> None:
        """Write the ontology out, over the source file unless told otherwise."""
        if save_path is None:
            save_path = self.path
        self.ontology.save(save_path)
