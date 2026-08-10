import logging

from owlapy.owl_axiom import OWLObjectPropertyAssertionAxiom
from owlapy.owl_ontology import SyncOntology
from owlapy.owl_reasoner import SyncReasoner

from semanticlog.onto.settings import OwlProfile

# Generators supported by all profiles
_GENERATORS_ALL = [
    "InferredClassAssertionAxiomGenerator",
    "InferredSubClassAxiomGenerator",
    "InferredDisjointClassesAxiomGenerator",
    "InferredEquivalentClassAxiomGenerator",
    "InferredEquivalentDataPropertiesAxiomGenerator",
    "InferredEquivalentObjectPropertyAxiomGenerator",
    "InferredInverseObjectPropertiesAxiomGenerator",
    "InferredPropertyAssertionGenerator",
    "InferredSubDataPropertyAxiomGenerator",
    "InferredSubObjectPropertyAxiomGenerator",
]

# Generators that require OWL 2 DL semantics (not valid in OWL 2 EL)
_GENERATORS_DL_ONLY = [
    "InferredDataPropertyCharacteristicAxiomGenerator",
    "InferredObjectPropertyCharacteristicAxiomGenerator",
]


class Reasoner:
    """Thin wrapper around owlapy's SyncReasoner for materialising and checking ontologies.

    Loads an OWL file from disk, runs inference using the selected backend, and
    can save the result back in place or to a new path.
    """

    def __init__(
        self,
        path: str,
        backend: str = "HermiT",
        profile: OwlProfile = OwlProfile.OWL2EL,
    ):
        """Load an ontology and initialise the reasoner.

        Args:
            path: Path to the OWL file to load.
            backend: Reasoner backend to use. One of ``"HermiT"``, ``"Pellet"``, ``"ELK"``.
            profile: OWL profile of the ontology.

        Raises:
            ValueError: If ``backend`` is not one of the supported values.
        """
        if backend not in ["HermiT", "Pellet", "ELK"]:
            raise ValueError(f"Unsupported reasoner backend: {backend}")

        self.path = path
        self.ontology = SyncOntology(self.path)
        self.reasoner = SyncReasoner(self.ontology, reasoner=backend)
        self.possible_generators = list(_GENERATORS_ALL)
        if profile != OwlProfile.OWL2EL:
            self.possible_generators.extend(_GENERATORS_DL_ONLY)

    def materialize(
        self, overwrite_old: bool = True, property_assertions: bool = True
    ) -> None:
        """Run all applicable inference generators and add the results to the ontology.

        Args:
            overwrite_old: If True (default), save the materialised ontology back to
                the original path after inference.
            property_assertions: Also write the entailed object property assertions
                that the OWL API generator omits (see
                :meth:`inferred_property_assertions`). On by default, because
                without them a materialised ontology is missing exactly the
                consequences a reasoner was run for. Turn off for a cheap
                class-level-only pass.
        """
        inferred = list(
            self.reasoner.infer_axioms(inference_types=self.possible_generators)
        )
        if property_assertions:
            inferred.extend(self.inferred_property_assertions())
        self.ontology.add_axiom(inferred)
        if overwrite_old:
            self.save_onto()

    def inferred_property_assertions(self) -> list[OWLObjectPropertyAssertionAxiom]:
        """Return the object property assertions the reasoner entails.

        ``InferredPropertyAssertionGenerator`` returns nothing through owlapy, so
        transitive and sub-property consequences never reach a materialised file.
        That silently empties out the point of
        :attr:`~semanticlog.onto.settings.EventOrderRepresentation.EF`, whose whole
        design is to let a reasoner derive the eventually-follows closure.

        Asking the reasoner directly is the only way to recover them, and it costs
        one query per individual per object property. That product, not the
        reasoning itself, is what makes materialisation unsuited to large logs.
        """
        individuals = list(self.ontology.individuals_in_signature())
        properties = list(self.ontology.object_properties_in_signature())
        logging.info(
            f"  querying {len(individuals)} individuals x {len(properties)} "
            "object properties for entailed assertions.."
        )
        axioms = [
            OWLObjectPropertyAssertionAxiom(individual, prop, value)
            for individual in individuals
            for prop in properties
            for value in self.reasoner.object_property_values(individual, prop)
        ]
        logging.info(f"  {len(axioms)} entailed object property assertion(s)")
        return axioms

    def infer(self, generator: str):
        """Run a single named inference generator and return the resulting axioms.

        Args:
            generator: One of the generator names in ``self.possible_generators``
                (e.g. ``"InferredClassAssertionAxiomGenerator"``).

        Returns:
            Inferred axioms produced by the generator.

        Raises:
            ValueError: If ``generator`` is not in ``self.possible_generators``.
        """
        if generator not in self.possible_generators:
            raise ValueError("Unknown generator type!")
        else:
            return self.reasoner.infer_axioms(inference_types=[generator])

    def check(self) -> bool:
        """Return True if the ontology is consistent, False otherwise."""
        return self.reasoner.has_consistent_ontology()

    def save_onto(self, save_path: str | None = None) -> None:
        """Save the ontology to disk.

        Args:
            save_path: Destination path. Defaults to the path the ontology was
                loaded from (overwrite in place).
        """
        if save_path is None:
            save_path = self.path
        self.ontology.save(save_path)
