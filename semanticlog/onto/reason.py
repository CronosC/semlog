from owlapy.owl_ontology import SyncOntology
from owlapy.owl_axiom import OWLDeclarationAxiom
from owlapy.class_expression import OWLClass
from owlapy.owl_reasoner import SyncReasoner
from owlapy.static_funcs import stopJVM, startJVM
from semanticlog.onto.settings import OwlProfile
import os


class Reasoner:

    def __init__(self, path, backend="HermiT", profile=OwlProfile.OWL2EL):
        if backend not in ["HermiT", "Pellet", "ELK"]:
            raise ValueError("Unsupported reasoner backend:", backend)
        
        self.path = path
        self.ontology = SyncOntology(self.path)
        self.reasoner = SyncReasoner(self.ontology, reasoner=backend)
        self.possible_generators = [
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
        if profile != OwlProfile.OWL2EL:
            self.possible_generators.extend(
                [
                    "InferredDataPropertyCharacteristicAxiomGenerator",
                    "InferredObjectPropertyCharacteristicAxiomGenerator",
                ]
            )

    def materialize(self, overwrite_old=True):
        inferred_axioms = self.reasoner.infer_axioms(
            inference_types=self.possible_generators
        )
        self.ontology.add_axiom(inferred_axioms)
        if overwrite_old:
            self.save_onto()

    def infer(self, generator):
        if generator not in self.possible_generators:
            raise ValueError("Unknown generator type!")
        else:
            return self.reasoner.infer_axioms(inference_types=[generator])

    def check(self):
        return self.reasoner.has_consistent_ontology()

    def save_onto(self, save_path=None):
        if save_path == None:
            save_path = self.path
        if not os.path.exists(save_path):
            with open(save_path, "w") as f:
                f.write("")
        self.ontology.save(save_path)
