import tempfile
import unittest
from pathlib import Path

from rdflib import RDF, Graph, Namespace, URIRef

from semanticlog.onto.builder import OCEDBaseBuilder
from semanticlog.onto.reason import Reasoner
from semanticlog.onto.settings import DEFAULT_IRI, OntoBuildSettings, OwlProfile


class TestReasoner(unittest.TestCase):
    def setUp(self):
        self.settings = OntoBuildSettings()
        self.settings.profile = OwlProfile.OWL2EL
        self.settings.order_representation = []
        self.settings.case_concepts = []
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_materialize(self):
        base_iri = DEFAULT_IRI
        out_path = str(Path(self._tmp.name) / "temp.owl")

        builder = OCEDBaseBuilder(base_iri, self.settings)

        # TBox
        builder.add_class("Human")
        builder.add_class("Parent", "Human")
        builder.add_class("Male")
        builder.add_class("Female")
        builder.add_class("Man", "Human and Male")
        builder.add_class("Woman", "Human and Female")

        # ABox
        individuals = ["a", "b", "c", "d", "e"]
        for ind in individuals:
            builder.assert_individual(ind, "Human")
        builder.assert_individual("a", "Man")
        builder.assert_individual("b", "Man")
        builder.assert_individual("c", "Man")
        builder.assert_individual("d", "Woman")
        builder.assert_individual("e", "Woman")

        builder.save_onto(out_path)

        # Materialize
        reasoner = Reasoner(out_path)
        reasoner.materialize()
        reasoner.save_onto()

        # Load a plain rdflib Graph from the saved file
        rg = Graph()
        rg.parse(out_path)  # auto-detects format

        for s, p, o in rg:
            print(s, p, o)

        Namespace(base_iri)

        def u(local: str) -> URIRef:
            return URIRef(base_iri + local)

        def has_triple(s, p, o) -> bool:
            return any(rg.triples((s, p, o)))

        # Everyone is Human
        for ind in individuals:
            self.assertTrue(
                has_triple(u(ind), RDF.type, u("Human")), f"Missing {ind} a Human"
            )

        # a,b,c are Man and (inferred) Male
        for ind in ["a", "b", "c"]:
            self.assertTrue(
                has_triple(u(ind), RDF.type, u("Man")), f"Missing {ind} a Man"
            )
            self.assertTrue(
                has_triple(u(ind), RDF.type, u("Male")), f"Missing {ind} a Male"
            )

        # d,e are Woman and (inferred) Female
        for ind in ["d", "e"]:
            self.assertTrue(
                has_triple(u(ind), RDF.type, u("Woman")), f"Missing {ind} a Woman"
            )
            self.assertTrue(
                has_triple(u(ind), RDF.type, u("Female")), f"Missing {ind} a Female"
            )

        # a,b,c should NOT be Female
        for ind in ["a", "b", "c"]:
            self.assertFalse(
                has_triple(u(ind), RDF.type, u("Female")), f"Unexpected {ind} a Female"
            )
