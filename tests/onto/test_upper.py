import unittest
from owlapy.class_expression import OWLClassExpression
from semanticlog.onto.upper import OCEDUpperOntoBuilder
from semanticlog.onto.settings import OntoBuildSettings, OwlProfile, EventOrderRepresentation


class TestUpper(unittest.TestCase):
    
    def setUp(self):
        self.settings = OntoBuildSettings()
        self.settings.profile = OwlProfile.OWL2EL
        self.settings.order_representation = []
        self.settings.case_concepts = []
    
    def test_build_upper(self):
        path = "data/temp.owl"
        builder = OCEDUpperOntoBuilder("data/upper.owl#", self.settings)
        builder.construct_upper_ontology(path)
        
        
if __name__ == "__main__":
    unittest.main()
