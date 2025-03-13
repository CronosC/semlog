import unittest
from owlapy.class_expression import OWLClassExpression
from semanticlog.onto.upper import OCEDUpperOntoBuilder


class TestUpper(unittest.TestCase):
    
    def test_build_upper(self):
        path = "data/temp.owl"
        builder = OCEDUpperOntoBuilder("data/upper.owl#")
        builder.construct_upper_ontology(path)
        
        
if __name__ == "__main__":
    unittest.main()
