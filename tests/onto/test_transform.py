import unittest

from semanticlog.onto.transform import (
    sanitize,
    transform_dl_to_sparql,
    transform_manchester_to_sparql,
    unsanitize,
)


class TestTransformations(unittest.TestCase):

    def test_sanitize(self):
        input_string = "Class[%], Value[12], List[1,2,3]"
        expected_output = "ClassSQUAREBRACKETOPENINGPERCENTAGESIGNSQUAREBRACKETCLOSINGCOMMATA ValueSQUAREBRACKETOPENING12SQUAREBRACKETCLOSINGCOMMATA ListSQUAREBRACKETOPENING1COMMATA2COMMATA3SQUAREBRACKETCLOSING"
        self.assertEqual(sanitize(input_string), expected_output)

    def test_unsanitize(self):
        input_string = "ClassSQUAREBRACKETOPENINGPERCENTAGESIGNSQUAREBRACKETCLOSINGCOMMATA ValueSQUAREBRACKETOPENING12SQUAREBRACKETCLOSINGCOMMATA ListSQUAREBRACKETOPENING1COMMATA2COMMATA3SQUAREBRACKETCLOSING"
        expected_output = "Class[%], Value[12], List[1,2,3]"
        self.assertEqual(unsanitize(input_string), expected_output)

    def test_sanitization_full(self):
        for string in [
            "Class[%], Value[12], List[1,2,3]",
            "qwe,rtzuiopüa,sdfghjklöäyxcvbnm",
            '!"§$%&,/()=?¹,,²³¼¼½¬{[]}',
            "",
            """SELECT
 DISTINCT ?x WHERE {
?x a <http://example.org#Person> .
?x <http://example.org#hasAge> ?s_1 .
?s_1 a <http://example.org#Number[]> .
 }""",
            """SELECT
 DISTINCT ?x WHERE {
?x a <http://example.org#Class> .
?x <http://example.org#hasProperty> ?s_1 .
?s_1 a <http://example.org#Value%> .
 }""",
        ]:
            self.assertEqual(string, unsanitize(sanitize(string)))

    def test_transform_manchester_to_sparql(self):
        namespace = "http://example.org#"
        manchester = "Person and hasAge some Number[]"

        sparql_output = transform_manchester_to_sparql(manchester, namespace)
        self.assertIsInstance(sparql_output, str)  # Check if the result is a string
        self.assertIn(namespace, sparql_output)
        self.assertEqual(
            sparql_output,
            """SELECT
 DISTINCT ?x WHERE {
?x a <http://example.org#Person> .
?x <http://example.org#hasAge> ?s_1 .
?s_1 a <http://example.org#Number[]> .
 }""",
        )

    def test_transform_dl_to_sparql(self):
        namespace = "http://example.org#"
        dl_expression = "(Class ⊓ (∃ hasProperty.Value%))"

        sparql_output = transform_dl_to_sparql(dl_expression, namespace)
        self.assertIsInstance(sparql_output, str)  # Check if the result is a string
        self.assertIn(namespace, sparql_output)
        self.assertEqual(
            sparql_output,
            """SELECT
 DISTINCT ?x WHERE {
?x a <http://example.org#Class> .
?x <http://example.org#hasProperty> ?s_1 .
?s_1 a <http://example.org#Value%> .
 }""",
        )


if __name__ == "__main__":
    unittest.main()
