import functools

from owlapy import manchester_to_owl_expression
from owlapy.class_expression import OWLClassExpression


@functools.cache
def parse_manchester(iri: str, manchester: str) -> OWLClassExpression:
    """Parse a Manchester Syntax class expression string into an OWLClassExpression.

    Results are cached: repeated calls with the same ``(iri, manchester)`` pair
    return the identical object without re-parsing.

    Args:
        iri: Base IRI prefix used to resolve unqualified class/property names.
        manchester: Manchester Syntax expression, e.g. ``"Event or Object"``.

    Returns:
        Parsed OWLClassExpression.
    """
    return manchester_to_owl_expression(manchester, iri)
