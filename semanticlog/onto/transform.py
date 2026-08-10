import functools

from owlapy import manchester_to_owl_expression
from owlapy.class_expression import OWLClassExpression

#: Cache bound. Every class and property name a build touches lands here, and the
#: cache is module-global, so an unbounded one would keep every name from every
#: build alive for the life of the process. A few thousand entries covers the
#: schema of any single build many times over.
_CACHE_SIZE = 8192


@functools.lru_cache(maxsize=_CACHE_SIZE)
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
