import enum
import functools

from owlapy import (
    dl_to_owl_expression,
    manchester_to_owl_expression,
    owl_expression_to_manchester,
    owl_expression_to_sparql,
)
from owlapy.class_expression import OWLClassExpression

from semanticlog.onto.settings import DEFAULT_IRI


class Format(enum.Enum):
    OWL = "OWL"
    DL = "DL"
    MANCHESTER = "Manchester"


# Parsing string expressions


@functools.lru_cache(maxsize=None)
def parse_manchester(iri: str, manchester: str) -> OWLClassExpression:
    return manchester_to_owl_expression(manchester, iri)


def parse_dl(iri: str, dl: str) -> OWLClassExpression:
    return dl_to_owl_expression(dl, iri)


# To sparql


def theory_to_sparql_query(theory: str, format: Format, iri: str = DEFAULT_IRI):
    match format:
        case Format.OWL:
            return transform_owl_to_sparql(iri, theory)
        case Format.DL:
            return transform_dl_to_sparql(iri, theory)
        case Format.MANCHESTER:
            return transform_manchester_to_sparql(iri, theory)
        case _:
            raise ValueError("Not a valid theory format!")


def transform_dl_to_manchester(dl: str, iri: str = DEFAULT_IRI) -> str:
    """
    Transforms a DL syntax expression into a Manchester syntax expression.

    Args:
        iri (str): The iri to use for resolving terms in the Manchester syntax.
        dl (str): The DL syntax expression.

    Returns:
        str: The equivalent Manchester syntax expression.
    """
    dl = sanitize(dl)
    owl = dl_to_owl_expression(dl, iri)
    manchester = owl_expression_to_manchester(owl)
    manchester = unsanitize(manchester)
    return manchester


def transform_manchester_to_sparql(manchester: str, iri: str = DEFAULT_IRI) -> str:
    """
    Transforms a Manchester syntax expression into a SPARQL query.

    Args:
        iri (str): The iri to use for resolving terms in the Manchester syntax.
        manchester (str): The Manchester syntax expression.

    Returns:
        str: The equivalent SPARQL query.
    """

    manchester = sanitize(manchester)
    owl = manchester_to_owl_expression(manchester, iri)
    sparql = owl_expression_to_sparql(expression=owl)
    sparql = "\n".join(line.rstrip() for line in sparql.splitlines())
    sparql = unsanitize(sparql)
    return sparql


def transform_dl_to_sparql(dl: str, iri: str = DEFAULT_IRI) -> str:
    """
    Transforms a Description Logic (DL) expression into a SPARQL query.

    Args:
        iri (str): The iri to use for resolving terms in the DL expression.
        dl (str): The Description Logic expression.

    Returns:
        str: The equivalent SPARQL query.
    """
    dl = sanitize(dl)
    owl = dl_to_owl_expression(dl, iri)
    sparql = owl_expression_to_sparql(expression=owl)
    sparql = "\n".join(line.rstrip() for line in sparql.splitlines())
    sparql = unsanitize(sparql)
    return sparql


def transform_owl_to_sparql(owl: OWLClassExpression, iri: str = DEFAULT_IRI) -> str:
    raise NotImplementedError("Has not been necessary so far!")


# Helper

_SANITIZE_MAP = {
    "%": "PERCENTAGESIGN",
    "[": "SQUAREBRACKETOPENING",
    "]": "SQUAREBRACKETCLOSING",
    ",": "COMMATA",
}


def sanitize(string: str) -> str:
    """
    Replaces special characters in a string with placeholders to ensure compatibility.

    Args:
        string (str): The input string to sanitize.

    Returns:
        str: The sanitized string with placeholders for special characters.
    """
    for key, val in _SANITIZE_MAP.items():
        string = string.replace(key, val)
    return string


def unsanitize(string: str) -> str:
    """
    Replaces placeholders in a string with the original special characters.

    Args:
        string (str): The input string to unsanitize.

    Returns:
        str: The unsanitized string with original special characters restored.
    """
    for key, val in _SANITIZE_MAP.items():
        string = string.replace(val, key)
    return string
