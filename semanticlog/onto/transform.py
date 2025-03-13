from owlapy import dl_to_owl_expression, manchester_to_owl_expression, owl_expression_to_sparql, owl_expression_to_manchester
from owlapy.class_expression import OWLClassExpression
import enum

class Format(enum.Enum):
    OWL = "OWL"
    DL = "DL"
    MANCHESTER = "Manchester"
    
### Parsing string expressions ####

def parse_manchester(iri: str, manchester : str) -> OWLClassExpression:
    return manchester_to_owl_expression(manchester, iri)

def parse_dl(iri: str, dl : str) -> OWLClassExpression:
    return dl_to_owl_expression(dl, iri)
    
#### To sparql ####

def theory_to_sparql_query(theory: str, format: Format, iri:str = "https://uni-bamberg.de/KIGA/OCED.owl#"):
    match format:
        case Format.OWL:
            return transform_owl_to_sparql(iri, theory)
        case Format.DL:
            return transform_dl_to_sparql(iri, theory)
        case Format.MANCHESTER:
            return transform_manchester_to_sparql(iri, theory)
        case _:
            raise ValueError("Not a valid theory format!")
        
def transform_dl_to_manchester(dl:str, iri:str = "https://uni-bamberg.de/KIGA/OCED.owl#") -> str:
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

def transform_manchester_to_sparql(manchester: str, iri:str = "https://uni-bamberg.de/KIGA/OCED.owl#") -> str:
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
    sparql = unsanitize(sparql)
    return sparql

def transform_dl_to_sparql(dl: str, iri:str = "https://uni-bamberg.de/KIGA/OCED.owl#") -> str:
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
    sparql = unsanitize(sparql)
    return sparql

def transform_owl_to_sparql(owl: OWLClassExpression, iri:str = "https://uni-bamberg.de/KIGA/OCED.owl#") -> str:
    raise NotImplementedError("Has not been necessary so far!")

#### Helper ####

# might not be necessary anymore, declare constraint representation in the log has changed
def sanitize(string: str) -> str:
    """
    Replaces special characters in a string with placeholders to ensure compatibility.

    Args:
        string (str): The input string to sanitize.

    Returns:
        str: The sanitized string with placeholders for special characters.
    """
    invalid_special_characters = {
        "%": "PERCENTAGESIGN",
        "[": "SQUAREBRACKETOPENING",
        "]": "SQUAREBRACKETCLOSING",
        ",": "COMMATA"
    }
    for key in invalid_special_characters:
        string = string.replace(key, invalid_special_characters[key])
    return string


def unsanitize(string: str) -> str:
    """
    Replaces placeholders in a string with the original special characters.

    Args:
        string (str): The input string to unsanitize.

    Returns:
        str: The unsanitized string with original special characters restored.
    """
    invalid_special_characters = {
        "%": "PERCENTAGESIGN",
        "[": "SQUAREBRACKETOPENING",
        "]": "SQUAREBRACKETCLOSING",
        ",": "COMMATA"
    }
    for key in invalid_special_characters:
        string = string.replace(invalid_special_characters[key], key)
    return string