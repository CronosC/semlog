"""Object-centric event data as OWL 2 ontologies.

A build runs in four steps: read the log into an `OCEDStore`, clean the names
it will emit, describe the ontology with `OntoBuildSettings`, and let
`OCEDOntoBuilder` produce it. The names used most often are re-exported here,
and the packages below hold the rest.

    from semanticlog import OCEDOntoBuilder, OntoBuildSettings, PM4PyLoader
    from semanticlog import clean_oced_data, DEFAULT_IRI

    data = clean_oced_data(PM4PyLoader("log.jsonocel", {}).get_oced())
    result = OCEDOntoBuilder(DEFAULT_IRI, OntoBuildSettings()).build_ontology(data)
    result.save("log.owl")
"""

from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.loading.classical_loader import ClassicalLoader
from semanticlog.data.loading.pm4py_loader import PM4PyLoader
from semanticlog.data.oced import OCEDStore
from semanticlog.enrich.declare import DeclareResult
from semanticlog.onto.builder import OCEDOntoBuilder, construct_tbox
from semanticlog.onto.reason import Reasoner
from semanticlog.onto.result import BuildResult
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    DataRepresentation,
    EventOrderRepresentation,
    InverseRepresentation,
    OCELExtension,
    OntoBuildSettings,
    OWLProfile,
)

__version__ = "0.4.0"

__all__ = [
    "DEFAULT_IRI",
    "ClassicalLoader",
    "DataRepresentation",
    "DeclareResult",
    "EventOrderRepresentation",
    "InverseRepresentation",
    "OCEDOntoBuilder",
    "OCEDStore",
    "OCELExtension",
    "OntoBuildSettings",
    "OWLProfile",
    "PM4PyLoader",
    "Reasoner",
    "BuildResult",
    "__version__",
    "clean_oced_data",
    "construct_tbox",
]
