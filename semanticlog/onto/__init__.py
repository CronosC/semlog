"""Turning stored event data into an OWL 2 ontology.

`OntoBuildSettings` says what to emit, `OCEDOntoBuilder` emits it through a
sink, and the result is a `BuildResult` that can be written out or reasoned
over.
"""

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

__all__ = [
    "DEFAULT_IRI",
    "DataRepresentation",
    "EventOrderRepresentation",
    "InverseRepresentation",
    "OCEDOntoBuilder",
    "OCELExtension",
    "OntoBuildSettings",
    "OWLProfile",
    "Reasoner",
    "BuildResult",
    "construct_tbox",
]
