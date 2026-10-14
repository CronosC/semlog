"""Turning stored event data into an OWL 2 ontology.

`OntoBuildSettings` says what to emit, `OCEDOntoBuilder` emits it through a
sink, and the result is a `BuildResult` that can be written out or reasoned
over. `Reasoner` starts the Java OWL API and is imported on first use.
"""

from importlib import import_module
from typing import Any

from semanticlog.core.enums import (
    DataRepresentation,
    EventOrderRepresentation,
    InverseRepresentation,
    OCELExtension,
    OWLProfile,
)
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.result import BuildResult
from semanticlog.onto.settings import OntoBuildSettings

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
]


def __getattr__(name: str) -> Any:
    if name != "Reasoner":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module("semanticlog.onto.reason"), name)


def __dir__() -> list[str]:
    return sorted(__all__)
