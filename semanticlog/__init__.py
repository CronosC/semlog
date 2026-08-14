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

Each name is resolved on first use rather than on import. pm4py and the Java
OWL API are between them most of the cost of loading this package, and a
caller that only reaches for the data layer should not pay for either.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from semanticlog.core.enums import DataRepresentation as DataRepresentation
    from semanticlog.core.enums import (
        EventOrderRepresentation as EventOrderRepresentation,
    )
    from semanticlog.core.enums import InverseRepresentation as InverseRepresentation
    from semanticlog.core.enums import OCELExtension as OCELExtension
    from semanticlog.core.enums import OWLProfile as OWLProfile
    from semanticlog.core.vocabulary import DEFAULT_IRI as DEFAULT_IRI
    from semanticlog.data.cleaning import clean_oced_data as clean_oced_data
    from semanticlog.data.loading.classical_loader import (
        ClassicalLoader as ClassicalLoader,
    )
    from semanticlog.data.loading.pm4py_loader import PM4PyLoader as PM4PyLoader
    from semanticlog.data.oced import OCEDStore as OCEDStore
    from semanticlog.enrich.declare import DeclareResult as DeclareResult
    from semanticlog.onto.builder import OCEDOntoBuilder as OCEDOntoBuilder
    from semanticlog.onto.reason import Reasoner as Reasoner
    from semanticlog.onto.result import BuildResult as BuildResult
    from semanticlog.onto.settings import OntoBuildSettings as OntoBuildSettings

try:
    __version__ = version("semlog")
except PackageNotFoundError:  # running from a checkout that was never installed
    __version__ = "unknown"

_EXPORTS: dict[str, str] = {
    "BuildResult": "semanticlog.onto.result",
    "ClassicalLoader": "semanticlog.data.loading.classical_loader",
    "DEFAULT_IRI": "semanticlog.core.vocabulary",
    "DataRepresentation": "semanticlog.core.enums",
    "DeclareResult": "semanticlog.enrich.declare",
    "EventOrderRepresentation": "semanticlog.core.enums",
    "InverseRepresentation": "semanticlog.core.enums",
    "OCEDOntoBuilder": "semanticlog.onto.builder",
    "OCEDStore": "semanticlog.data.oced",
    "OCELExtension": "semanticlog.core.enums",
    "OWLProfile": "semanticlog.core.enums",
    "OntoBuildSettings": "semanticlog.onto.settings",
    "PM4PyLoader": "semanticlog.data.loading.pm4py_loader",
    "Reasoner": "semanticlog.onto.reason",
    "clean_oced_data": "semanticlog.data.cleaning",
}

__all__ = [*sorted(_EXPORTS), "__version__"]


def __getattr__(name: str) -> Any:
    """Import the module behind a re-exported name the first time it is used."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)
