"""Object-centric event data as OWL 2 ontologies.

A build runs in three steps: load and clean the log, choose the case
concepts, and build. The names used most often are re-exported here, and the
packages below hold the rest.

    from semanticlog import BuildInputs, OntoBuildSettings
    from semanticlog import build_log, choose_case_concepts, load_and_clean

    data = load_and_clean("log.jsonocel")
    case_concepts, _ = choose_case_concepts(data, None)
    build_log(data, OntoBuildSettings(), BuildInputs(case_concepts), "log.ofn")

Each name is imported on first use. A name from the data layer does not load
the Java OWL API.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from semanticlog.core.enums import (
        AttributeRepresentation as AttributeRepresentation,
    )
    from semanticlog.core.enums import (
        EventOrderRepresentation as EventOrderRepresentation,
    )
    from semanticlog.core.enums import InverseRepresentation as InverseRepresentation
    from semanticlog.core.enums import OCELExtension as OCELExtension
    from semanticlog.core.enums import OWLProfile as OWLProfile
    from semanticlog.core.vocabulary import DEFAULT_IRI as DEFAULT_IRI
    from semanticlog.data.cleaning import clean_oced_data as clean_oced_data
    from semanticlog.data.declare import DeclareResult as DeclareResult
    from semanticlog.data.flatten import (
        choose_case_concepts as choose_case_concepts,
    )
    from semanticlog.data.loading import load_and_clean as load_and_clean
    from semanticlog.data.loading.classical_loader import (
        ClassicalLoader as ClassicalLoader,
    )
    from semanticlog.data.loading.ocel_loader import OCELLoader as OCELLoader
    from semanticlog.data.loading.pm4py_loader import PM4PyLoader as PM4PyLoader
    from semanticlog.data.oced import OCEDStore as OCEDStore
    from semanticlog.data.readable import ReadableStore as ReadableStore
    from semanticlog.data.tables import EntityKind as EntityKind
    from semanticlog.data.tables import RelationKind as RelationKind
    from semanticlog.onto.builder import OCEDOntoBuilder as OCEDOntoBuilder
    from semanticlog.onto.inputs import BuildInputs as BuildInputs
    from semanticlog.onto.pipeline import build_log as build_log
    from semanticlog.onto.reason import Reasoner as Reasoner
    from semanticlog.onto.result import BuildResult as BuildResult
    from semanticlog.onto.settings import OntoBuildSettings as OntoBuildSettings

try:
    __version__ = version("semlog")
except PackageNotFoundError:
    __version__ = "unknown"

_EXPORTS: dict[str, str] = {
    "BuildInputs": "semanticlog.onto.inputs",
    "BuildResult": "semanticlog.onto.result",
    "ClassicalLoader": "semanticlog.data.loading.classical_loader",
    "DEFAULT_IRI": "semanticlog.core.vocabulary",
    "AttributeRepresentation": "semanticlog.core.enums",
    "DeclareResult": "semanticlog.data.declare",
    "EntityKind": "semanticlog.data.tables",
    "EventOrderRepresentation": "semanticlog.core.enums",
    "InverseRepresentation": "semanticlog.core.enums",
    "OCEDOntoBuilder": "semanticlog.onto.builder",
    "OCEDStore": "semanticlog.data.oced",
    "OCELExtension": "semanticlog.core.enums",
    "OCELLoader": "semanticlog.data.loading.ocel_loader",
    "OWLProfile": "semanticlog.core.enums",
    "OntoBuildSettings": "semanticlog.onto.settings",
    "PM4PyLoader": "semanticlog.data.loading.pm4py_loader",
    "ReadableStore": "semanticlog.data.readable",
    "RelationKind": "semanticlog.data.tables",
    "Reasoner": "semanticlog.onto.reason",
    "build_log": "semanticlog.onto.pipeline",
    "choose_case_concepts": "semanticlog.data.flatten",
    "clean_oced_data": "semanticlog.data.cleaning",
    "load_and_clean": "semanticlog.data.loading",
}

__all__ = [*sorted(_EXPORTS), "__version__"]


def __getattr__(name: str) -> Any:
    """Import the module behind a re-exported name on first use."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)
