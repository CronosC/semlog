"""Object-centric event data as OWL 2 ontologies.

A build runs in three steps: load and clean the log, choose the case
concepts, and build. The names used most often are re-exported here, and the
packages below hold the rest.

    from semlog import BuildInputs, OntoBuildSettings
    from semlog import build_log, choose_case_concepts, load_and_clean

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
    from semlog.core.enums import (
        AttributeRepresentation as AttributeRepresentation,
    )
    from semlog.core.enums import (
        EventOrderRepresentation as EventOrderRepresentation,
    )
    from semlog.core.enums import InverseRepresentation as InverseRepresentation
    from semlog.core.enums import OCELExtension as OCELExtension
    from semlog.core.enums import OWLProfile as OWLProfile
    from semlog.core.vocabulary import DEFAULT_IRI as DEFAULT_IRI
    from semlog.data.cleaning import clean_oced_data as clean_oced_data
    from semlog.data.declare import DeclareResult as DeclareResult
    from semlog.data.flatten import (
        choose_case_concepts as choose_case_concepts,
    )
    from semlog.data.loading import load_and_clean as load_and_clean
    from semlog.data.loading.classical_loader import (
        ClassicalLoader as ClassicalLoader,
    )
    from semlog.data.loading.ocel_loader import OCELLoader as OCELLoader
    from semlog.data.loading.pm4py_loader import PM4PyLoader as PM4PyLoader
    from semlog.data.oced import OCEDStore as OCEDStore
    from semlog.data.readable import ReadableStore as ReadableStore
    from semlog.data.tables import EntityKind as EntityKind
    from semlog.data.tables import RelationKind as RelationKind
    from semlog.onto.builder import OCEDOntoBuilder as OCEDOntoBuilder
    from semlog.onto.inputs import BuildInputs as BuildInputs
    from semlog.onto.pipeline import build_log as build_log
    from semlog.onto.reason import Reasoner as Reasoner
    from semlog.onto.result import BuildResult as BuildResult
    from semlog.onto.settings import OntoBuildSettings as OntoBuildSettings

try:
    __version__ = version("semlog")
except PackageNotFoundError:
    __version__ = "unknown"

_EXPORTS: dict[str, str] = {
    "BuildInputs": "semlog.onto.inputs",
    "BuildResult": "semlog.onto.result",
    "ClassicalLoader": "semlog.data.loading.classical_loader",
    "DEFAULT_IRI": "semlog.core.vocabulary",
    "AttributeRepresentation": "semlog.core.enums",
    "DeclareResult": "semlog.data.declare",
    "EntityKind": "semlog.data.tables",
    "EventOrderRepresentation": "semlog.core.enums",
    "InverseRepresentation": "semlog.core.enums",
    "OCEDOntoBuilder": "semlog.onto.builder",
    "OCEDStore": "semlog.data.oced",
    "OCELExtension": "semlog.core.enums",
    "OCELLoader": "semlog.data.loading.ocel_loader",
    "OWLProfile": "semlog.core.enums",
    "OntoBuildSettings": "semlog.onto.settings",
    "PM4PyLoader": "semlog.data.loading.pm4py_loader",
    "ReadableStore": "semlog.data.readable",
    "RelationKind": "semlog.data.tables",
    "Reasoner": "semlog.onto.reason",
    "build_log": "semlog.onto.pipeline",
    "choose_case_concepts": "semlog.data.flatten",
    "clean_oced_data": "semlog.data.cleaning",
    "load_and_clean": "semlog.data.loading",
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
