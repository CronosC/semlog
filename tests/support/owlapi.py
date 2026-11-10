"""Reading an emitted ontology through the OWL API.

What the OWL API makes of a written file: its profile, its ontology
annotations, and its undeclared signature. Every function starts a JVM, so a
test using one carries the `reasoner` marker.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from semanticlog.core.enums import OWLProfile
from semanticlog.core.vocabulary import DEFAULT_IRI, TBOX_IRI

# The entity types OWL 2 DL requires a declaration for. A named individual is
# not one of them, and neither sink declares one.
DECLARED_ENTITY_TYPES: frozenset[str] = frozenset(
    {"Class", "ObjectProperty", "DataProperty", "AnnotationProperty", "Datatype"}
)


def load(path: str | Path) -> Any:
    """Return the `OWLOntology` the OWL API parses from a file, not the owlapy wrapper."""
    from owlapy.owl_ontology import SyncOntology

    from semanticlog.onto.reason import start_jvm

    start_jvm()
    return SyncOntology(str(path)).get_owlapi_ontology()


def profile_violations(path: str | Path, profile: OWLProfile) -> dict[str, list[str]]:
    """Return what keeps a file out of a profile.

    Returns:
        Violation kind to the text of each violation of that kind. Empty when
        the file is in the profile.
    """
    ontology = load(path)
    from org.semanticweb.owlapi.profiles import OWL2DLProfile, OWL2ELProfile

    checker = OWL2ELProfile() if profile is OWLProfile.OWL2EL else OWL2DLProfile()
    found: dict[str, list[str]] = defaultdict(list)
    for violation in checker.checkOntology(ontology).getViolations():
        found[str(violation.getClass().getSimpleName())].append(str(violation))
    return dict(found)


def ontology_annotations(path: str | Path) -> dict[str, str]:
    """Return the annotations on the ontology itself.

    An `AnnotationAssertion` whose subject is the ontology IRI is not one of
    these.

    Returns:
        Local name of the annotation property to its text.
    """
    return {
        str(annotation.getProperty().getIRI().getShortForm()): _text(
            annotation.getValue()
        )
        for annotation in load(path).getAnnotations()
    }


def _text(value: Any) -> str:
    """Return the text of an annotation value, literal or not."""
    return (
        str(value.asLiteral().get().getLiteral()) if value.isLiteral() else str(value)
    )


def undeclared_signature(
    path: str | Path, iri: str = DEFAULT_IRI
) -> dict[str, list[str]]:
    """Return the entities a file uses without declaring.

    Only the entity types OWL 2 DL requires a declaration for, and only names
    under `iri` or the TBox namespace, so a built-in such as `rdfs:label` is not
    counted.

    Returns:
        Entity type to the local names of that type that were never declared.
    """
    ontology = load(path)
    base = iri if iri.endswith("/") else iri + "/"
    found: dict[str, list[str]] = defaultdict(list)
    for entity in ontology.getSignature():
        kind = str(entity.getEntityType().getName())
        name = str(entity.getIRI().toString())
        namespace = next((n for n in (TBOX_IRI, base) if name.startswith(n)), None)
        if kind not in DECLARED_ENTITY_TYPES or namespace is None:
            continue
        if not ontology.isDeclared(entity):
            found[kind].append(name.removeprefix(namespace))
    return {kind: sorted(names) for kind, names in found.items()}


def ontology_iris(path: str | Path) -> tuple[str | None, str | None]:
    """Return the ontology IRI and the version IRI, each None when the file states none."""
    ontology_id = load(path).getOntologyID()
    return (
        _optional(ontology_id.getOntologyIRI()),
        _optional(ontology_id.getVersionIRI()),
    )


def _optional(value: Any) -> str | None:
    """Return the contents of a Java `Optional`, or None when it is empty."""
    return str(value.get()) if value.isPresent() else None
