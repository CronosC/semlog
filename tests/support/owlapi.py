"""Reading an emitted ontology through the OWL API.

`tests/support/ontology.py` reads a build as text and answers what its axioms
say. This answers what the OWL API makes of the same file: which profile it is
in, what it carries as an ontology annotation, and what its signature holds
against what it declares. Those are questions about the structure a parser
builds, which a reader working from the lines of the file cannot see.

Every function here starts a JVM, so a test using one carries the `reasoner`
marker.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from semanticlog.core.enums import OWLProfile
from semanticlog.core.vocabulary import DEFAULT_IRI

# The entity types OWL 2 DL requires a declaration for. A named individual is
# not one of them, and neither sink declares one.
DECLARED_ENTITY_TYPES: frozenset[str] = frozenset(
    {"Class", "ObjectProperty", "DataProperty", "AnnotationProperty", "Datatype"}
)


def load(path: str | Path) -> Any:
    """Return the OWL API ontology a file holds.

    Args:
        path: Ontology file, in either syntax a sink writes.

    Returns:
        The `OWLOntology` the OWL API parsed, not the owlapy wrapper around it.
    """
    from owlapy.owl_ontology import SyncOntology

    from semanticlog.onto.reason import start_jvm

    start_jvm()
    return SyncOntology(str(path)).get_owlapi_ontology()


def profile_violations(path: str | Path, profile: OWLProfile) -> dict[str, list[str]]:
    """Return what keeps a file out of a profile.

    Args:
        path: Ontology file to check.
        profile: Profile to check it against.

    Returns:
        Violation kind to the text of each violation of that kind, so a
        failure names the sorts of problem rather than one example of each.
        Empty when the file is in the profile.
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

    An `AnnotationAssertion` whose subject happens to be the ontology IRI is
    not one of these, which is the difference this answers.

    Args:
        path: Ontology file to read.

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
    in the build namespace, so a built-in such as `rdfs:label` is not counted.

    Args:
        path: Ontology file to read.
        iri: Base IRI whose names are the ones this asks about.

    Returns:
        Entity type to the local names of that type that were never declared.
    """
    ontology = load(path)
    base = iri if iri.endswith("#") else iri + "#"
    found: dict[str, list[str]] = defaultdict(list)
    for entity in ontology.getSignature():
        kind = str(entity.getEntityType().getName())
        name = str(entity.getIRI().toString())
        if kind not in DECLARED_ENTITY_TYPES or not name.startswith(base):
            continue
        if not ontology.isDeclared(entity):
            found[kind].append(name.removeprefix(base))
    return {kind: sorted(names) for kind, names in found.items()}


def ontology_iris(path: str | Path) -> tuple[str | None, str | None]:
    """Return the ontology IRI and the version IRI, either of which may be absent.

    Args:
        path: Ontology file to read.

    Returns:
        The two IRIs, each None when the file states none.
    """
    ontology_id = load(path).getOntologyID()
    return (
        _optional(ontology_id.getOntologyIRI()),
        _optional(ontology_id.getVersionIRI()),
    )


def _optional(value: Any) -> str | None:
    """Return the contents of a Java `Optional`, or None when it is empty."""
    return str(value.get()) if value.isPresent() else None
