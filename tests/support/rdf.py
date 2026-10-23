"""Reading RDF/XML, as `InMemorySink` writes it, into an `Emitted`."""

from __future__ import annotations

from pathlib import Path

from semanticlog.core.enums import IndividualKind
from semanticlog.core.vocabulary import TBOX_IRI
from tests.support.emitted import Emitted, inverse_pair


def _rdf_datatype(datatype: object) -> str:
    """Return the local name of a literal's datatype."""
    return str(datatype).rpartition("#")[2] if datatype is not None else "string"


def read_rdf(path: str | Path, iri: str, tbox_iri: str = TBOX_IRI) -> Emitted:
    """Read an ontology in RDF/XML, as InMemorySink writes it.

    Reads with rdflib. An object property assertion is recorded whether or not
    its property is declared.

    Args:
        path: File to read.
        iri: Log base IRI. Names under it and under `tbox_iri` are read as
            their local names.
        tbox_iri: Namespace of the schema names.

    Returns:
        The same `Emitted` that `read_functional` returns, without axiom
        lines. `counts()["axioms"]` is 0 for RDF.
    """
    from rdflib import OWL, RDF, RDFS, BNode, Graph, Literal, URIRef

    graph = Graph()
    graph.parse(str(path))

    base = iri if iri.endswith("/") else iri + "/"
    ontology_iri = URIRef(base.removesuffix("/"))
    namespaces = (tbox_iri, *(f"{base}{kind.path}" for kind in IndividualKind), base)

    def local(term: object) -> str | None:
        """Return the local name of a log, TBox or individual term, or None."""
        if not isinstance(term, URIRef):
            return None
        text = str(term)
        for namespace in namespaces:
            if text.startswith(namespace):
                return text[len(namespace) :]
        return None

    def expression(term: object) -> set[str]:
        """Resolve a class expression, following owl:unionOf into its members."""
        if isinstance(term, BNode):
            found: set[str] = set()
            for collection in graph.objects(term, OWL.unionOf):
                found |= {
                    name
                    for member in graph.items(collection)
                    if (name := local(member)) is not None
                }
            return found
        if isinstance(term, URIRef):
            name = local(term)
            if name is not None:
                return {name}
            text = str(term)
            if text.startswith("http://www.w3.org/2001/XMLSchema#"):
                return {text.rpartition("#")[2]}
        return set()

    emitted = Emitted()

    declared_as = {
        OWL.Class: emitted.classes,
        OWL.ObjectProperty: emitted.object_properties,
        OWL.DatatypeProperty: emitted.data_properties,
    }
    for kind, target in declared_as.items():
        for subject in graph.subjects(RDF.type, kind):
            if (name := local(subject)) is not None:
                target.add(name)

    for subject, _, obj in graph.triples((None, RDF.type, None)):
        name, class_name = local(subject), local(obj)
        if name is not None and class_name is not None:
            emitted.types[name].add(class_name)

    for subject, _, obj in graph.triples((None, RDFS.subClassOf, None)):
        if (name := local(subject)) is not None:
            emitted.parents[name] |= expression(obj)

    for subject, _, obj in graph.triples((None, RDFS.subPropertyOf, None)):
        name, parent = local(subject), local(obj)
        if name is not None and parent is not None:
            emitted.property_parents[name].add(parent)

    for predicate, target in (
        (RDFS.domain, emitted.domains),
        (RDFS.range, emitted.ranges),
    ):
        for subject, _, obj in graph.triples((None, predicate, None)):
            if (name := local(subject)) is not None:
                target[name] |= expression(obj)

    for subject, _, obj in graph.triples((None, OWL.inverseOf, None)):
        name, inverse = local(subject), local(obj)
        if name is not None and inverse is not None:
            emitted.inverses.add(inverse_pair(name, inverse))

    for subject in graph.subjects(RDF.type, OWL.TransitiveProperty):
        if (name := local(subject)) is not None:
            emitted.transitive.add(name)

    for kind, target in (
        (OWL.FunctionalProperty, emitted.functional),
        (OWL.InverseFunctionalProperty, emitted.inverse_functional),
    ):
        for subject in graph.subjects(RDF.type, kind):
            if (name := local(subject)) is not None:
                target.add(name)

    for subject in graph.subjects(OWL.equivalentClass, None):
        if (name := local(subject)) is not None:
            emitted.defined.add(name)

    for subject, _, obj in graph.triples((None, OWL.disjointWith, None)):
        emitted.disjoint.add(frozenset({local(subject), local(obj)} - {None}))
    for node in graph.subjects(RDF.type, OWL.AllDisjointClasses):
        for members in graph.objects(node, OWL.members):
            emitted.disjoint.add(
                frozenset(n for m in graph.items(members) if (n := local(m)))
            )

    annotation_properties = {
        name
        for prop in graph.subjects(RDF.type, OWL.AnnotationProperty)
        if (name := local(prop)) is not None
    }
    for subject, predicate, obj in graph:
        prop = local(predicate)
        if prop is None or prop in annotation_properties:
            continue
        subject_name = local(subject)
        if subject_name is None:
            continue
        if isinstance(obj, Literal):
            if prop in emitted.data_properties:
                emitted.data_assertions.add((prop, subject_name))
                emitted.data_values.add(
                    (prop, subject_name, str(obj), _rdf_datatype(obj.datatype))
                )
        elif (name := local(obj)) is not None:
            emitted.relations.add((prop, subject_name, name))

    for kind, predicate in (("label", RDFS.label), ("comment", RDFS.comment)):
        for subject, _, obj in graph.triples((None, predicate, None)):
            if (name := local(subject)) is not None:
                emitted.annotations.add((name, kind, str(obj)))

    for prop in graph.subjects(RDF.type, OWL.AnnotationProperty):
        kind = local(prop)
        if kind is None:
            continue
        for subject, _, obj in graph.triples((None, prop, None)):
            name = local(subject)
            if name is None or subject == ontology_iri:
                continue
            value = str(obj) if isinstance(obj, Literal) else local(obj)
            if value is not None:
                emitted.annotations.add((name, kind, value))

    for _, predicate, obj in graph.triples((ontology_iri, None, None)):
        if isinstance(obj, Literal):
            emitted.ontology_annotations[str(predicate).rpartition("#")[2]] = str(obj)

    return emitted
