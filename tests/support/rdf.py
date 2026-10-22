"""Reading RDF/XML, as `InMemorySink` writes it, into an `Emitted`."""

from __future__ import annotations

from pathlib import Path

from semanticlog.core.vocabulary import TBOX_IRI
from tests.support.emitted import Emitted, inverse_pair

# owlapy writes every float as xsd:decimal whatever the literal declares. The
# two are read as one.
_EQUIVALENT_DATATYPES = {"decimal": "double"}


def _rdf_datatype(datatype: object) -> str:
    """Return the local name of a literal's datatype, as the two sinks compare."""
    name = str(datatype).rpartition("#")[2] if datatype is not None else "string"
    return _EQUIVALENT_DATATYPES.get(name, name)


def read_rdf(path: str | Path, iri: str) -> Emitted:
    """Read an ontology in RDF/XML, as InMemorySink writes it.

    Reads with rdflib. An object property assertion is recorded whether or not
    its property is declared.

    Args:
        path: File to read.
        iri: Log base IRI. Names under it and under the TBox IRI are read as
            their local names.

    Returns:
        The same `Emitted` that `read_functional` returns, without axiom
        lines. `counts()["axioms"]` is 0 for RDF.
    """
    from rdflib import OWL, RDF, RDFS, BNode, Graph, Literal, URIRef

    graph = Graph()
    graph.parse(str(path))

    base = iri if iri.endswith("/") else iri + "/"
    ontology_iri = URIRef(base.removesuffix("/"))

    def local(term: object) -> str | None:
        """Return the local name of a log or TBox term, or None."""
        if not isinstance(term, URIRef):
            return None
        text = str(term)
        for namespace in (TBOX_IRI, base):
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

    for subject, predicate, obj in graph:
        prop = local(predicate)
        if prop is None:
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

    for _, predicate, obj in graph.triples((ontology_iri, None, None)):
        if isinstance(obj, Literal):
            emitted.ontology_annotations[str(predicate).rpartition("#")[2]] = str(obj)

    return emitted
