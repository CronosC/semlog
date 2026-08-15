"""Reading an emitted ontology back as one comparable value.

A build writes either OWL functional syntax or RDF/XML depending on the sink.
Both are read into the same `Emitted` object here, so a test asserts against
the ontology rather than against the syntax that carried it, and the two sinks
can be compared directly.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

CLASS = "class"
OBJECT_PROPERTY = "object"
DATA_PROPERTY = "data"

_NOT_AN_AXIOM = ("Prefix(", "Ontology(", "Annotation(", ")")

# A prefixed name, with the backslash escapes StreamingSink writes for the
# characters OWL functional syntax reserves.
_REF = r"(?:[^\s()\\]|\\.)+"

_DECLARATION = re.compile(rf"^Declaration\((\w+)\(:({_REF})\)\)$")
_CLASS_ASSERTION = re.compile(rf"^ClassAssertion\(:({_REF}) :({_REF})\)$")
_OBJECT_ASSERTION = re.compile(
    rf"^ObjectPropertyAssertion\(:({_REF}) :({_REF}) :({_REF})\)$"
)
_DATA_ASSERTION = re.compile(
    rf'^DataPropertyAssertion\(:({_REF}) :({_REF}) "(.*)"\^\^xsd:(\w+)\)$', re.DOTALL
)
_SUB_CLASS = re.compile(rf"^SubClassOf\(:({_REF}) (.+)\)$")
_SUB_PROPERTY = re.compile(rf"^Sub(?:Object|Data)PropertyOf\(:({_REF}) :({_REF})\)$")
_DOMAIN = re.compile(rf"^(?:Object|Data)PropertyDomain\(:({_REF}) (.+)\)$")
_RANGE = re.compile(rf"^(?:Object|Data)PropertyRange\(:({_REF}) (.+)\)$")
_INVERSE = re.compile(rf"^InverseObjectProperties\(:({_REF}) :({_REF})\)$")
_TRANSITIVE = re.compile(rf"^TransitiveObjectProperty\(:({_REF})\)$")
_ENTITY_ANNOTATION = re.compile(
    rf'^AnnotationAssertion\(rdfs:(label|comment) :({_REF}) "(.*)"\)$', re.DOTALL
)
# The sink writes the property as a full IRI. A file the OWL API has rewritten
# after materializing carries it as a prefixed name, with or without a prefix.
_HEADER_ANNOTATION = re.compile(
    r"^Annotation\("
    r"(?:<[^>]*[#/](?P<name>\w+)>|[A-Za-z]*:(?P<prefixed>\w+))"
    r' "(?P<value>.*)"\)$',
    re.DOTALL,
)

# The sink writes the property as a full IRI. A file the OWL API has rewritten
# after materializing carries it as a prefixed name instead.
_ONTOLOGY_ANNOTATION = re.compile(
    r"^AnnotationAssertion\("
    r"(?:<[^>]*[#/](?P<name>\w+)>|:(?P<prefixed>\w+))"
    r' <[^>]*> "(?P<value>.*)"\)$',
    re.DOTALL,
)

_DECLARATION_KIND = {
    "Class": CLASS,
    "ObjectProperty": OBJECT_PROPERTY,
    "DataProperty": DATA_PROPERTY,
}

_OWL = "http://www.w3.org/2002/07/owl#"
_RDFS = "http://www.w3.org/2000/01/rdf-schema#"


def _unescape(name: str) -> str:
    """Undo the backslash escapes of a functional-syntax prefixed name."""
    return re.sub(r"\\(.)", r"\1", name)


def _names(expression: str) -> set[str]:
    """Return the class names of a class expression, union or not."""
    return {_unescape(name) for name in re.findall(rf":({_REF})", expression)}


def _pair(one: str, other: str) -> tuple[str, str]:
    """Return an inverse pair in a fixed order, so the two syntaxes compare."""
    return (one, other) if one <= other else (other, one)


# owlapy writes every float as xsd:decimal whatever datatype the literal
# declares, and every other serialization is converted from that RDF/XML, so
# the in-memory sink cannot emit the xsd:double the streaming sink writes.
# Reading them as one keeps the sink comparison about everything else.
_EQUIVALENT_DATATYPES = {"decimal": "double"}


def _rdf_datatype(datatype: object) -> str:
    """Return the local name of a literal's datatype, as the two sinks compare."""
    name = str(datatype).rpartition("#")[2] if datatype is not None else "string"
    return _EQUIVALENT_DATATYPES.get(name, name)


@dataclass
class Emitted:
    """One emitted ontology, indexed by what a test wants to ask of it.

    Attributes:
        classes: Names declared as a class.
        object_properties: Names declared as an object property.
        data_properties: Names declared as a data property.
        types: Individual name to the classes it is asserted into.
        parents: Class name to its declared superclasses.
        property_parents: Property name to its declared superproperties.
        domains: Property name to the classes of its domain.
        ranges: Property name to the classes or datatypes of its range.
        relations: Object property assertions, as (property, subject, object).
        data_assertions: Data property assertions, as (property, subject).
        data_values: Data property assertions, with the literal text and
            the local name of its datatype.
        inverses: Axiomatic inverse pairs, each sorted, since RDF/XML records
            the axiom from both sides and functional syntax from one.
        transitive: Properties asserted transitive.
        annotations: Entity annotations, as (name, label or comment, text).
        ontology_annotations: The provenance record, keyed by annotation name.
        axiom_lines: Every axiom line, for the syntaxes that have lines.
    """

    classes: set[str] = field(default_factory=set)
    object_properties: set[str] = field(default_factory=set)
    data_properties: set[str] = field(default_factory=set)
    types: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    parents: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    property_parents: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    domains: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    ranges: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    relations: set[tuple[str, str, str]] = field(default_factory=set)
    data_assertions: set[tuple[str, str]] = field(default_factory=set)
    data_values: set[tuple[str, str, str, str]] = field(default_factory=set)
    inverses: set[tuple[str, str]] = field(default_factory=set)
    transitive: set[str] = field(default_factory=set)
    annotations: set[tuple[str, str, str]] = field(default_factory=set)
    ontology_annotations: dict[str, str] = field(default_factory=dict)
    axiom_lines: list[str] = field(default_factory=list)

    @property
    def properties(self) -> set[str]:
        """Every declared property, of either kind."""
        return self.object_properties | self.data_properties

    @property
    def declared(self) -> set[str]:
        """Every declared name."""
        return self.classes | self.properties

    @property
    def individuals(self) -> set[str]:
        """Every name asserted into a class."""
        return set(self.types)

    @property
    def relation_endpoints(self) -> set[str]:
        """Every name used on either side of an object property assertion."""
        return {name for _, s, o in self.relations for name in (s, o)}

    @property
    def duplicate_axioms(self) -> Mapping[str, int]:
        """Axiom lines written more than once, with how often."""
        return {line: n for line, n in Counter(self.axiom_lines).items() if n > 1}

    def counts(self) -> dict[str, int]:
        """Return the entity and axiom totals, as the shape table records them."""
        return {
            "classes": len(self.classes),
            "object_properties": len(self.object_properties),
            "data_properties": len(self.data_properties),
            "individuals": len(self.individuals),
            "axioms": len(self.axiom_lines),
        }

    def ancestors(self, name: str) -> set[str]:
        """Return every class above `name`, following SubClassOf transitively."""
        seen: set[str] = set()
        stack = list(self.parents.get(name, ()))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.parents.get(current, ()))
        return seen

    def lineage(self, individual: str) -> set[str]:
        """Return the classes of an individual and everything above them."""
        classes = self.types.get(individual, set())
        return (
            set(classes).union(*(self.ancestors(c) for c in classes))
            if classes
            else set()
        )


def read_functional(path: str | Path) -> Emitted:
    """Read an ontology in OWL 2 functional syntax, as StreamingSink writes it."""
    emitted = Emitted()
    _read_header_annotations(Path(path), emitted)
    for line in _axiom_lines(Path(path)):
        emitted.axiom_lines.append(line)
        _read_functional_axiom(line, emitted)
    return emitted


def _read_header_annotations(path: Path, emitted: Emitted) -> None:
    """Read the annotations inside the `Ontology(...)` header.

    They sit between the header line and the first axiom, so reading stops at
    the first line that is not one. A file the OWL API has rewritten carries
    them as annotation assertions instead, which the axiom reader picks up.
    """
    pending: list[str] = []
    started = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not started:
            started = line.startswith("Ontology(")
            continue
        if not pending and not line.startswith("Annotation("):
            return
        pending.append(line)
        annotation = "\n".join(pending)
        if _in_quoted_string(annotation):
            continue
        pending = []
        if m := _HEADER_ANNOTATION.match(annotation):
            name = m["name"] or m["prefixed"]
            emitted.ontology_annotations[name] = _unquote(m["value"])


def _axiom_lines(path: Path) -> Iterator[str]:
    """Yield the axioms of a functional-syntax file, dropping the frame.

    A literal holding a newline puts one axiom on several lines, so lines are
    joined until the quoted string that opened is closed again.
    """
    pending: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not pending:
            if not line or line.startswith(_NOT_AN_AXIOM):
                continue
            pending = [line]
        else:
            pending.append(line)
        axiom = "\n".join(pending)
        if not _in_quoted_string(axiom):
            pending = []
            yield axiom
    if pending:
        yield "\n".join(pending)


def _in_quoted_string(text: str) -> bool:
    """Report whether text ends inside an unterminated quoted string."""
    inside = False
    position = 0
    while position < len(text):
        if text[position] == "\\":
            position += 2
            continue
        if text[position] == '"':
            inside = not inside
        position += 1
    return inside


def _read_functional_axiom(line: str, emitted: Emitted) -> None:
    """Add one functional-syntax axiom to `emitted`."""
    if m := _DECLARATION.match(line):
        kind, name = m.groups()
        # A materialized file comes back from the OWL API, which also declares
        # named individuals and annotation properties. The sink writes three.
        if (declared := _DECLARATION_KIND.get(kind)) is not None:
            {
                CLASS: emitted.classes,
                OBJECT_PROPERTY: emitted.object_properties,
                DATA_PROPERTY: emitted.data_properties,
            }[declared].add(_unescape(name))
    elif m := _CLASS_ASSERTION.match(line):
        class_name, individual = m.groups()
        emitted.types[_unescape(individual)].add(_unescape(class_name))
    elif m := _OBJECT_ASSERTION.match(line):
        prop, subject, obj = (_unescape(g) for g in m.groups())
        emitted.relations.add((prop, subject, obj))
    elif m := _DATA_ASSERTION.match(line):
        prop, subject, value, datatype = m.groups()
        emitted.data_assertions.add((_unescape(prop), _unescape(subject)))
        emitted.data_values.add(
            (_unescape(prop), _unescape(subject), _unquote(value), datatype)
        )
    elif m := _SUB_CLASS.match(line):
        child, expression = m.groups()
        emitted.parents[_unescape(child)] |= _names(expression)
    elif m := _SUB_PROPERTY.match(line):
        child, parent = m.groups()
        emitted.property_parents[_unescape(child)].add(_unescape(parent))
    elif m := _DOMAIN.match(line):
        prop, expression = m.groups()
        emitted.domains[_unescape(prop)] |= _names(expression)
    elif m := _RANGE.match(line):
        prop, expression = m.groups()
        emitted.ranges[_unescape(prop)] |= _range_names(expression)
    elif m := _INVERSE.match(line):
        prop, inverse = m.groups()
        emitted.inverses.add(_pair(_unescape(prop), _unescape(inverse)))
    elif m := _TRANSITIVE.match(line):
        emitted.transitive.add(_unescape(m.group(1)))
    elif m := _ENTITY_ANNOTATION.match(line):
        kind, name, text = m.groups()
        emitted.annotations.add((_unescape(name), kind, _unquote(text)))
    elif m := _ONTOLOGY_ANNOTATION.match(line):
        name = m["name"] or m["prefixed"]
        emitted.ontology_annotations[name] = _unquote(m["value"])


def _range_names(expression: str) -> set[str]:
    """Return the range of a property, which may be a datatype rather than a class."""
    if expression.startswith("xsd:"):
        return {expression.removeprefix("xsd:")}
    return _names(expression)


def _unquote(text: str) -> str:
    """Undo the escaping of a quoted literal.

    One pass left to right, since a quoted string escapes only `"` and `\\` and
    a run of backslashes would otherwise be decoded twice.
    """
    return re.sub(r"\\(.)", r"\1", text)


def read_rdf(path: str | Path, iri: str) -> Emitted:
    """Read an ontology in RDF/XML, as InMemorySink writes it.

    Uses rdflib rather than the library that wrote the file, so this doubles as
    a check that the output is readable by something else.

    Args:
        path: File to read.
        iri: Base IRI, whose local names are the ones a test asks about.

    Returns:
        The same shape `read_functional` returns, minus the axiom lines, which
        RDF/XML does not have.
    """
    from rdflib import OWL, RDF, RDFS, BNode, Graph, Literal, URIRef

    graph = Graph()
    graph.parse(str(path))

    base = iri if iri.endswith("#") else iri + "#"
    ontology_iri = URIRef(iri.rstrip("#"))

    def local(term: object) -> str | None:
        """Return the local name of a term of this ontology, or None."""
        if not isinstance(term, URIRef):
            return None
        text = str(term)
        if text.startswith(base):
            return text[len(base) :]
        if text.startswith(iri) and text[len(iri) :].startswith("#"):
            return text[len(iri) + 1 :]
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
            emitted.inverses.add(_pair(name, inverse))

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
        if prop in emitted.object_properties and (name := local(obj)) is not None:
            emitted.relations.add((prop, subject_name, name))
        elif prop in emitted.data_properties and isinstance(obj, Literal):
            emitted.data_assertions.add((prop, subject_name))
            emitted.data_values.add(
                (prop, subject_name, str(obj), _rdf_datatype(obj.datatype))
            )

    for kind, predicate in (("label", RDFS.label), ("comment", RDFS.comment)):
        for subject, _, obj in graph.triples((None, predicate, None)):
            if (name := local(subject)) is not None:
                emitted.annotations.add((name, kind, str(obj)))

    for _, predicate, obj in graph.triples((ontology_iri, None, None)):
        if isinstance(obj, Literal):
            emitted.ontology_annotations[str(predicate).rpartition("#")[2]] = str(obj)

    return emitted
