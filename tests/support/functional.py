"""Reading OWL 2 functional syntax, as `StreamingSink` writes it, into an `Emitted`."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from semanticlog.onto.record import _record_patterns, _unescape_functional
from tests.support.emitted import Emitted, inverse_pair

_NOT_AN_AXIOM = ("Prefix(", "Ontology(", "Annotation(", ")")

# A prefixed name, with the backslash escapes StreamingSink writes for the
# characters OWL functional syntax reserves.
_REF = r"(?:[^\s()\\]|\\.)+"

# A name is written under the log prefix `:` or the TBox prefix `oced:`. Both
# are read as their local name.
_P = r"(?:oced)?:"

_DECLARATION = re.compile(rf"^Declaration\((\w+)\({_P}({_REF})\)\)$")
_CLASS_ASSERTION = re.compile(rf"^ClassAssertion\({_P}({_REF}) {_P}({_REF})\)$")
_OBJECT_ASSERTION = re.compile(
    rf"^ObjectPropertyAssertion\({_P}({_REF}) {_P}({_REF}) {_P}({_REF})\)$"
)
_DATA_ASSERTION = re.compile(
    rf'^DataPropertyAssertion\({_P}({_REF}) {_P}({_REF}) "(.*)"\^\^xsd:(\w+)\)$',
    re.DOTALL,
)
_SUB_CLASS = re.compile(rf"^SubClassOf\({_P}({_REF}) (.+)\)$")
_SUB_PROPERTY = re.compile(
    rf"^Sub(?:Object|Data)PropertyOf\({_P}({_REF}) {_P}({_REF})\)$"
)
_DOMAIN = re.compile(rf"^(?:Object|Data)PropertyDomain\({_P}({_REF}) (.+)\)$")
_RANGE = re.compile(rf"^(?:Object|Data)PropertyRange\({_P}({_REF}) (.+)\)$")
_INVERSE = re.compile(rf"^InverseObjectProperties\({_P}({_REF}) {_P}({_REF})\)$")
_TRANSITIVE = re.compile(rf"^TransitiveObjectProperty\({_P}({_REF})\)$")
_ENTITY_ANNOTATION = re.compile(
    rf'^AnnotationAssertion\(rdfs:(label|comment) {_P}({_REF}) "(.*)"\)$', re.DOTALL
)

# The one ontology annotation whose property the sink does not declare.
_VERSION_INFO = "versionInfo"


def read_functional(path: str | Path) -> Emitted:
    """Read an ontology in OWL 2 functional syntax, as StreamingSink writes it."""
    text = Path(path).read_text(encoding="utf-8")
    emitted = Emitted()
    annotation_properties = {_VERSION_INFO}
    for line in _axiom_lines(text):
        emitted.axiom_lines.append(line)
        if m := _DECLARATION.match(line):
            kind, name = m.groups()
            if kind == "AnnotationProperty":
                annotation_properties.add(_unescape(name))
        _read_axiom(line, emitted)
    emitted.ontology_annotations = _ontology_annotations(text, annotation_properties)
    return emitted


def _ontology_annotations(text: str, names: set[str]) -> dict[str, str]:
    """Return the header annotations among `names`, read as `semanticlog.onto.record` reads one."""
    found: dict[str, str] = {}
    for name in names:
        pattern, _ = _record_patterns(name)[0]
        if m := pattern.search(text):
            found[name] = _unescape_functional(m.group(1))
    return found


def _axiom_lines(text: str) -> Iterator[str]:
    """Yield the axioms of a functional-syntax file, dropping the frame.

    A literal holding a newline puts one axiom on several lines, so lines are
    joined until the quoted string that opened is closed again.
    """
    pending: list[str] = []
    for line in text.splitlines():
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


def _unescape(name: str) -> str:
    """Undo the backslash escapes of a prefixed name or a quoted literal."""
    return re.sub(r"\\(.)", r"\1", name)


def _names(expression: str) -> set[str]:
    """Return the class names of a class expression, union or not."""
    return {_unescape(name) for name in re.findall(rf":({_REF})", expression)}


def _range_names(expression: str) -> set[str]:
    """Return the range of a property, a datatype or a set of class names."""
    if expression.startswith("xsd:"):
        return {expression.removeprefix("xsd:")}
    return _names(expression)


def _read_axiom(line: str, emitted: Emitted) -> None:
    """Add one axiom to `emitted`."""
    if m := _DECLARATION.match(line):
        kind, name = m.groups()
        declared = {
            "Class": emitted.classes,
            "ObjectProperty": emitted.object_properties,
            "DataProperty": emitted.data_properties,
        }.get(kind)
        if declared is not None:
            declared.add(_unescape(name))
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
            (_unescape(prop), _unescape(subject), _unescape(value), datatype)
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
        emitted.inverses.add(inverse_pair(_unescape(prop), _unescape(inverse)))
    elif m := _TRANSITIVE.match(line):
        emitted.transitive.add(_unescape(m.group(1)))
    elif m := _ENTITY_ANNOTATION.match(line):
        kind, name, text = m.groups()
        emitted.annotations.add((_unescape(name), kind, _unescape(text)))
