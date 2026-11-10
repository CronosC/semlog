"""One emitted ontology, indexed by what a test asks of it.

`read_functional` and `read_rdf` fill an `Emitted` from a written file.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field


def inverse_pair(one: str, other: str) -> tuple[str, str]:
    """Return an inverse pair in a fixed order."""
    return (one, other) if one <= other else (other, one)


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
        disjoint: Sets of classes stated pairwise disjoint.
        functional: Properties declared functional.
        defined: Classes stated equivalent to a class expression.
        inverse_functional: Object properties declared inverse functional.
        annotations: Entity annotations, as (name, label or comment, text).
        ontology_annotations: The build record, keyed by annotation name.
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
    disjoint: set[frozenset[str]] = field(default_factory=set)
    functional: set[str] = field(default_factory=set)
    defined: set[str] = field(default_factory=set)
    inverse_functional: set[str] = field(default_factory=set)
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
