"""The boundary between a build and the axioms it produces.

`AxiomSink` takes a small vocabulary of intents from the emitters, such as
adding a class or asserting a set of relations. It validates each one,
suppresses schema statements it has already made, and hands the survivors to a
backend through the abstract `write_*` methods. Name checking happens here.

Two backends implement those methods, one building an ontology in memory and
one writing functional syntax straight to a file. Everything above this module
works the same way with either.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd

from semanticlog.core.enums import OWLProfile
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks.ledger import NameLedger

OWL_DATATYPES: frozenset[str] = frozenset({"integer", "double", "boolean", "dateTime"})

_XSD = "http://www.w3.org/2001/XMLSchema#"
_RDFS = "http://www.w3.org/2000/01/rdf-schema#"


def normalize_value(value: Any) -> Any:
    """Convert a pandas value into something a literal can be made from.

    Every flavour of missing value becomes None, and numpy scalars become
    their Python equivalents, so that a backend only has to handle plain
    types.

    Args:
        value: A cell from a table.

    Returns:
        The plain Python value, or None when it was missing.
    """
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.to_pydatetime()
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, np.generic):
        value = value.item()
        return None if isinstance(value, float) and pd.isna(value) else value
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _pairs(rows: pd.DataFrame, reverse: bool = False) -> Iterator[tuple[str, str]]:
    """Yield the two columns of a table as pairs of names.

    A generator rather than a list, because a relation of a large log holds one
    pair per row and a backend that writes them out needs to see only one at a
    time.

    Args:
        rows: Two columns, the subject and the target.
        reverse: Swap the two, for asserting the inverse of a property.
    """
    for subject, target in rows.itertuples(index=False, name=None):
        yield (str(target), str(subject)) if reverse else (str(subject), str(target))


class AxiomSink(ABC):
    """Accepts build intents, checks them, and writes axioms.

    Subclasses implement the `write_*` methods and inherit everything else.
    A name that is unsafe in an IRI, or that would be used for two different
    things, is refused here with a message naming both sources.

    Attributes:
        iri: Base IRI of the ontology, always ending in a fragment separator.
        settings: The settings of this build, consulted for the profile and
            for how inverses are represented.
    """

    iri: str
    settings: OntoBuildSettings

    def __init__(self, iri: str, settings: OntoBuildSettings) -> None:
        """Prepare a sink.

        Args:
            iri: Base IRI. A trailing separator is added when missing.
            settings: Settings of the build this sink serves.
        """
        self.iri = iri if iri.endswith("#") else iri + "#"
        self.settings = settings
        self.names = NameLedger()
        self._dropped_unions: list[int] = []

    def _declare_name(self, name: str, kind: str, source: str) -> None:
        """Claim a name for one thing, and refuse a second claim on it."""
        self.names.declare(name, kind, source)

    def _check_name(self, name: str, source: str) -> None:
        """Refuse a name that cannot be an IRI local name."""
        self.names.check_name(name, source)

    def _check_individuals(self, names: Iterable[str], source: str) -> pd.Series:
        """Check a batch of individual names, and return the ones to emit."""
        return self.names.check_individuals(names, source)

    @property
    def declared_count(self) -> int:
        """How many distinct classes and properties have been declared."""
        return self.names.count

    def declared_by_kind(self) -> dict[str, int]:
        """How many classes, object properties and data properties there are.

        Both backends can answer this from the name ledger, so a streamed
        build and an in-memory one report the same measurements.
        """
        return self.names.by_kind()

    def declared_since(self, mark: int) -> dict[str, list[str]]:
        """Return the names declared after a mark, grouped by kind.

        Args:
            mark: A `declared_count` read before the step being measured.

        Returns:
            Names by kind, in the order they were declared.
        """
        return self.names.since(mark)

    @property
    def dropped_unions(self) -> tuple[int, int]:
        """How many unions the profile refused, and over how many classes."""
        return len(self._dropped_unions), sum(self._dropped_unions)

    def is_declared(self, name: str) -> bool:
        """Whether a class or property of this name has been declared."""
        return self.names.is_declared(name)

    @property
    def axiom_count(self) -> int:
        """How many axioms were written, or -1 when the backend cannot say."""
        return -1

    def _first_time(self, *key: object) -> bool:
        """Report whether this schema statement is new, and record it."""
        return self.names.first_time(*key)

    @abstractmethod
    def write_declaration(self, name: str, kind: str) -> None:
        """Declare a class, object property or data property.

        Args:
            name: Local name of the entity.
            kind: One of `class`, `object` or `data`.
        """

    @abstractmethod
    def write_sub_class_of(self, sub: str, superclass: list[str]) -> None:
        """State that a class is below a superclass, or below their union."""

    @abstractmethod
    def write_sub_property_of(self, sub: str, parent: str, data_prop: bool) -> None:
        """State that a property is below another property."""

    @abstractmethod
    def write_domain(self, prop: str, expression: list[str], data_prop: bool) -> None:
        """State the domain of a property, as a class or a union of classes."""

    @abstractmethod
    def write_range(self, prop: str, expression: list[str], data_prop: bool) -> None:
        """State the range of a property, as classes or as a datatype."""

    @abstractmethod
    def write_inverse_properties(self, prop: str, inverse: str) -> None:
        """State that two object properties are inverses of each other."""

    @abstractmethod
    def write_transitive(self, prop: str) -> None:
        """State that an object property is transitive."""

    @abstractmethod
    def write_class_assertions(
        self, class_name: str, individuals: Iterable[str]
    ) -> None:
        """Assert that each individual is a member of the class."""

    @abstractmethod
    def write_object_assertions(
        self, prop: str, pairs: Iterable[tuple[str, str]]
    ) -> None:
        """Assert the property between each pair of individuals."""

    @abstractmethod
    def write_data_assertions(
        self, prop: str, pairs: Iterable[tuple[str, Any]]
    ) -> None:
        """Assert the property between each individual and its literal value."""

    @abstractmethod
    def write_annotations(
        self, annotation: str, pairs: Iterable[tuple[str, str]]
    ) -> None:
        """Annotate entities.

        Args:
            annotation: Either `label` or `comment`.
            pairs: Entity name and the text to attach to it.
        """

    @abstractmethod
    def write_ontology_annotation(self, property_iri: str, value: str) -> None:
        """Annotate the ontology itself, which is where provenance goes."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Write the ontology to a file."""

    def result_target(self) -> tuple[Any, str | None]:
        """Return what a finished build carries away from this sink.

        A sink holding an ontology returns it, one that wrote it out returns
        where, and one that keeps neither returns neither. The builder asks
        instead of looking for attributes, so a backend states which it is.
        """
        return None, None

    def close(self) -> None:  # noqa: B027
        """Release whatever the backend holds open.

        Optional. A sink that keeps nothing open need not override this.
        """

    def abort(self) -> None:
        """Give up on this build and leave nothing half-finished behind.

        A sink that has been writing to a file removes it, since a partial
        ontology that parses is worse than no file at all. The default is to
        close, which is right for a sink that only holds memory.
        """
        self.close()

    def _disjuncts(
        self, expression: str | list[str] | None, source: str = "a class expression"
    ) -> list[str] | None:
        """Normalise a class expression to a checked list, or None if empty."""
        if expression is None:
            return None
        parts = [expression] if isinstance(expression, str) else list(expression)
        for part in parts:
            self._check_name(part, source)
        return parts or None

    def _profile_allows(self, expression: list[str] | None) -> list[str] | None:
        """Drop a union under OWL 2 EL, which has no disjunction."""
        if expression is None or len(expression) <= 1:
            return expression
        if self.settings.profile == OWLProfile.OWL2EL:
            self._dropped_unions.append(len(expression))
            return None
        return expression

    def add_class(
        self,
        class_name: str,
        superclass: str | list[str] | None = None,
        *,
        source: str = "a build step",
    ) -> None:
        """Declare a class and place it under its superclass.

        Args:
            class_name: Local name of the class.
            superclass: One superclass, or several to place it under their
                union. None leaves it at the top.
            source: Where this class comes from, quoted in any error about a
                name collision.

        Raises:
            UnsafeLocalNameError: If a name cannot appear in an IRI.
            NameCollisionError: If the name is already used for something else.
        """
        self._declare_name(class_name, "class", source)
        parents = self._profile_allows(
            self._disjuncts(superclass, f"the superclass of {class_name!r}")
        )
        if self._first_time("declare", class_name, "class"):
            self.write_declaration(class_name, "class")
        if parents and self._first_time("sub_class", class_name, tuple(parents)):
            self.write_sub_class_of(class_name, parents)

    def add_classes(
        self,
        class_list: list[str],
        superclasses: list[str] | str | None = None,
        *,
        source: str = "a build step",
    ) -> None:
        """Declare several classes at once.

        Args:
            class_list: Local names of the classes.
            superclasses: One superclass for all of them, one per class in the
                same order, or None.
            source: Where these classes come from.

        Raises:
            ValueError: If a list of superclasses does not match the number of
                classes.
        """
        if isinstance(superclasses, str) or superclasses is None:
            for class_name in class_list:
                self.add_class(class_name, superclasses, source=source)
        elif isinstance(superclasses, list) and len(class_list) == len(superclasses):
            for class_name, superclass in zip(class_list, superclasses, strict=True):
                self.add_class(class_name, superclass, source=source)
        else:
            raise ValueError(
                "superclasses must be a list of str with the same length as "
                "class_list, a str, or None"
            )

    def _validate_property_args(
        self,
        range_class: str | list[str] | None,
        data_prop: bool,
        create_inverse: bool,
        inverse_name: str | None,
    ) -> None:
        """Reject property arguments that contradict each other."""
        if data_prop and create_inverse:
            raise ValueError(
                "Data properties cannot have inverses; set create_inverse to False for data properties."
            )
        if inverse_name is not None and not create_inverse:
            raise ValueError(
                "When adding an inverse via inverse_name create_inverse must be set to true!"
            )
        parts = self._disjuncts(range_class, "a property range")
        if data_prop and parts is not None and not set(parts) <= OWL_DATATYPES:
            raise ValueError(f"Unknown range for a data property: {range_class}")
        if not data_prop and parts is not None and set(parts) & OWL_DATATYPES:
            raise ValueError("Can not create an object property with a data as range!")

    def add_property(
        self,
        prop_name: str,
        domain_class: str | list[str] | None = None,
        range_class: str | list[str] | None = None,
        superprop: str | None = None,
        create_inverse: bool = False,
        inverse_name: str | None = None,
        data_prop: bool = False,
        *,
        source: str = "a build step",
    ) -> None:
        """Declare a property, with its domain, range and optional inverse.

        The inverse is declared alongside the property. Whether the two are
        also related by an axiom depends on the inverse representation of the
        build, so an emitter asks for an inverse without deciding how it is
        expressed.

        Args:
            prop_name: Local name of the property.
            domain_class: Class or classes the subject may belong to.
            range_class: Class or classes the target may belong to, or a
                datatype for a data property.
            superprop: Property to place this one below. The inverse is placed
                below the inverse of that property.
            create_inverse: Also declare the inverse property.
            inverse_name: Name for the inverse, instead of the derived one.
            data_prop: Whether this is a data property.
            source: Where this property comes from.

        Raises:
            ValueError: If the arguments contradict each other, such as an
                inverse for a data property or a datatype range on an object
                property.
            NameCollisionError: If the name is already used for something
                else, or the inverse name is already the inverse of another
                property.
        """
        self._validate_property_args(
            range_class, data_prop, create_inverse, inverse_name
        )
        kind = "data" if data_prop else "object"
        self._declare_name(prop_name, kind, source)

        if self._first_time("declare", prop_name, kind):
            self.write_declaration(prop_name, kind)
        if superprop is not None:
            self.add_sub_property(prop_name, superprop, data_prop, source=source)
        domain = self._profile_allows(
            self._disjuncts(domain_class, f"the domain of {prop_name!r}")
        )
        if domain and self._first_time("domain", prop_name, tuple(domain), data_prop):
            self.write_domain(prop_name, domain, data_prop)
        range_ = self._profile_allows(
            self._disjuncts(range_class, f"the range of {prop_name!r}")
        )
        if range_ and self._first_time("range", prop_name, tuple(range_), data_prop):
            self.write_range(prop_name, range_, data_prop)

        if create_inverse:
            self._declare_inverse(prop_name, inverse_name, kind, superprop, source)

    def _declare_inverse(
        self,
        prop_name: str,
        inverse_name: str | None,
        kind: str,
        superprop: str | None,
        source: str,
    ) -> None:
        """Declare the inverse of a property, and relate the two if asked.

        Whether the pair also gets an axiom depends on the inverse
        representation of the build, so the caller asks for an inverse without
        deciding how it is expressed.

        Args:
            prop_name: Property the inverse belongs to.
            inverse_name: Name for it, or None for the derived one.
            kind: `object` or `data`, matching the property.
            superprop: Property the original sits below, whose own inverse the
                new one is placed below.
            source: Where the property comes from.
        """
        if inverse_name is None:
            inverse_name = self.inverse_name(prop_name)
        self.names.claim_inverse(prop_name, inverse_name, source)
        self._declare_name(inverse_name, kind, f"inverse of {source}")
        if self._first_time("declare", inverse_name, kind):
            self.write_declaration(inverse_name, kind)
        if self.settings.axiomatic_inverses and self._first_time(
            "inverse", prop_name, inverse_name
        ):
            self.write_inverse_properties(prop_name, inverse_name)
        if superprop is not None:
            self.add_sub_property(
                inverse_name,
                self.inverse_name(superprop),
                kind == "data",
                source=f"inverse of {source}",
            )

    def add_sub_property(
        self,
        sub: str,
        parent: str,
        data_prop: bool = False,
        *,
        source: str = "a build step",
    ) -> None:
        """Place a property below another one.

        Raises:
            UnsafeLocalNameError: If either name cannot appear in an IRI.
        """
        self._check_name(sub, source)
        self._check_name(parent, source)
        if self._first_time("sub_property", sub, parent, data_prop):
            self.write_sub_property_of(sub, parent, data_prop)

    def add_transitive(self, prop: str) -> None:
        """Declare an object property transitive."""
        self._check_name(prop, "a transitivity axiom")
        if self._first_time("transitive", prop):
            self.write_transitive(prop)

    def assert_individuals(self, individuals: Sequence[str], class_name: str) -> None:
        """Assert that each individual belongs to a class.

        Repeats and missing values are dropped, so a column can be passed
        straight from the store. A missing value carries no name to assert,
        and every one of them would otherwise become the same individual.
        """
        names = self._check_individuals(
            individuals, f"an individual of class {class_name!r}"
        )
        if not names.empty:
            self.write_class_assertions(class_name, names)

    def assert_relations(
        self,
        df: pd.DataFrame,
        prop: str,
        from_key: str,
        to_key: str,
        data_prop: bool = False,
        inverse_name: str | None = None,
    ) -> None:
        """Assert one property for every row of a table.

        Rows with a missing endpoint are dropped and duplicate pairs are
        collapsed, so callers can pass a table straight from the store. When
        the build materialises inverses, the reversed pairs are asserted here
        as well.

        Args:
            df: Table holding the pairs.
            prop: Property to assert.
            from_key: Column holding the subject.
            to_key: Column holding the target, or the literal value.
            data_prop: Whether this is a data property.
            inverse_name: Name to use for the reversed pairs, instead of the
                derived one.
        """
        rows = df[[from_key, to_key]]
        rows = rows[rows[from_key].notna() & rows[to_key].notna()]
        rows = rows.drop_duplicates([from_key, to_key])

        if data_prop:
            self._check_individuals(rows[from_key], f"the subject of {prop!r}")
            self.write_data_assertions(prop, rows.itertuples(index=False, name=None))
            return

        if rows.empty:
            return
        self._check_individuals(rows[from_key], f"the subject of {prop!r}")
        self._check_individuals(rows[to_key], f"the target of {prop!r}")
        self.write_object_assertions(prop, _pairs(rows))
        if self.settings.implicit_inverses:
            inv = inverse_name or self.inverse_name(prop)
            self.write_object_assertions(inv, _pairs(rows, reverse=True))

    def annotate_entity(
        self, iri_fragment: str, label: str | None = None, comment: str | None = None
    ) -> None:
        """Attach a label and a comment to a declared class or property.

        Args:
            iri_fragment: Local name of the entity.
            label: Text for `rdfs:label`.
            comment: Text for `rdfs:comment`.

        Raises:
            ValueError: If nothing declared the entity, which would otherwise
                annotate an IRI that appears nowhere else.
        """
        if not self.is_declared(iri_fragment):
            raise ValueError(
                f"Cannot annotate {iri_fragment!r}: no build step declared it as a "
                "class or a property. Declare it first, or filter the names with "
                "is_declared()."
            )
        if label and self._first_time("label", iri_fragment, label):
            self.write_annotations("label", [(iri_fragment, label)])
        if comment and self._first_time("comment", iri_fragment, comment):
            self.write_annotations("comment", [(iri_fragment, comment)])

    def annotate_individuals_bulk(self, pairs: list[tuple[str, str]]) -> None:
        """Label many individuals at once.

        Individuals are not declared as classes or properties are, so these
        labels are written without the declaration check.
        """
        if pairs:
            self.write_annotations("label", pairs)

    def inverse_name(self, prop: str) -> str:
        """Return the inverse of a property, including inverses named in this build."""
        return self.names.inverse_name(prop)
