"""The boundary between a build and the axioms it produces.

`AxiomSink` takes a small vocabulary of intents from the emitters, such as
adding a class or asserting a set of relations. It validates each one,
suppresses schema statements it has already made, and hands the survivors to a
backend through the abstract `write_*` methods. Name checking happens here.

A sink also has a life: `open_ontology` with the record of the build, then the
axioms, then `close`, or `abort` if the build gave up. Both ends are stated
here rather than left to a backend, since a backend that needs one of them
would otherwise impose it on everything above.

Two backends implement those methods, one building an ontology in memory and
one writing functional syntax straight to a file. Everything above this module
works the same way with either.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from semanticlog.core.enums import OWLProfile
from semanticlog.onto.sinks.ledger import NameLedger

OWL_DATATYPES: frozenset[str] = frozenset({"integer", "double", "boolean", "dateTime"})

PARTIAL_SUFFIX = ".partial"
"""What an unfinished ontology is called while it is being written.

A sink writes here and moves the file into place once the ontology is
complete, so a build that fails leaves whatever was at the destination
untouched rather than a truncated file where a finished one used to be.
"""

_XSD = "http://www.w3.org/2001/XMLSchema#"
_RDFS = "http://www.w3.org/2000/01/rdf-schema#"


class AxiomSink(ABC):
    """Accepts build intents, checks them, and writes axioms.

    Subclasses implement the `write_*` methods and inherit everything else.
    A name that is unsafe in an IRI, or that would be used for two different
    things, is refused here with a message naming both sources.

    What a sink knows about a build is its base IRI and its profile. Which
    axioms are worth writing is decided above it, so that the boundary can be
    described on its own terms rather than as whatever the two backends agree
    on.

    Attributes:
        iri: Base IRI of the ontology, always ending in a fragment separator.
        profile: OWL 2 profile the output has to stay inside.
    """

    iri: str
    profile: OWLProfile

    def __init__(self, iri: str, profile: OWLProfile) -> None:
        """Prepare a sink.

        Args:
            iri: Base IRI. A trailing separator is added when missing.
            profile: Profile of the build this sink serves.
        """
        self.iri = iri if iri.endswith("#") else iri + "#"
        self.profile = profile
        self.names = NameLedger()
        self._dropped_unions: list[int] = []
        self._opened = False
        self._closed = False

    def _declare_name(self, name: str, kind: str, source: str) -> None:
        self.names.declare(name, kind, source)

    def _check_name(self, name: str, source: str) -> None:
        self.names.check_name(name, source)

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

    @abstractmethod
    def write_declaration(self, name: str, kind: str) -> None:
        """Declare a class, a property or an annotation property.

        Args:
            name: Local name of the entity.
            kind: One of `class`, `object`, `data` or `annotation`.
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
    def write_ontology_header(self, annotations: Mapping[str, str]) -> None:
        """Begin the ontology and annotate it, which is where provenance goes."""

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

    def open_ontology(self, annotations: Mapping[str, str] | None = None) -> None:
        """Begin the ontology, annotated with the record of the build.

        An ontology annotation belongs inside the ontology header, and a sink
        writing a file has put that header out as soon as it writes anything
        else. So the whole record arrives here, once, before any axiom. A sink
        nothing opened opens itself on its first axiom, and then carries no
        record.

        A build invents its own annotation properties for the record it keeps,
        and OWL 2 DL wants every one of them declared. A property from another
        namespace, such as `owl:versionInfo`, is already declared where it is
        defined and is left alone.

        Args:
            annotations: Full IRI of each annotation property, to the text to
                record under it.

        Raises:
            ValueError: If the ontology is already open, which means an axiom
                was written before the record.
        """
        if self._opened:
            raise ValueError(
                f"{type(self).__name__} has already begun its ontology, and an "
                "ontology annotation belongs inside the header. Open the sink "
                "with the whole record before emitting anything."
            )
        self._opened = True
        record = dict(annotations or {})
        self.write_ontology_header(record)
        for property_iri in record:
            if not property_iri.startswith(self.iri):
                continue
            name = property_iri[len(self.iri) :]
            if self.names.first_declaration(name, "annotation"):
                self.write_declaration(name, "annotation")

    def _ensure_open(self) -> None:
        """Begin an ontology nothing opened, so it carries no record."""
        if not self._opened:
            self.open_ontology()

    def _begin_write(self) -> None:
        """Take one axiom, opening the ontology when it is the first.

        Every backend calls this before it writes, so where an ontology begins
        and where it stops taking axioms are the same two points either way.

        Raises:
            ValueError: If the sink is closed. A backend can finish an ontology
                in a way that changes how the axioms before it read, so nothing
                may arrive afterwards.
        """
        if self._closed:
            raise ValueError(
                f"{type(self).__name__} is closed and cannot take more axioms. "
                "A sink closes once, when the build that owns it has finished."
            )
        self._ensure_open()

    def close(self) -> None:
        """Finish the ontology, which is the last thing a sink does.

        Whatever a backend can only do once no further axioms are coming
        happens in `_finish`. Closing twice is harmless, and an ontology
        nothing opened is opened here so that an empty build still produces
        one.
        """
        if self._closed:
            return
        self._ensure_open()
        try:
            self._finish()
        finally:
            self._closed = True

    def _finish(self) -> None:  # noqa: B027
        """Complete the ontology, once the last axiom has arrived.

        Optional. A backend with nothing left to do need not override this.
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
        """Normalize a class expression to a checked list, or None if empty."""
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
        if self.profile == OWLProfile.OWL2EL:
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
        if self.names.first_declaration(class_name, "class"):
            self.write_declaration(class_name, "class")
        if parents and self.names.first_sub_class(class_name, parents):
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
        axiomatic_inverse: bool,
    ) -> None:
        """Reject property arguments that contradict each other."""
        if data_prop and create_inverse:
            raise ValueError(
                "A data property cannot have an inverse. Set create_inverse=False, "
                "or data_prop=False if an object property was meant."
            )
        if inverse_name is not None and not create_inverse:
            raise ValueError(
                f"inverse_name={inverse_name!r} was given with create_inverse=False, "
                f"so no inverse would be declared to carry it. Set "
                f"create_inverse=True, or drop inverse_name."
            )
        if axiomatic_inverse and not create_inverse:
            raise ValueError(
                "axiomatic_inverse=True was given with create_inverse=False, so "
                "there would be no inverse for the axiom to relate. Set "
                "create_inverse=True, or drop axiomatic_inverse."
            )
        parts = self._disjuncts(range_class, "a property range")
        if data_prop and parts is not None and not set(parts) <= OWL_DATATYPES:
            raise ValueError(
                f"Range {range_class!r} is not a datatype, and a data property "
                f"ranges over one of {sorted(OWL_DATATYPES)}."
            )
        if not data_prop and parts is not None and set(parts) & OWL_DATATYPES:
            raise ValueError(
                f"Range {range_class!r} names a datatype, which an object property "
                f"cannot range over. Set data_prop=True, or name a class."
            )

    def add_property(
        self,
        prop_name: str,
        domain_class: str | list[str] | None = None,
        range_class: str | list[str] | None = None,
        superprop: str | None = None,
        create_inverse: bool = False,
        inverse_name: str | None = None,
        data_prop: bool = False,
        axiomatic_inverse: bool = False,
        *,
        source: str = "a build step",
    ) -> None:
        """Declare a property, with its domain, range and optional inverse.

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
            axiomatic_inverse: Also state that the two are inverses of each
                other. A build that materializes the reversed assertions
                instead declares the inverse without the axiom.
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
            range_class, data_prop, create_inverse, inverse_name, axiomatic_inverse
        )
        kind = "data" if data_prop else "object"
        self._declare_name(prop_name, kind, source)

        if self.names.first_declaration(prop_name, kind):
            self.write_declaration(prop_name, kind)
        if superprop is not None:
            self.add_sub_property(prop_name, superprop, data_prop, source=source)
        domain = self._profile_allows(
            self._disjuncts(domain_class, f"the domain of {prop_name!r}")
        )
        if domain and self.names.first_domain(prop_name, domain, data_prop):
            self.write_domain(prop_name, domain, data_prop)
        range_ = self._profile_allows(
            self._disjuncts(range_class, f"the range of {prop_name!r}")
        )
        if range_ and self.names.first_range(prop_name, range_, data_prop):
            self.write_range(prop_name, range_, data_prop)

        if create_inverse:
            self._declare_inverse(
                prop_name, inverse_name, kind, superprop, source, axiomatic_inverse
            )

    def _declare_inverse(
        self,
        prop_name: str,
        inverse_name: str | None,
        kind: str,
        superprop: str | None,
        source: str,
        axiomatic: bool,
    ) -> None:
        """Declare the inverse of a property, and relate the two if asked.

        Args:
            prop_name: Property the inverse belongs to.
            inverse_name: Name for it, or None for the derived one.
            kind: `object` or `data`, matching the property.
            superprop: Property the original sits below, whose own inverse the
                new one is placed below.
            source: Where the property comes from.
            axiomatic: Also state that the two are inverses of each other.
        """
        if inverse_name is None:
            inverse_name = self.inverse_name(prop_name)
        self.names.claim_inverse(prop_name, inverse_name, source)
        self._declare_name(inverse_name, kind, f"inverse of {source}")
        if self.names.first_declaration(inverse_name, kind):
            self.write_declaration(inverse_name, kind)
        if axiomatic and self.names.first_inverse(prop_name, inverse_name):
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
        if self.names.first_sub_property(sub, parent, data_prop):
            self.write_sub_property_of(sub, parent, data_prop)

    def add_transitive(self, prop: str) -> None:
        """Declare an object property transitive."""
        self._check_name(prop, "a transitivity axiom")
        if self.names.first_transitive(prop):
            self.write_transitive(prop)

    def assert_individuals(self, class_name: str, individuals: Iterable[Any]) -> None:
        """Assert that each individual belongs to a class.

        Repeats and missing values are dropped, so a column can be passed
        straight from the store. A missing value carries no name to assert,
        and every one of them would otherwise become the same individual.
        """
        names = self.names.check_individuals(
            individuals, f"an individual of class {class_name!r}"
        )
        if names:
            self.write_class_assertions(class_name, names)

    def assert_relations(self, prop: str, pairs: Iterable[tuple[Any, Any]]) -> None:
        """Assert an object property between each pair of individuals.

        Every name of every assertion is checked, so the accepting case stays
        cheap and only a refusal pays for the message naming where it came
        from.

        Args:
            prop: Property to assert.
            pairs: Subject and target of each assertion.
        """
        self.write_object_assertions(prop, self._checked_pairs(prop, pairs))

    def assert_data(self, prop: str, pairs: Iterable[tuple[Any, Any]]) -> None:
        """Assert a data property between each individual and its value.

        Args:
            prop: Property to assert.
            pairs: Subject and literal value of each assertion.
        """
        self.write_data_assertions(prop, self._checked_subjects(prop, pairs))

    def _checked_pairs(
        self, prop: str, pairs: Iterable[tuple[Any, Any]]
    ) -> Iterator[tuple[str, str]]:
        """Yield each pair as the two names to emit it under."""
        accepts = self.names.accepts_individual
        for subject, target in pairs:
            name, other = str(subject), str(target)
            if not accepts(name):
                self.names.check_individual(name, f"the subject of {prop!r}")
            if not accepts(other):
                self.names.check_individual(other, f"the target of {prop!r}")
            yield name, other

    def _checked_subjects(
        self, prop: str, pairs: Iterable[tuple[Any, Any]]
    ) -> Iterator[tuple[str, Any]]:
        """Yield each pair with its subject as a name and its value untouched."""
        accepts = self.names.accepts_individual
        for subject, value in pairs:
            name = str(subject)
            if not accepts(name):
                self.names.check_individual(name, f"the subject of {prop!r}")
            yield name, value

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
        if label and self.names.first_annotation("label", iri_fragment, label):
            self.write_annotations("label", [(iri_fragment, label)])
        if comment and self.names.first_annotation("comment", iri_fragment, comment):
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
