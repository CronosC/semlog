"""Base class of the sinks that turn build steps into axioms.

`AxiomSink` takes requests from the emitters, such as adding a class or
asserting a set of relations. It checks each name, drops a schema statement it
has written before, and passes the rest to the abstract `write_*` methods. A
sink is opened with `open_ontology`, takes axioms, and ends with `close`, or
with `abort` when the build failed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import OWLProfile, SinkEntity
from semanticlog.core.errors import ConfigError
from semanticlog.onto.sinks.ledger import NameLedger
from semanticlog.onto.sinks.literals import OWL_DATATYPES, Literal, resolve_literal

if TYPE_CHECKING:
    from owlapy.owl_ontology import Ontology

PARTIAL_SUFFIX = ".partial"
"""Suffix of the file a sink writes to until the ontology is complete.

The finished file then replaces the destination in one move.
"""


class AxiomSink(ABC):
    """Checks build requests and writes them as axioms.

    Subclasses implement the `write_*` methods. A name that cannot appear in an
    IRI, or that is used for two different entities, raises with a message
    naming both sources.

    A TBox `Term` is written under `TBOX_IRI`, every other name under the log
    base.

    Args:
        iri: Log base IRI. A trailing `/` is added when missing.
        profile: Profile of the build.

    Attributes:
        iri: Log base IRI, ending in `/`.
        profile: OWL 2 profile the output stays inside.

    Raises:
        ConfigError: If the base IRI is empty, holds whitespace or `>`, or
            ends in `#`.
    """

    iri: str
    profile: OWLProfile

    def __init__(self, iri: str, profile: OWLProfile) -> None:
        if not iri or any(c.isspace() for c in iri) or ">" in iri:
            raise ConfigError(
                f"Base IRI {iri!r} cannot be written into an ontology. Use a "
                "non-empty IRI without whitespace or '>'."
            )
        if iri.endswith("#"):
            raise ConfigError(
                f"iri must end in '/', got {iri!r}. Replace the trailing '#' with '/'."
            )
        self.iri = iri if iri.endswith("/") else iri + "/"
        self.profile = profile
        self._names = NameLedger()
        self._dropped_unions = 0
        self._replaced_unions = 0
        self._axiom_count = 0
        self._opened = False
        self._closed = False

    @property
    def ontology_iri(self) -> str:
        """IRI of the ontology: the log base without its trailing `/`."""
        return self.iri.removesuffix("/")

    def iri_of(self, name: str) -> str:
        """Return the full IRI of a name, under the TBox or the log base."""
        return f"{vocab.TBOX_IRI if vocab.is_tbox(name) else self.iri}{name}"

    def _declare_name(self, name: str, kind: SinkEntity, source: str) -> None:
        self._names.declare(name, kind, source)

    def _check_name(self, name: str, source: str) -> None:
        self._names.check_name(name, source)

    @property
    def declared_count(self) -> int:
        """How many distinct classes and properties have been declared."""
        return self._names.count

    def declared_by_kind(self) -> dict[str, int]:
        """How many classes and properties of each kind have been declared."""
        return self._names.by_kind()

    def declared_since(self, mark: int) -> dict[str, list[str]]:
        """Return the names declared after a mark, grouped by kind.

        Args:
            mark: A `declared_count` read before the step being measured.

        Returns:
            Names by kind, in declaration order.
        """
        return self._names.since(mark)

    @property
    def dropped_unions(self) -> int:
        """How many axioms were dropped under OWL 2 EL for holding a union."""
        return self._dropped_unions

    @property
    def replaced_unions(self) -> int:
        """How many unions OWL 2 EL replaced by a single class."""
        return self._replaced_unions

    def is_declared(self, name: str) -> bool:
        """Whether a class or property of this name has been declared."""
        return self._names.is_declared(name)

    @property
    def axiom_count(self) -> int:
        """How many axioms have been written."""
        return self._axiom_count

    @abstractmethod
    def write_declaration(self, name: str, kind: SinkEntity) -> None:
        """Declare a class, a property or an annotation property.

        Args:
            name: Local name of the entity.
            kind: What to declare it as.
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
    def write_range(self, prop: str, expression: list[str]) -> None:
        """State the range of an object property, as a class or a union."""

    @abstractmethod
    def write_data_range(self, prop: str, datatype: str) -> None:
        """State the range of a data property, as an XSD datatype local name."""

    @abstractmethod
    def write_inverse_properties(self, prop: str, inverse: str) -> None:
        """State that two object properties are inverses of each other."""

    @abstractmethod
    def write_transitive(self, prop: str) -> None:
        """State that an object property is transitive."""

    @abstractmethod
    def write_disjoint_classes(self, classes: list[str]) -> None:
        """State that no individual belongs to two of the classes."""

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
        self, prop: str, pairs: Iterable[tuple[str, Literal]]
    ) -> None:
        """Assert the property between each individual and its resolved literal."""

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
        """Begin the ontology and write its annotations."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Write the ontology to a file."""

    def result_target(self) -> tuple[Ontology | None, str | None]:
        """Return the ontology object and the path written. Either may be None."""
        return None, None

    def open_ontology(self, annotations: Mapping[str, str] | None = None) -> None:
        """Begin the ontology and write the build record as ontology annotations.

        Call this before the first axiom. A sink that takes an axiom first
        opens itself with no record. Each annotation property in the TBox
        namespace is checked, claimed and declared. A property from another
        namespace, such as `owl:versionInfo`, is not.

        Args:
            annotations: Text to record, keyed by the full IRI of its
                annotation property.

        Raises:
            ValueError: If an axiom was written before this call.
            UnsafeLocalNameError: If a property name cannot appear in an IRI.
            NameCollisionError: If a property name is declared as another kind.
        """
        if self._opened:
            raise ValueError(
                f"{type(self).__name__} is open. Call open_ontology() with the "
                "build record before the first axiom."
            )
        record = dict(annotations or {})
        names = [
            vocab.Term(property_iri.removeprefix(vocab.TBOX_IRI))
            for property_iri in record
            if property_iri.startswith(vocab.TBOX_IRI)
        ]
        for name in names:
            self._declare_name(name, SinkEntity.ANNOTATION_PROPERTY, "the build record")
        self._opened = True
        self.write_ontology_header(record)
        for name in names:
            if self._names.first_declaration(name, SinkEntity.ANNOTATION_PROPERTY):
                self.write_declaration(name, SinkEntity.ANNOTATION_PROPERTY)

    def _ensure_open(self) -> None:
        """Open the ontology with no record if it is not open."""
        if not self._opened:
            self.open_ontology()

    def _begin_write(self, count: int = 1) -> None:
        """Open the ontology if needed, and count the axioms about to be written.

        Raises:
            ValueError: If the sink is closed.
        """
        if self._closed:
            raise ValueError(
                f"{type(self).__name__} is closed and takes no more axioms."
            )
        self._ensure_open()
        self._axiom_count += count

    def close(self) -> None:
        """Finish the ontology. A second call does nothing."""
        if self._closed:
            return
        self._ensure_open()
        try:
            self._finish()
        finally:
            self._closed = True

    def _finish(self) -> None:  # noqa: B027
        """Run backend work after the last axiom. Does nothing by default."""

    def abort(self) -> None:
        """Stop the build. A file-writing sink removes its partial file.

        The default closes the sink.
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

    def _profile_allows(
        self, expression: list[str] | None, fallback: str | None = None
    ) -> list[str] | None:
        """Return the expression, or what replaces a union under OWL 2 EL.

        Under OWL 2 EL a union becomes `fallback`, or None without one. Call it
        after the ledger check, so a repeated axiom is counted once.
        """
        if expression is None or len(expression) <= 1:
            return expression
        if self.profile != OWLProfile.OWL2EL:
            return expression
        if fallback is None:
            self._dropped_unions += 1
            return None
        self._replaced_unions += 1
        return [fallback]

    def _resolve(self, value: Any) -> Literal | None:
        """Return the literal a value is emitted as, or None when it is missing."""
        return resolve_literal(value, self.profile)

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
            source: Origin of the class, quoted in an error about a name
                collision.

        Raises:
            UnsafeLocalNameError: If a name cannot appear in an IRI.
            NameCollisionError: If the name is declared as another kind.
        """
        self._declare_name(class_name, SinkEntity.CLASS, source)
        parents = self._disjuncts(superclass, f"the superclass of {class_name!r}")
        if self._names.first_declaration(class_name, SinkEntity.CLASS):
            self.write_declaration(class_name, SinkEntity.CLASS)
        if parents and self._names.first_sub_class(class_name, parents):
            allowed = self._profile_allows(parents)
            if allowed:
                self.write_sub_class_of(class_name, allowed)

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
                f"Got inverse_name={inverse_name!r} without create_inverse=True. "
                "Set create_inverse=True, or drop inverse_name."
            )
        if axiomatic_inverse and not create_inverse:
            raise ValueError(
                "Got axiomatic_inverse=True without create_inverse=True. Set "
                "create_inverse=True, or drop axiomatic_inverse."
            )
        parts = self._disjuncts(range_class, "a property range")
        if parts is None:
            return
        if data_prop and len(parts) > 1:
            raise ValueError(
                f"Range {range_class!r} is a list. A data property range is one "
                f"datatype out of {sorted(OWL_DATATYPES)}."
            )
        if data_prop and parts[0] not in OWL_DATATYPES:
            raise ValueError(
                f"Range {range_class!r} is not a datatype. A data property range "
                f"is one of {sorted(OWL_DATATYPES)}."
            )
        if not data_prop and set(parts) & OWL_DATATYPES:
            raise ValueError(
                f"Range {range_class!r} is a datatype. Set data_prop=True, or "
                "name a class."
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
        domain_fallback: str | None = None,
        range_fallback: str | None = None,
        source: str = "a build step",
    ) -> None:
        """Declare a property, with its domain, range and optional inverse.

        Args:
            prop_name: Local name of the property.
            domain_class: Class or classes the subject may belong to.
            range_class: Class or classes the target may belong to, or one
                datatype for a data property.
            superprop: Property to place this one below. The inverse is placed
                below the inverse of that property.
            create_inverse: Also declare the inverse property.
            inverse_name: Name for the inverse, instead of the derived one.
            data_prop: Whether this is a data property.
            axiomatic_inverse: Also state that the two are inverses of each
                other.
            domain_fallback: Class that replaces a union domain under OWL 2
                EL. Without one the domain is dropped.
            range_fallback: Class that replaces a union range under OWL 2 EL.
                Without one the range is dropped.
            source: Origin of the property.

        Raises:
            ValueError: If the arguments contradict each other, such as an
                inverse for a data property or a datatype range on an object
                property.
            UnsafeLocalNameError: If a name cannot appear in an IRI, or a
                property name does not start with a letter or an underscore.
            NameCollisionError: If the name is declared as another kind, or
                the inverse name is the inverse of another property.
        """
        self._validate_property_args(
            range_class, data_prop, create_inverse, inverse_name, axiomatic_inverse
        )
        kind = SinkEntity.property_for(data_prop)
        self._declare_name(prop_name, kind, source)

        if self._names.first_declaration(prop_name, kind):
            self.write_declaration(prop_name, kind)
        if superprop is not None:
            self.add_sub_property(prop_name, superprop, data_prop, source=source)
        domain = self._disjuncts(domain_class, f"the domain of {prop_name!r}")
        if domain and self._names.first_domain(prop_name, domain, data_prop):
            allowed = self._profile_allows(domain, domain_fallback)
            if allowed:
                self.write_domain(prop_name, allowed, data_prop)
        range_ = self._disjuncts(range_class, f"the range of {prop_name!r}")
        if range_ and self._names.first_range(prop_name, range_, data_prop):
            if data_prop:
                self.write_data_range(prop_name, range_[0])
            else:
                allowed = self._profile_allows(range_, range_fallback)
                if allowed:
                    self.write_range(prop_name, allowed)

        if create_inverse:
            self._declare_inverse(
                prop_name, inverse_name, kind, superprop, source, axiomatic_inverse
            )

    def _declare_inverse(
        self,
        prop_name: str,
        inverse_name: str | None,
        kind: SinkEntity,
        superprop: str | None,
        source: str,
        axiomatic: bool,
    ) -> None:
        """Declare the inverse of a property, and state the inverse axiom when `axiomatic` is set."""
        if inverse_name is None:
            inverse_name = self.inverse_name(prop_name)
        inverse_source = f"inverse of {source}"
        self._names.claim_inverse(prop_name, inverse_name, source)
        self._declare_name(inverse_name, kind, inverse_source)
        if self._names.first_declaration(inverse_name, kind):
            self.write_declaration(inverse_name, kind)
        if axiomatic and self._names.first_inverse(prop_name, inverse_name):
            self.write_inverse_properties(prop_name, inverse_name)
        if superprop is not None:
            parent_inverse = self.inverse_name(superprop)
            if not self.is_declared(parent_inverse):
                self._declare_name(parent_inverse, kind, inverse_source)
                if self._names.first_declaration(parent_inverse, kind):
                    self.write_declaration(parent_inverse, kind)
            self.add_sub_property(
                inverse_name,
                parent_inverse,
                kind is SinkEntity.DATA_PROPERTY,
                source=inverse_source,
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
        if self._names.first_sub_property(sub, parent, data_prop):
            self.write_sub_property_of(sub, parent, data_prop)

    def add_transitive(self, prop: str) -> None:
        """Declare an object property transitive."""
        self._check_name(prop, "a transitivity axiom")
        if self._names.first_transitive(prop):
            self.write_transitive(prop)

    def add_disjoint_classes(
        self, classes: Iterable[str], *, source: str = "a disjointness axiom"
    ) -> None:
        """State that the classes share no individual.

        Repeated names count once, and fewer than two classes write nothing.

        Raises:
            UnsafeLocalNameError: If a name cannot appear in an IRI.
        """
        distinct = list(dict.fromkeys(classes))
        if len(distinct) < 2:
            return
        for name in distinct:
            self._check_name(name, source)
        if self._names.first_disjoint(distinct):
            self.write_disjoint_classes(distinct)

    def assert_individuals(self, class_name: str, individuals: Iterable[Any]) -> None:
        """Assert that each individual belongs to a class.

        Duplicates and missing values are dropped.

        Raises:
            UnsafeLocalNameError: If a name cannot appear in an IRI.
            NameCollisionError: If a name is declared as a class or property.
        """
        names = self._names.check_individuals(
            individuals, f"an individual of class {class_name!r}"
        )
        if names:
            self.write_class_assertions(class_name, names)

    def assert_relations(self, prop: str, pairs: Iterable[tuple[Any, Any]]) -> None:
        """Assert an object property between each pair of individuals.

        Every name of every pair is checked.

        Args:
            prop: Property to assert.
            pairs: Subject and target of each assertion.

        Raises:
            UnsafeLocalNameError: If a name cannot appear in an IRI.
            NameCollisionError: If a name is declared as a class or property.
        """
        self.write_object_assertions(prop, self._checked_pairs(prop, pairs))

    def assert_data(self, prop: str, pairs: Iterable[tuple[Any, Any]]) -> None:
        """Assert a data property between each individual and its value.

        A pair whose value is missing is dropped.

        Args:
            prop: Property to assert.
            pairs: Subject and literal value of each assertion.

        Raises:
            InputError: If a value is infinite.
            UnsafeLocalNameError: If a subject cannot appear in an IRI.
            NameCollisionError: If a subject is declared as a class or
                property.
        """
        self.write_data_assertions(prop, self._checked_subjects(prop, pairs))

    def _checked_pairs(
        self, prop: str, pairs: Iterable[tuple[Any, Any]]
    ) -> Iterator[tuple[str, str]]:
        """Yield each pair as text, with both names checked."""
        accepts = self._names.accepts_individual
        for subject, target in pairs:
            name, other = str(subject), str(target)
            if not accepts(name):
                self._names.check_individual(name, f"the subject of {prop!r}")
            if not accepts(other):
                self._names.check_individual(other, f"the target of {prop!r}")
            yield name, other

    def _checked_subjects(
        self, prop: str, pairs: Iterable[tuple[Any, Any]]
    ) -> Iterator[tuple[str, Literal]]:
        """Yield each pair with the subject as checked text and the value resolved."""
        accepts = self._names.accepts_individual
        for subject, value in pairs:
            literal = self._resolve(value)
            if literal is None:
                continue
            name = str(subject)
            if not accepts(name):
                self._names.check_individual(name, f"the subject of {prop!r}")
            yield name, literal

    def annotate_entity(
        self, iri_fragment: str, label: str | None = None, comment: str | None = None
    ) -> None:
        """Attach a label and a comment to a declared class or property.

        Args:
            iri_fragment: Local name of the entity.
            label: Text for `rdfs:label`.
            comment: Text for `rdfs:comment`.

        Raises:
            ValueError: If the entity is not declared.
        """
        if not self.is_declared(iri_fragment):
            raise ValueError(
                f"Cannot annotate {iri_fragment!r}: no build step declared it as a "
                "class or a property. Declare it first, or filter the names with "
                "is_declared()."
            )
        if label and self._names.first_annotation("label", iri_fragment, label):
            self.write_annotations("label", [(iri_fragment, label)])
        if comment and self._names.first_annotation("comment", iri_fragment, comment):
            self.write_annotations("comment", [(iri_fragment, comment)])

    def annotate_individuals_bulk(self, pairs: list[tuple[str, str]]) -> None:
        """Label many individuals at once.

        Each name is checked as an individual name.

        Raises:
            UnsafeLocalNameError: If a name cannot appear in an IRI.
            NameCollisionError: If a name is declared as a class or property.
        """
        accepts = self._names.accepts_individual
        for name, _ in pairs:
            if not accepts(name):
                self._names.check_individual(name, "a labeled individual")
        if pairs:
            self.write_annotations("label", pairs)

    def inverse_name(self, prop: str) -> str:
        """Return the inverse name of a property, from the vocabulary or from this build."""
        return self._names.inverse_name(prop)
