"""The one place semlog talks to an OWL representation.

Everything that turns a name into an axiom goes through :class:`AxiomSink`. The
emitters above it deal in strings and DataFrames and never import owlapy, which
is what lets one build target different backends.

:class:`AxiomSink` owns the *policy*: what a property with an inverse expands to,
which rows of a frame are worth asserting, how duplicates are dropped. Subclasses
supply only the writes. Keeping the split there means a new backend cannot
quietly disagree with the existing one about what a build emits.

Two backends live here:

- :class:`InMemorySink` builds an owlapy ``Ontology``. Needed for reasoning and
  inspection; holds every axiom until :meth:`~AxiomSink.save`.
- :class:`StreamingSink` writes OWL functional syntax straight to disk at
  near-constant memory, for logs too large to hold.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from typing import Any, TextIO

import pandas as pd
from owlapy.iri import IRI
from owlapy.owl_axiom import (
    OWLAnnotation,
    OWLAnnotationAssertionAxiom,
    OWLAnnotationProperty,
    OWLClassAssertionAxiom,
    OWLDataPropertyAssertionAxiom,
    OWLDataPropertyDomainAxiom,
    OWLDataPropertyRangeAxiom,
    OWLDeclarationAxiom,
    OWLInverseObjectPropertiesAxiom,
    OWLObjectPropertyAssertionAxiom,
    OWLObjectPropertyDomainAxiom,
    OWLObjectPropertyRangeAxiom,
    OWLSubClassOfAxiom,
    OWLSubDataPropertyOfAxiom,
    OWLSubObjectPropertyOfAxiom,
    OWLTransitiveObjectPropertyAxiom,
)
from owlapy.owl_datatype import OWLDatatype
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_literal import OWLLiteral
from owlapy.owl_ontology import Ontology
from owlapy.owl_property import OWLDataProperty, OWLObjectProperty

from semanticlog.onto import vocabulary as vocab
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.transform import parse_manchester

_OWL_DATATYPES: frozenset[str] = frozenset({"integer", "double", "boolean", "dateTime"})

_XSD = "http://www.w3.org/2001/XMLSchema#"
_RDFS = "http://www.w3.org/2000/01/rdf-schema#"

_RDFS_LABEL = OWLAnnotationProperty(IRI(_RDFS, "label"))
_RDFS_COMMENT = OWLAnnotationProperty(IRI(_RDFS, "comment"))

#: The class-expression operators the builders actually produce. Anything richer
#: has to be added to every backend deliberately rather than silently mistranslated.
_UNION_SEPARATOR = " or "


def split_union(expression: str) -> list[str]:
    """Split a class expression into its disjuncts.

    The builders only ever produce a bare class name or names joined by ``" or "``.

    Raises:
        NotImplementedError: If the expression uses any richer Manchester syntax,
            which a backend would otherwise have to guess at.
    """
    parts = [p.strip() for p in expression.split(_UNION_SEPARATOR)]
    for part in parts:
        if not part or any(c in part for c in "()[]{},") or " " in part:
            raise NotImplementedError(
                f"Unsupported class expression {expression!r}: this sink handles a "
                "class name or names joined by ' or '."
            )
    return parts


class AxiomSink(ABC):
    """Where a build sends its axioms.

    Callers pass IRI local names, never IRIs or owlapy objects. Subclasses
    implement the ``write_*`` primitives; everything else here is shared policy.
    """

    iri: str
    settings: OntoBuildSettings

    def __init__(self, iri: str, settings: OntoBuildSettings) -> None:
        self.iri = iri if iri.endswith("#") else iri + "#"
        self.settings = settings
        self._named_inverse_map: dict[str, str] = dict(vocab.SCHEMA_NAMED_INVERSES)

    # ------------------------------------------------------------------
    # Write primitives, one per axiom shape
    # ------------------------------------------------------------------

    @abstractmethod
    def write_declaration(self, name: str, kind: str) -> None:
        """Declare an entity. ``kind`` is ``class``, ``object`` or ``data``."""

    @abstractmethod
    def write_sub_class_of(self, sub: str, superclass: str) -> None:
        """Assert that one named class is a subclass of a class expression."""

    @abstractmethod
    def write_sub_property_of(self, sub: str, parent: str, data_prop: bool) -> None:
        """Assert that one property is a sub-property of another."""

    @abstractmethod
    def write_domain(self, prop: str, expression: str, data_prop: bool) -> None:
        """Assert a property's domain."""

    @abstractmethod
    def write_range(self, prop: str, expression: str, data_prop: bool) -> None:
        """Assert a property's range, which may be an XSD datatype name."""

    @abstractmethod
    def write_inverse_properties(self, prop: str, inverse: str) -> None:
        """Assert that two object properties are inverses of each other."""

    @abstractmethod
    def write_transitive(self, prop: str) -> None:
        """Assert that an object property is transitive."""

    @abstractmethod
    def write_class_assertions(
        self, class_expression: str, individuals: Iterable[str]
    ) -> None:
        """Assert class membership for a run of individuals."""

    @abstractmethod
    def write_object_assertions(
        self, prop: str, pairs: Iterable[tuple[str, str]]
    ) -> None:
        """Assert an object property for a run of subject/object pairs."""

    @abstractmethod
    def write_data_assertions(
        self, prop: str, pairs: Iterable[tuple[str, Any]]
    ) -> None:
        """Assert a data property for a run of subject/value pairs."""

    @abstractmethod
    def write_annotations(
        self, annotation: str, pairs: Iterable[tuple[str, str]]
    ) -> None:
        """Attach ``label`` or ``comment`` to a run of subjects."""

    @abstractmethod
    def write_ontology_annotation(self, property_iri: str, value: str) -> None:
        """Attach metadata to the ontology itself.

        The subject is the ontology IRI rather than a local name, which is why
        this is separate from :meth:`write_annotations`.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Write the ontology to disk."""

    def close(self) -> None:
        """Release anything the sink is holding open. Safe to call twice."""

    # ------------------------------------------------------------------
    # Shared policy
    # ------------------------------------------------------------------

    def add_class(self, class_name: str, superclass: str | None = None) -> None:
        """Declare an OWL class, optionally as a subclass of another."""
        logging.info(f"  adding class {class_name} as subclass of {superclass}..")
        self.write_declaration(class_name, "class")
        if superclass is not None:
            self.write_sub_class_of(class_name, superclass)

    def add_classes(
        self, class_list: list[str], superclasses: list[str] | str | None = None
    ) -> None:
        """Declare several classes, with a shared or per-class superclass.

        Args:
            class_list: Names of the classes to declare.
            superclasses: One superclass name applied to all, a list of the same
                length as ``class_list``, or ``None`` for root classes.

        Raises:
            ValueError: If ``superclasses`` is a list of a different length.
        """
        logging.info(f"  adding {len(class_list)} classes..")
        if isinstance(superclasses, str) or superclasses is None:
            for class_name in class_list:
                self.add_class(class_name, superclasses)
        elif isinstance(superclasses, list) and len(class_list) == len(superclasses):
            for class_name, superclass in zip(class_list, superclasses):
                self.add_class(class_name, superclass)
        else:
            raise ValueError(
                "superclasses must be a list of str with the same length as "
                "class_list, a str, or None"
            )

    def _validate_property_args(
        self,
        range_class: str | None,
        data_prop: bool,
        create_inverse: bool,
        inverse_name: str | None,
    ) -> None:
        if data_prop and create_inverse:
            raise ValueError(
                "Data properties cannot have inverses; set create_inverse to False for data properties."
            )
        if inverse_name is not None and not create_inverse:
            raise ValueError(
                "When adding an inverse via inverse_name create_inverse must be set to true!"
            )
        if data_prop and range_class is not None and range_class not in _OWL_DATATYPES:
            raise ValueError(f"Unknown range for a data property: {range_class}")
        if not data_prop and range_class in _OWL_DATATYPES:
            raise ValueError("Can not create an object property with a data as range!")

    def add_property(
        self,
        prop_name: str,
        domain_class: str | None = None,
        range_class: str | None = None,
        superprop: str | None = None,
        create_inverse: bool = False,
        inverse_name: str | None = None,
        data_prop: bool = False,
    ) -> None:
        """Declare a property with optional domain, range, parent and inverse.

        Raises:
            ValueError: For invalid combinations (data property with inverse,
                unknown data-property range, object property with datatype range).
        """
        logging.info(f"  adding property {prop_name} ..")
        self._validate_property_args(
            range_class, data_prop, create_inverse, inverse_name
        )
        kind = "data" if data_prop else "object"

        self.write_declaration(prop_name, kind)
        if superprop is not None:
            self.write_sub_property_of(prop_name, superprop, data_prop)
        if domain_class is not None:
            self.write_domain(prop_name, domain_class, data_prop)
        if range_class is not None:
            self.write_range(prop_name, range_class, data_prop)

        if not create_inverse:
            return

        if inverse_name is None:
            inverse_name = self.inverse_name(prop_name)
        self._named_inverse_map[prop_name] = inverse_name
        self._named_inverse_map[inverse_name] = prop_name
        self.write_declaration(inverse_name, kind)
        if self.settings.axiomatic_inverses:
            self.write_inverse_properties(prop_name, inverse_name)
        if superprop is not None:
            self.write_sub_property_of(
                inverse_name, self.inverse_name(superprop), data_prop
            )

    def add_sub_property(self, sub: str, parent: str) -> None:
        """Assert that one object property is a sub-property of another."""
        self.write_sub_property_of(sub, parent, False)

    def make_transitive(self, prop: str) -> None:
        """Assert that an object property is transitive."""
        self.write_transitive(prop)

    def assert_individual(self, individual: str, class_name: str) -> None:
        """Assert that a named individual belongs to a class."""
        self.write_class_assertions(class_name, [individual])

    def assert_individuals(self, individuals: Sequence[str], class_name: str) -> None:
        """Assert that several named individuals belong to a class.

        Duplicates are removed, preserving first-seen order.
        """
        # dict.fromkeys preserves insertion order while deduplicating
        unique = list(dict.fromkeys(individuals))
        logging.info(f"  asserting {len(unique)} individuals of class {class_name}..")
        if unique:
            self.write_class_assertions(class_name, unique)

    def assert_relation(
        self,
        prop_name: str,
        from_name: str,
        to: str | int | bool | float,
        data_prop: bool = False,
        inverse_name: str | None = None,
    ) -> None:
        """Assert one property between two individuals, or an individual and a literal.

        Raises:
            ValueError: For type mismatches between ``to`` and ``data_prop``.
        """
        if not isinstance(to, str) and not data_prop:
            raise ValueError(
                "When creating an object property the type of the to must be str."
            )
        if isinstance(to, str) and data_prop:
            raise ValueError(
                "When creating an data property the type of the to must not be str."
            )

        if data_prop:
            self.write_data_assertions(prop_name, [(from_name, to)])
            return

        self.write_object_assertions(prop_name, [(from_name, str(to))])
        if self.settings.implicit_inverses:
            inv = inverse_name or self.inverse_name(prop_name)
            self.write_object_assertions(inv, [(str(to), from_name)])

    def assert_relations(
        self,
        df: pd.DataFrame,
        prop: str,
        from_key: str,
        to_key: str,
        data_prop: bool = False,
        inverse_name: str | None = None,
    ) -> None:
        """Assert a property for every row of a DataFrame, dropping nulls and duplicates.

        Args:
            df: Frame containing at least ``from_key`` and ``to_key``.
            prop: Property name to assert.
            from_key: Column holding the subject local names.
            to_key: Column holding object local names, or literal values.
            data_prop: Assert as a data property rather than an object property.
            inverse_name: Explicit inverse name for the implicit-inverse pass.
        """
        rows = df[[from_key, to_key]]
        rows = rows[rows[from_key].notna() & rows[to_key].notna()]
        rows = rows.drop_duplicates([from_key, to_key])

        if data_prop:
            logging.info(f"  asserting {len(rows)} relations of data property {prop}..")
            self.write_data_assertions(prop, rows.itertuples(index=False, name=None))
            return

        logging.info(f"  asserting {len(rows)} relations of object property {prop}..")
        pairs = [(str(s), str(o)) for s, o in rows.itertuples(index=False, name=None)]
        if pairs:
            self.write_object_assertions(prop, pairs)
        if self.settings.implicit_inverses and pairs:
            inv = inverse_name or self.inverse_name(prop)
            self.write_object_assertions(inv, [(o, s) for s, o in pairs])

    def annotate_entity(
        self, iri_fragment: str, label: str | None = None, comment: str | None = None
    ) -> None:
        """Attach ``rdfs:label`` and/or ``rdfs:comment`` to an entity."""
        if label:
            self.write_annotations("label", [(iri_fragment, label)])
        if comment:
            self.write_annotations("comment", [(iri_fragment, comment)])

    def annotate_individuals_bulk(self, pairs: list[tuple[str, str]]) -> None:
        """Attach ``rdfs:label`` to each ``(local name, label)`` pair."""
        if pairs:
            self.write_annotations("label", pairs)

    def inverse_name(self, prop: str) -> str:
        """Return the inverse property name for ``prop``.

        Consults the named-inverse map (seeded from the upper ontology and
        extended during :meth:`add_property`) before falling back to the
        generated ``inv_{prop}`` form.
        """
        return vocab.inverse(prop, self._named_inverse_map)


def read_named_inverses(onto: Ontology) -> dict[str, str]:
    """Return the named inverse pairs declared in an ontology loaded from disk.

    The only owlapy-private access in this package.  owlapy 1.6 exposes no public
    equivalent: ``object_properties_in_signature()`` yields ``OWLObjectProperty``
    objects whose ``get_inverse_property()`` returns a syntactic ``OWLObjectInverseOf``
    rather than the named inverse, and ``get_tbox_axioms()`` raises
    ``NotImplementedError``.  Revisit when owlapy grows a public accessor.
    """
    pairs: dict[str, str] = {}
    for prop in onto._onto.object_properties():
        if prop.inverse_property is not None:
            pairs[prop.name] = prop.inverse_property.name
    return pairs


class InMemorySink(AxiomSink):
    """Builds an owlapy ``Ontology`` in memory.

    Holds every axiom until :meth:`save`, so peak memory scales with the size of
    the ontology rather than the size of the input. Use :class:`StreamingSink`
    when that does not fit.
    """

    def __init__(self, iri: str, settings: OntoBuildSettings, load: bool = False):
        """Open a new or existing ontology.

        Args:
            iri: Ontology IRI, or a resolvable document IRI when ``load`` is True.
                A trailing ``#`` is appended if missing.
            settings: Build configuration; the inverse strategy is read from it.
            load: Read an existing ontology from ``iri`` instead of starting empty.
                Off by default: resolving a non-document IRI makes owlapy attempt
                a network fetch.
        """
        onto, resolved = self._open_ontology(iri, load)
        super().__init__(resolved, settings)
        self.onto = onto
        if load:
            self._named_inverse_map.update(read_named_inverses(self.onto))

    @staticmethod
    def _open_ontology(iri: str, load: bool) -> tuple[Ontology, str]:
        if not load:
            return Ontology(IRI.create(iri), load=False), iri
        try:
            onto = Ontology(IRI.create(iri), load=True)
            return onto, onto.get_ontology_id().get_ontology_iri().str  # type: ignore[union-attr]
        except Exception as e:
            # Not resolvable (file missing, no network, unsupported kwarg) — start empty.
            logging.warning(
                f"Could not load ontology from IRI '{iri}': {e}. Creating empty ontology."
            )
            return Ontology(IRI.create(iri), load=False), iri

    def _entity(self, name: str, kind: str) -> Any:
        if kind == "class":
            return parse_manchester(self.iri, name)
        if kind == "data":
            return OWLDataProperty(self.iri + name)
        return OWLObjectProperty(self.iri + name)

    def write_declaration(self, name: str, kind: str) -> None:
        self.onto.add_axiom(OWLDeclarationAxiom(self._entity(name, kind)))

    def write_sub_class_of(self, sub: str, superclass: str) -> None:
        self.onto.add_axiom(
            OWLSubClassOfAxiom(
                parse_manchester(self.iri, sub), parse_manchester(self.iri, superclass)
            )
        )

    def write_sub_property_of(self, sub: str, parent: str, data_prop: bool) -> None:
        axiom = OWLSubDataPropertyOfAxiom if data_prop else OWLSubObjectPropertyOfAxiom
        kind = "data" if data_prop else "object"
        self.onto.add_axiom(axiom(self._entity(sub, kind), self._entity(parent, kind)))

    def write_domain(self, prop: str, expression: str, data_prop: bool) -> None:
        axiom = (
            OWLDataPropertyDomainAxiom if data_prop else OWLObjectPropertyDomainAxiom
        )
        kind = "data" if data_prop else "object"
        self.onto.add_axiom(
            axiom(self._entity(prop, kind), parse_manchester(self.iri, expression))
        )

    def write_range(self, prop: str, expression: str, data_prop: bool) -> None:
        axiom = OWLDataPropertyRangeAxiom if data_prop else OWLObjectPropertyRangeAxiom
        kind = "data" if data_prop else "object"
        if expression in _OWL_DATATYPES:
            owl_range: Any = OWLDatatype(IRI(_XSD, expression))
        else:
            owl_range = parse_manchester(self.iri, expression)
        self.onto.add_axiom(axiom(self._entity(prop, kind), owl_range))

    def write_inverse_properties(self, prop: str, inverse: str) -> None:
        self.onto.add_axiom(
            OWLInverseObjectPropertiesAxiom(
                OWLObjectProperty(self.iri + prop),
                OWLObjectProperty(self.iri + inverse),
            )
        )

    def write_transitive(self, prop: str) -> None:
        self.onto.add_axiom(
            OWLTransitiveObjectPropertyAxiom(OWLObjectProperty(self.iri + prop))
        )

    def write_class_assertions(
        self, class_expression: str, individuals: Iterable[str]
    ) -> None:
        owl_class = parse_manchester(self.iri, class_expression)
        iri = self.iri
        axioms = [
            OWLClassAssertionAxiom(OWLNamedIndividual(f"{iri}{individual}"), owl_class)
            for individual in individuals
        ]
        if axioms:
            self.onto.add_axiom(axioms)

    def write_object_assertions(
        self, prop: str, pairs: Iterable[tuple[str, str]]
    ) -> None:
        iri = self.iri
        owl_prop = OWLObjectProperty(f"{iri}{prop}")
        axioms = [
            OWLObjectPropertyAssertionAxiom(
                OWLNamedIndividual(f"{iri}{s}"),
                owl_prop,
                OWLNamedIndividual(f"{iri}{o}"),
            )
            for s, o in pairs
        ]
        if axioms:
            self.onto.add_axiom(axioms)

    def write_data_assertions(
        self, prop: str, pairs: Iterable[tuple[str, Any]]
    ) -> None:
        iri = self.iri
        owl_prop = OWLDataProperty(f"{iri}{prop}")
        axioms = []
        for subject, value in pairs:
            literal = self._literal(value)
            if literal is None:
                continue
            axioms.append(
                OWLDataPropertyAssertionAxiom(
                    OWLNamedIndividual(f"{iri}{subject}"), owl_prop, literal
                )
            )
        if axioms:
            self.onto.add_axiom(axioms)

    @staticmethod
    def _literal(value: Any) -> Any:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, pd.Timestamp) or hasattr(value, "to_pydatetime"):
            if pd.isna(value):
                return None
            return OWLLiteral(value.to_pydatetime())  # type: ignore[abstract]
        return OWLLiteral(value)  # type: ignore[abstract]

    def write_annotations(
        self, annotation: str, pairs: Iterable[tuple[str, str]]
    ) -> None:
        prop = _RDFS_LABEL if annotation == "label" else _RDFS_COMMENT
        axioms = []
        for name, text in pairs:
            axioms.append(
                OWLAnnotationAssertionAxiom(
                    IRI.create(f"{self.iri}{name}"),
                    OWLAnnotation(prop, OWLLiteral(text)),  # type: ignore[abstract]
                )
            )
        if not axioms:
            return
        try:
            self.onto.add_axiom(axioms)
        except AssertionError:
            # owlready2 rejects annotations on entities it has not seen.
            logging.warning(
                f"Skipping {len(axioms)} {annotation} annotation(s): "
                "entity not found in ontology."
            )

    def write_ontology_annotation(self, property_iri: str, value: str) -> None:
        namespace, _, local = property_iri.rpartition("#")
        self.onto.add_axiom(
            OWLAnnotationAssertionAxiom(
                IRI.create(self.iri.rstrip("#")),
                OWLAnnotation(
                    OWLAnnotationProperty(IRI(namespace + "#", local)),
                    OWLLiteral(value),  # type: ignore[abstract]
                ),
            )
        )

    def save(self, path: str) -> None:
        self.onto.save(path)


class StreamingSink(AxiomSink):
    """Writes OWL functional syntax straight to disk.

    Functional syntax is a flat list of axioms, so nothing has to be buffered and
    peak memory stays flat in the size of the log. The trade-off is that there is
    no ontology object afterwards: the result can only be read back from the file.

    Unlike the owlapy-backed sink this one owns its own literal serialisation, so
    ``xsd:dateTime`` values survive (owlapy 1.6 cannot map them).

    Use as a context manager, or call :meth:`close` when done.
    """

    def __init__(self, path: str, iri: str, settings: OntoBuildSettings):
        """Open ``path`` and write the ontology header.

        Args:
            path: Destination ``.ofn`` file.
            iri: Ontology IRI. A trailing ``#`` is appended if missing.
            settings: Build configuration; the inverse strategy is read from it.
        """
        super().__init__(iri, settings)
        self.path = path
        self._file: TextIO | None = open(path, "w", encoding="utf-8")
        self.axiom_count = 0
        self._write_header()

    def __enter__(self) -> StreamingSink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def _out(self) -> TextIO:
        if self._file is None:
            raise ValueError(f"StreamingSink for {self.path} is already closed.")
        return self._file

    def _write_header(self) -> None:
        base = self.iri.rstrip("#")
        self._out.write(
            f"Prefix(:=<{self.iri}>)\n"
            f"Prefix(owl:=<http://www.w3.org/2002/07/owl#>)\n"
            f"Prefix(rdf:=<http://www.w3.org/1999/02/22-rdf-syntax-ns#>)\n"
            f"Prefix(rdfs:=<{_RDFS}>)\n"
            f"Prefix(xsd:=<{_XSD}>)\n\n"
            f"Ontology(<{base}>\n"
        )

    def _axiom(self, text: str) -> None:
        self._out.write(text)
        self._out.write("\n")
        self.axiom_count += 1

    @staticmethod
    def _ref(name: str) -> str:
        return f":{name}"

    def _class_expression(self, expression: str) -> str:
        parts = split_union(expression)
        if len(parts) == 1:
            return self._ref(parts[0])
        return "ObjectUnionOf(" + " ".join(self._ref(p) for p in parts) + ")"

    _DECLARATION_KIND = {
        "class": "Class",
        "object": "ObjectProperty",
        "data": "DataProperty",
    }

    def write_declaration(self, name: str, kind: str) -> None:
        self._axiom(f"Declaration({self._DECLARATION_KIND[kind]}({self._ref(name)}))")

    def write_sub_class_of(self, sub: str, superclass: str) -> None:
        self._axiom(
            f"SubClassOf({self._class_expression(sub)} "
            f"{self._class_expression(superclass)})"
        )

    def write_sub_property_of(self, sub: str, parent: str, data_prop: bool) -> None:
        keyword = "SubDataPropertyOf" if data_prop else "SubObjectPropertyOf"
        self._axiom(f"{keyword}({self._ref(sub)} {self._ref(parent)})")

    def write_domain(self, prop: str, expression: str, data_prop: bool) -> None:
        keyword = "DataPropertyDomain" if data_prop else "ObjectPropertyDomain"
        self._axiom(
            f"{keyword}({self._ref(prop)} {self._class_expression(expression)})"
        )

    def write_range(self, prop: str, expression: str, data_prop: bool) -> None:
        keyword = "DataPropertyRange" if data_prop else "ObjectPropertyRange"
        if expression in _OWL_DATATYPES:
            target = f"xsd:{expression}"
        else:
            target = self._class_expression(expression)
        self._axiom(f"{keyword}({self._ref(prop)} {target})")

    def write_inverse_properties(self, prop: str, inverse: str) -> None:
        self._axiom(f"InverseObjectProperties({self._ref(prop)} {self._ref(inverse)})")

    def write_transitive(self, prop: str) -> None:
        self._axiom(f"TransitiveObjectProperty({self._ref(prop)})")

    def write_class_assertions(
        self, class_expression: str, individuals: Iterable[str]
    ) -> None:
        expression = self._class_expression(class_expression)
        for individual in individuals:
            self._axiom(f"ClassAssertion({expression} {self._ref(str(individual))})")

    def write_object_assertions(
        self, prop: str, pairs: Iterable[tuple[str, str]]
    ) -> None:
        ref = self._ref(prop)
        for subject, obj in pairs:
            self._axiom(
                f"ObjectPropertyAssertion({ref} {self._ref(str(subject))} "
                f"{self._ref(str(obj))})"
            )

    def write_data_assertions(
        self, prop: str, pairs: Iterable[tuple[str, Any]]
    ) -> None:
        ref = self._ref(prop)
        for subject, value in pairs:
            literal = self._literal(value)
            if literal is None:
                continue
            self._axiom(
                f"DataPropertyAssertion({ref} {self._ref(str(subject))} {literal})"
            )

    @classmethod
    def _literal(cls, value: Any) -> str | None:
        """Render a Python value as a typed functional-syntax literal."""
        if value is None or value is pd.NaT:
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        if isinstance(value, bool):
            return f'"{str(value).lower()}"^^xsd:boolean'
        if isinstance(value, (pd.Timestamp, datetime)):
            if pd.isna(value):
                return None
            return f'"{value.isoformat()}"^^xsd:dateTime'
        if isinstance(value, date):
            return f'"{value.isoformat()}"^^xsd:date'
        if isinstance(value, (int,)):
            return f'"{value}"^^xsd:integer'
        if isinstance(value, float):
            return f'"{value}"^^xsd:double'
        return f'"{cls._escape(str(value))}"^^xsd:string'

    @staticmethod
    def _escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    def write_annotations(
        self, annotation: str, pairs: Iterable[tuple[str, str]]
    ) -> None:
        prop = f"rdfs:{annotation}"
        for name, text in pairs:
            self._axiom(
                f"AnnotationAssertion({prop} {self._ref(str(name))} "
                f'"{self._escape(str(text))}")'
            )

    def write_ontology_annotation(self, property_iri: str, value: str) -> None:
        self._axiom(
            f"AnnotationAssertion(<{property_iri}> <{self.iri.rstrip('#')}> "
            f'"{self._escape(value)}")'
        )

    def close(self) -> None:
        """Close the ontology block and the file. Safe to call twice."""
        if self._file is None:
            return
        self._file.write(")\n")
        self._file.close()
        self._file = None

    def save(self, path: str) -> None:
        """Finish the file this sink has been streaming to.

        Args:
            path: Must match the path the sink was opened on. A streaming sink
                holds no ontology, so it cannot write somewhere else.

        Raises:
            ValueError: If ``path`` differs from the path being streamed to.
        """
        if path != self.path:
            raise ValueError(
                f"StreamingSink is writing to {self.path!r} and cannot save to "
                f"{path!r}. Open it on the destination path instead."
            )
        self.close()
