"""The builder facade over an :class:`~semanticlog.onto.sink.AxiomSink`.

Axiom construction itself lives in :mod:`semanticlog.onto.sink`. What remains
here is the state a build carries around it: which sink to write to, the
settings, the annotation writer, the annotation document, and the guarantee that
one builder produces one ontology.

The emission methods are kept as delegating wrappers so that the builder still
satisfies the ``AnnotationSink`` protocol and so that code holding a builder can
reach the sink without knowing it exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sink import AxiomSink, InMemorySink

if TYPE_CHECKING:
    from owlapy.owl_ontology import Ontology


class OCEDBaseBuilder:
    """Carries the sink, settings and annotation state shared by every build step.

    Subclassed by :class:`~semanticlog.onto.upper.OCEDUpperOntoBuilder` (TBox)
    and :class:`~semanticlog.onto.instantiate.OCEDOntoBuilder` (full pipeline).
    """

    def __init__(
        self,
        iri: str,
        settings: OntoBuildSettings,
        load: bool = False,
        sink: AxiomSink | None = None,
    ):
        """Create a builder over a new or existing ontology.

        Args:
            iri: Ontology IRI, or a resolvable document IRI when ``load`` is True.
                A trailing ``#`` is appended if missing.
            settings: Build configuration.
            load: Read an existing ontology from ``iri`` instead of starting empty.
                Off by default: resolving a non-document IRI makes owlapy attempt a
                network fetch.
            sink: Where axioms go. Defaults to an in-memory owlapy ontology.
                ``iri`` and ``load`` are ignored when a sink is supplied, since the
                sink already owns them.
        """
        self.sink: AxiomSink = (
            sink if sink is not None else InMemorySink(iri, settings, load)
        )
        self.iri = self.sink.iri
        self.settings = settings
        self._annotation_doc: dict[str, dict[str, str]] = settings.annotation_data or {}
        self.annotations = AnnotationWriter(self, settings)
        self._claimed = False

    @property
    def onto(self) -> Ontology:
        """The ontology being built.

        Only meaningful for a sink that keeps one in memory. A sink that streams
        straight to disk has no ontology object to hand back.
        """
        onto = getattr(self.sink, "onto", None)
        if onto is None:
            raise AttributeError(
                f"{type(self.sink).__name__} does not keep an ontology in memory."
            )
        return onto

    def _claim_single_use(self) -> None:
        """Reserve this builder for one build.

        Axioms accumulate in a single ontology object, so a second build on the
        same instance merges two unrelated logs into one file without any error.
        """
        if self._claimed:
            raise ValueError(
                "This builder has already produced an ontology. Axioms accumulate "
                "in one ontology object, so building again would merge the two "
                "logs. Construct a new builder for each build."
            )
        self._claimed = True

    def apply_annotation_document(self) -> None:
        """Annotate all entities listed in the annotation document that exist here."""
        for name, ann in self._annotation_doc.items():
            self.annotate_entity(
                name, label=ann.get("label"), comment=ann.get("comment")
            )

    # ------------------------------------------------------------------
    # Delegation to the sink
    # ------------------------------------------------------------------

    def annotate_entity(
        self,
        iri_fragment: str,
        label: str | None = None,
        comment: str | None = None,
    ) -> None:
        """Add rdfs:label and/or rdfs:comment to a class or property."""
        self.sink.annotate_entity(iri_fragment, label, comment)

    def annotate_individuals_bulk(self, pairs: list[tuple[str, str]]) -> None:
        """Emit rdfs:label for each (iri_fragment, label) pair in one batch."""
        self.sink.annotate_individuals_bulk(pairs)

    def add_classes(
        self, class_list: list[str], superclasses: list[str] | str | None = None
    ) -> None:
        """Add multiple OWL classes, optionally with a shared or per-class superclass."""
        self.sink.add_classes(class_list, superclasses)

    def add_class(self, class_name: str, superclass: str | None = None) -> None:
        """Declare a single OWL class, optionally as a subclass of another."""
        self.sink.add_class(class_name, superclass)

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
        """Add a property with optional domain, range, super-property, and inverse."""
        self.sink.add_property(
            prop_name,
            domain_class,
            range_class,
            superprop,
            create_inverse,
            inverse_name,
            data_prop,
        )

    def assert_individual(self, individual: str, class_name: str) -> None:
        """Assert that a single named individual belongs to a class."""
        self.sink.assert_individual(individual, class_name)

    def assert_individuals(self, individuals: list[str], class_name: str) -> None:
        """Assert that a list of named individuals all belong to a class."""
        self.sink.assert_individuals(individuals, class_name)

    def assert_relation(
        self,
        prop_name: str,
        from_name: str,
        to: str | int | bool | float,
        data_prop: bool = False,
        inverse_name: str | None = None,
    ) -> None:
        """Assert a single property between two individuals or a literal."""
        self.sink.assert_relation(prop_name, from_name, to, data_prop, inverse_name)

    def assert_relations(
        self,
        df: pd.DataFrame,
        prop: str,
        from_key: str,
        to_key: str,
        data_prop: bool = False,
        inverse_name: str | None = None,
    ) -> None:
        """Assert property relations from a DataFrame, dropping nulls and duplicates."""
        self.sink.assert_relations(df, prop, from_key, to_key, data_prop, inverse_name)

    def generate_inverse_property_name(self, prop: str) -> str:
        """Return the inverse property name for ``prop``."""
        return self.sink.inverse_name(prop)

    def save_onto(self, path: str) -> None:
        """Save the current ontology to an OWL file."""
        self.sink.save(path)
