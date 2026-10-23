"""The base class of every emitter."""

from __future__ import annotations

import pandas as pd

from semanticlog.core.enums import IndividualKind
from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.emitted import EmittedIndividuals
from semanticlog.onto.emitters.pairs import clean_pairs, reversed_pairs
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import AxiomSink


class Emitter:
    """Base for a build step that writes axioms.

    Attributes:
        sink: Where axioms go.
        settings: Settings of this build.
        annotations: Writer for the labels and comments that go with the
            axioms. It does nothing unless the build asked for annotations.
        emitted: The individuals the build has asserted, shared by every
            emitter.
    """

    def __init__(
        self,
        sink: AxiomSink,
        settings: OntoBuildSettings,
        annotations: AnnotationWriter,
        emitted: EmittedIndividuals,
    ) -> None:
        self.sink = sink
        self.settings = settings
        self.annotations = annotations
        self.emitted = emitted

    def _states_inverse_twins(self) -> bool:
        """Whether an inverse property needs its characteristic stated, having no inverse axiom."""
        return self.settings.has_inverses and not self.settings.axiomatic_inverses

    def declare_functional(self, prop: str) -> None:
        """Declare an object property functional, and its inverse inverse functional.

        The inverse gets its axiom only when no inverse axiom lets a reasoner
        derive it.
        """
        self.sink.add_functional(prop)
        if self._states_inverse_twins():
            self.sink.add_inverse_functional(self.sink.inverse_name(prop))

    def declare_inverse_functional(self, prop: str) -> None:
        """Declare an object property inverse functional, and its inverse functional.

        The inverse gets its axiom only when no inverse axiom lets a reasoner
        derive it.
        """
        self.sink.add_inverse_functional(prop)
        if self._states_inverse_twins():
            self.sink.add_functional(self.sink.inverse_name(prop))

    def assert_relations(
        self,
        prop: str,
        df: pd.DataFrame,
        from_key: str,
        to_key: str,
        kinds: tuple[IndividualKind, IndividualKind],
        inverse_name: str | None = None,
    ) -> None:
        """Assert an object property for every row of a table.

        With implicit inverses, the reversed pairs are also asserted under the
        inverse property.

        Args:
            prop: Property to assert.
            df: Table holding the pairs.
            from_key: Column holding the subject.
            to_key: Column holding the target.
            kinds: The kind of every subject and the kind of every target.
            inverse_name: Property for the reversed pairs. Defaults to
                `sink.inverse_name(prop)`.
        """
        if not self.settings.implicit_inverses:
            self.sink.assert_relations(prop, clean_pairs(df, from_key, to_key), kinds)
            return
        pairs = list(clean_pairs(df, from_key, to_key))
        self.sink.assert_relations(prop, pairs, kinds)
        inverse = inverse_name or self.sink.inverse_name(prop)
        self.sink.assert_relations(inverse, reversed_pairs(pairs), kinds[::-1])

    def assert_data(
        self,
        prop: str,
        df: pd.DataFrame,
        from_key: str,
        to_key: str,
        kind: IndividualKind,
    ) -> None:
        """Assert a data property for every row of a table.

        Args:
            prop: Property to assert.
            df: Table holding the individuals and their values.
            from_key: Column holding the subject.
            to_key: Column holding the literal value.
            kind: The kind of every subject.
        """
        self.sink.assert_data(prop, clean_pairs(df, from_key, to_key), kind)
