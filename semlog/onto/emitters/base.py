"""The base class of every emitter."""

from __future__ import annotations

from typing import Any

import pandas as pd

from semlog.core.enums import IndividualKind
from semlog.onto.annotations import AnnotationWriter
from semlog.onto.emitted import EmittedIndividuals
from semlog.onto.emitters.pairs import clean_pairs, reversed_pairs
from semlog.onto.settings import OntoBuildSettings
from semlog.onto.sinks import AxiomSink


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

    def add_object_property(
        self,
        name: str,
        domain_class: str | list[str] | None = None,
        range_class: str | list[str] | None = None,
        superprop: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Declare an object property, with the inverse the settings ask for.

        Other keyword arguments go to `AxiomSink.add_property`.
        """
        self.sink.add_property(
            name,
            domain_class,
            range_class,
            superprop,
            create_inverse=self.settings.has_inverses,
            axiomatic_inverse=self.settings.axiomatic_inverses,
            **kwargs,
        )

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
        """
        if not self.settings.implicit_inverses:
            self.sink.assert_relations(prop, clean_pairs(df, from_key, to_key), kinds)
            return
        pairs = list(clean_pairs(df, from_key, to_key))
        self.sink.assert_relations(prop, pairs, kinds)
        self.sink.assert_relations(
            self.sink.inverse_name(prop), reversed_pairs(pairs), kinds[::-1]
        )

    def assert_data(
        self,
        prop: str,
        df: pd.DataFrame,
        from_key: str,
        to_key: str,
        kind: IndividualKind,
    ) -> None:
        """Assert a data property from `from_key` to the literal in `to_key`, per row."""
        self.sink.assert_data(prop, clean_pairs(df, from_key, to_key), kind)
