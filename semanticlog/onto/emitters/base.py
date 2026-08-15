"""What every emitter is given to work with."""

from __future__ import annotations

import pandas as pd

from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.emitted import EmittedIndividuals
from semanticlog.onto.emitters.pairs import clean_pairs, reversed_pairs
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import AxiomSink


class Emitter:
    """Base for a build step that writes axioms.

    An emitter holds no data of its own beyond what it is given here, so it
    can be tested against a recording sink with no ontology involved.

    Attributes:
        sink: Where axioms go.
        settings: Settings of this build.
        annotations: Writer for the labels and comments that go with the
            axioms. It does nothing unless the build asked for annotations.
        emitted: What the build has already asserted as an individual, shared
            with every other emitter that reaches the same ones.
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

    def assert_relations(
        self,
        prop: str,
        df: pd.DataFrame,
        from_key: str,
        to_key: str,
        inverse_name: str | None = None,
    ) -> None:
        """Assert an object property for every row of a table.

        A build that materializes its inverses asserts the reversed pairs as
        well. Which of the two an inverse is expressed as is a choice about
        the build, so it is made here rather than at the axiom boundary.

        Args:
            prop: Property to assert.
            df: Table holding the pairs.
            from_key: Column holding the subject.
            to_key: Column holding the target.
            inverse_name: Name to use for the reversed pairs, instead of the
                derived one.
        """
        if not self.settings.implicit_inverses:
            self.sink.assert_relations(prop, clean_pairs(df, from_key, to_key))
            return
        pairs = list(clean_pairs(df, from_key, to_key))
        self.sink.assert_relations(prop, pairs)
        inverse = inverse_name or self.sink.inverse_name(prop)
        self.sink.assert_relations(inverse, reversed_pairs(pairs))

    def assert_data(
        self, prop: str, df: pd.DataFrame, from_key: str, to_key: str
    ) -> None:
        """Assert a data property for every row of a table.

        Args:
            prop: Property to assert.
            df: Table holding the individuals and their values.
            from_key: Column holding the subject.
            to_key: Column holding the literal value.
        """
        self.sink.assert_data(prop, clean_pairs(df, from_key, to_key))
