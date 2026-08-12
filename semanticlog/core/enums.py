"""The choices a build offers.

These sit in `core` because the data, ontology and CLI layers all name them.
`OntoBuildSettings` re-exports them, which is where callers normally reach
them from.
"""

from __future__ import annotations

from enum import Enum


class OWLProfile(str, Enum):
    """OWL 2 profile the emitted ontology stays inside.

    Attributes:
        OWL2DL: The full profile, including inverse and transitivity axioms.
        OWL2EL: A lightweight subset. Constructs it forbids are omitted from
            the output.
    """

    OWL2DL = "OWL2DL"
    OWL2EL = "OWL2EL"


class DataRepresentation(str, Enum):
    """How numeric attribute values reach the ontology.

    `VALUES` may be combined with one binning mode. The two binning modes are
    mutually exclusive. An empty selection omits numeric attributes.

    Attributes:
        VALUES: One individual per distinct value, carrying the number itself
            as a literal.
        EQUAL_BINS: Equal-frequency bins, each becoming a subclass of the
            attribute class.
        DIST_BINS: Bins cut from a fitted distribution, falling back to equal
            frequency when no distribution fits.
    """

    VALUES = "values"
    EQUAL_BINS = "equal_bins"
    DIST_BINS = "dist_bins"


class OCELExtension(str, Enum):
    """Optional parts of OCEL 2.0 to load and emit.

    Attributes:
        ATTRIBUTE_HISTORY: Per-object attribute changes with their timestamps,
            not just the latest value of each attribute.
    """

    ATTRIBUTE_HISTORY = "attribute_history"


class EventOrderRepresentation(str, Enum):
    """How the order of events within a trace is expressed.

    Several may be combined. All of them need at least one case concept,
    because a trace is the projection of the log onto one object.

    Attributes:
        DF: Directly-follows assertions between adjacent events.
        EF: Eventually-follows as a transitive property, leaving the pairs
            for a reasoner to derive. OWL2DL only.
        EF_EXPLICIT: Eventually-follows with every pair asserted. Works in
            both profiles and is quadratic in trace length.
        DECLARE: Each trace is typed as a member of every mined DECLARE
            constraint it satisfies.
    """

    DF = "df"
    EF = "ef"
    EF_EXPLICIT = "ef_explicit"
    DECLARE = "declare"


class InverseRepresentation(str, Enum):
    """How the inverse of each object property is expressed.

    An empty selection emits no inverses at all.

    Attributes:
        AXIOMATIC: Declare the inverse and relate the two with an axiom.
            OWL2DL only.
        IMPLICIT: Assert the reversed pairs directly, which needs no axiom and
            no reasoner.
    """

    AXIOMATIC = "axiomatic"
    IMPLICIT = "implicit"
