"""The choices a build offers.

These sit in `core` because the data, ontology and CLI layers all name them.
`semanticlog` re-exports the ones a caller sets on a build.
"""

from __future__ import annotations

from enum import Enum
from typing import Self


class BuildEnum(str, Enum):
    """Base for the choices, which arrive as text as often as they do as members.

    A setting reaches a build from a TOML file, a command line or Python, so
    `parse` is the one place that turns whatever was written into a member.
    """

    @classmethod
    def parse(cls, value: object, *, key: str) -> Self:
        """Return the member a written setting names.

        Both spellings are accepted, case-insensitively, so `df` and `DF` mean
        the same thing.

        Args:
            value: What was written. A member is returned unchanged.
            key: Setting name, quoted in the error.

        Returns:
            The member the value names.

        Raises:
            ValueError: If nothing matches, listing what would.
        """
        if isinstance(value, cls):
            return value
        folded = str(value).strip().casefold()
        for member in cls:
            if folded in (str(member.value).casefold(), member.name.casefold()):
                return member
        allowed = ", ".join(str(m.value) for m in cls)
        raise ValueError(f"{key}: unknown value {value!r}. Choose from: {allowed}.")

    @classmethod
    def spellings(cls) -> list[str]:
        """List the ways this enum can be written, for help text."""
        values = [str(m.value) for m in cls]
        return values + [m.name for m in cls if m.name not in values]


class OWLProfile(BuildEnum):
    """OWL 2 profile the emitted ontology stays inside.

    Attributes:
        OWL2DL: The full profile, including inverse and transitivity axioms.
        OWL2EL: A lightweight subset. Constructs it forbids are omitted from
            the output.
    """

    OWL2DL = "OWL2DL"
    OWL2EL = "OWL2EL"


class DataRepresentation(BuildEnum):
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


class OCELExtension(BuildEnum):
    """Optional parts of OCEL 2.0 to load and emit.

    Attributes:
        ATTRIBUTE_HISTORY: Per-object attribute changes with their timestamps,
            not just the latest value of each attribute.
    """

    ATTRIBUTE_HISTORY = "attribute_history"


class EventOrderRepresentation(BuildEnum):
    """How the order of events within a trace is expressed.

    Several may be combined. `DF`, `EF` and `EF_EXPLICIT` need at least one
    case concept, because a trace is the projection of the log onto one
    object. `DECLARE` takes its traces from the mined constraints instead.

    Attributes:
        DF: Directly-follows assertions between adjacent events.
        EF: Eventually-follows as a transitive property, leaving the pairs
            for a reasoner to derive. It derives them over directly-follows,
            so it needs `DF` alongside it. OWL2DL only.
        EF_EXPLICIT: Eventually-follows with every pair asserted. Works in
            both profiles and is quadratic in trace length.
        DECLARE: Each trace is typed as a member of every mined DECLARE
            constraint it satisfies.
    """

    DF = "df"
    EF = "ef"
    EF_EXPLICIT = "ef_explicit"
    DECLARE = "declare"


class ReasonerBackend(BuildEnum):
    """Reasoner that answers for a built ontology.

    No backend handles everything. What each one refuses is recorded in
    `semanticlog.onto.reason`.

    Attributes:
        HERMIT: Answers for everything, but refuses a timestamp carrying
            microseconds.
        PELLET: Answers for everything, and takes those timestamps.
        ELK: Answers for classes only, and is the reasoner for OWL 2 EL.
    """

    HERMIT = "HermiT"
    PELLET = "Pellet"
    ELK = "ELK"


REASONER_BACKENDS: tuple[str, ...] = tuple(b.value for b in ReasonerBackend)


def resolve_reasoner_backend(text: str, *, key: str) -> str:
    """Return the backend a written name selects, spelled as the OWL API wants it.

    The name reaches the OWL API and the provenance record as plain text, so
    this hands back the value rather than the member.

    Args:
        text: Backend name, in any case.
        key: Setting or flag being read, quoted in the error.

    Returns:
        One of `REASONER_BACKENDS`.

    Raises:
        ValueError: If the name is not one of them.
    """
    return ReasonerBackend.parse(text, key=key).value


class InverseRepresentation(BuildEnum):
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
