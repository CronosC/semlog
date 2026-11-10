"""The enumerated settings of a build.

`semlog` re-exports the ones a caller sets on a build.
"""

from __future__ import annotations

from enum import Enum
from pathlib import PurePath
from typing import Self

from semlog.core.errors import ConfigError


class SinkEntity(str, Enum):
    """The OWL entity kind a name is declared as.

    A sink writes a different declaration for each.

    Attributes:
        CLASS: A class.
        OBJECT_PROPERTY: An object property, which relates two individuals.
        DATA_PROPERTY: A data property, which relates an individual to a literal.
        ANNOTATION_PROPERTY: An annotation property, which carries no meaning
            for a reasoner.
    """

    CLASS = "class"
    OBJECT_PROPERTY = "object"
    DATA_PROPERTY = "data"
    ANNOTATION_PROPERTY = "annotation"

    @classmethod
    def property_for(cls, data_prop: bool) -> SinkEntity:
        """Return `DATA_PROPERTY` when `data_prop` is true, else `OBJECT_PROPERTY`."""
        return cls.DATA_PROPERTY if data_prop else cls.OBJECT_PROPERTY


class IndividualKind(str, Enum):
    """The kind of an individual, which decides the path it is written under.

    Every kind has a path of its own below the log base, so an id or value
    never shares an IRI with a class, a property or an individual of another
    kind.

    Attributes:
        EVENT: An event.
        OBJECT: An object.
        VALUE: An attribute value.
        TIME: A time point.
        CHANGE: An attribute change record.
    """

    EVENT = "event"
    OBJECT = "object"
    VALUE = "value"
    TIME = "time"
    CHANGE = "change"

    @property
    def path(self) -> str:
        """The path segment below the log base, ending in `/`."""
        return f"{self.value}/"


class BuildEnum(str, Enum):
    """Base for a setting given as text or as a member.

    `parse` turns a TOML, command-line or Python value into a member.
    """

    @classmethod
    def parse(cls, value: object, *, key: str) -> Self:
        """Return the member a written setting names, in any case.

        Args:
            value: What was written. A member is returned unchanged.
            key: Setting name, quoted in the error.

        Raises:
            ConfigError: If no member matches. The message lists the accepted
                values.
        """
        if isinstance(value, cls):
            return value
        folded = str(value).strip().casefold()
        for member in cls:
            if folded == str(member.value).casefold():
                return member
        allowed = ", ".join(str(m.value) for m in cls)
        raise ConfigError(f"{key}: unknown value {value!r}. Choose from: {allowed}.")


class OWLProfile(BuildEnum):
    """OWL 2 profile the emitted ontology stays inside.

    Attributes:
        OWL2DL: The full profile, including inverse and transitivity axioms.
        OWL2EL: A lightweight subset. Constructs it forbids are omitted from
            the output.
    """

    OWL2DL = "OWL2DL"
    OWL2EL = "OWL2EL"


class AttributeRepresentation(BuildEnum):
    """How attribute values reach the ontology.

    `INDIVIDUALS` and `LITERALS` may be combined. A bin mode needs
    `INDIVIDUALS`, and the two bin modes are mutually exclusive. The selection
    must not be empty.

    Attributes:
        INDIVIDUALS: One class per attribute and one individual per distinct
            value, carrying its literal. The event or object points at the
            individual.
        LITERALS: The literal on the event or object itself.
        EQUAL_BINS: Equal-frequency bins of each numeric attribute, each a
            subclass of the attribute class.
        DIST_BINS: Bins one standard deviation wide around the median, after
            a power transform makes the column close to normal.
    """

    INDIVIDUALS = "individuals"
    LITERALS = "literals"
    EQUAL_BINS = "equal_bins"
    DIST_BINS = "dist_bins"


class OCELExtension(BuildEnum):
    """Optional parts of OCEL 2.0 to load and emit.

    Attributes:
        ATTRIBUTE_HISTORY: Per-object attribute changes with their timestamps,
            in addition to the latest value of each attribute.
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
            so it needs `DF` alongside it. OWL2DL only. An event in two
            traces of one case concept joins their directly-follows chains,
            so the reasoner also derives pairs across those traces. The
            build warns when that happens.
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
    """Reasoner run over a built ontology.

    No backend supports every query. `semlog.onto.reason` records what
    each one does not support.

    Attributes:
        HERMIT: Supports every query. Rejects a timestamp with microseconds.
        PELLET: Supports every query, including those timestamps.
        ELK: Supports class queries only. The reasoner for OWL 2 EL.
    """

    HERMIT = "HermiT"
    PELLET = "Pellet"
    ELK = "ELK"


class OutputFormat(BuildEnum):
    """Format a built ontology is also written in, named by its file suffix.

    Attributes:
        OFN: OWL functional syntax.
        OWL: RDF/XML.
        OWX: OWL/XML.
        TTL: Turtle.
        OMN: Manchester syntax.
    """

    OFN = "ofn"
    OWL = "owl"
    OWX = "owx"
    TTL = "ttl"
    OMN = "omn"

    @classmethod
    def of_path(cls, path: str | PurePath) -> OutputFormat:
        """Return the format the suffix of a path names, in any case.

        Raises:
            ConfigError: If the suffix names no format.
        """
        suffix = PurePath(path).suffix.removeprefix(".").casefold()
        for member in cls:
            if member.value == suffix:
                return member
        suffixes = ", ".join(f".{m.value}" for m in cls)
        raise ConfigError(
            f"Cannot write an ontology to {path}: its suffix names no format. "
            f"Use one of {suffixes}."
        )


class InverseRepresentation(BuildEnum):
    """How the inverse of each object property is expressed.

    An empty selection emits no inverses.

    Attributes:
        AXIOMATIC: Declare the inverse and relate the two with an axiom.
            OWL2DL only.
        IMPLICIT: Assert the reversed pairs directly, which needs no axiom and
            no reasoner.
    """

    AXIOMATIC = "axiomatic"
    IMPLICIT = "implicit"
