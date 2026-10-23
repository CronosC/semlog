"""The settings of a build.

`OntoBuildSettings` holds the choices a build is configured with. It is frozen
and holds nothing read from a log or a file. Construction parses the enum
fields and checks the bins. `validate` checks the fields against each other.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, ClassVar

from semanticlog.core.enums import (
    AttributeRepresentation,
    BuildEnum,
    EventOrderRepresentation,
    InverseRepresentation,
    OCELExtension,
    OWLProfile,
)
from semanticlog.core.errors import ConfigError
from semanticlog.core.vocabulary import TBOX_IRI
from semanticlog.data.binning import DEFAULT_BIN_LABELS, DEFAULT_BINS, check_bins

DEFAULT_MAX_EF_PAIRS = 50_000_000


@dataclass(frozen=True)
class OntoBuildSettings:
    """The choices a build is configured with.

    Sequence fields accept any iterable and are stored as tuples.

    Attributes:
        profile: OWL 2 profile the output stays inside.
        tbox_iri: Namespace of the schema classes and properties. Ends in
            `#` or `/`.
        order_representations: How event order is expressed. Empty for none.
        ocel_extensions: Optional OCEL 2.0 features to emit.
        inverse_representations: How inverse properties are expressed. Empty
            for no inverses.
        attribute_representations: How attribute values are expressed. Not
            empty.
        bins: Number of bins for the binning modes. At least two.
        bin_labels: Label per bin. Must have exactly `bins` entries, all
            different.
        derive_annotations: Emit labels and comments derived from the original
            names in the log.
        max_ef_pairs: Ceiling on materialized eventually-follows pairs. Zero
            for no limit.

    Raises:
        ConfigError: If a sequence field is given a plain string, an enum
            field an unknown value, or `tbox_iri` is empty, holds whitespace
            or `>`, or ends in neither `#` nor `/`. With a binning mode
            selected, also if `bins` is below two or `bin_labels` does not
            hold exactly `bins` entries that all differ.
    """

    profile: OWLProfile = OWLProfile.OWL2DL
    tbox_iri: str = TBOX_IRI
    order_representations: tuple[EventOrderRepresentation, ...] = ()
    ocel_extensions: tuple[OCELExtension, ...] = ()
    inverse_representations: tuple[InverseRepresentation, ...] = ()
    attribute_representations: tuple[AttributeRepresentation, ...] = (
        AttributeRepresentation.INDIVIDUALS,
    )
    bins: int = DEFAULT_BINS
    bin_labels: tuple[str, ...] = tuple(DEFAULT_BIN_LABELS)
    derive_annotations: bool = False
    max_ef_pairs: int = DEFAULT_MAX_EF_PAIRS

    _SEQUENCES: ClassVar[tuple[str, ...]] = (
        "order_representations",
        "ocel_extensions",
        "inverse_representations",
        "attribute_representations",
        "bin_labels",
    )

    _ENUM_FIELDS: ClassVar[dict[str, type[BuildEnum]]] = {
        "profile": OWLProfile,
        "order_representations": EventOrderRepresentation,
        "ocel_extensions": OCELExtension,
        "inverse_representations": InverseRepresentation,
        "attribute_representations": AttributeRepresentation,
    }

    def __post_init__(self) -> None:
        for name in self._SEQUENCES:
            value = getattr(self, name)
            if isinstance(value, str):
                raise ConfigError(
                    f"{name} must be a list, got the string {value!r}. "
                    f"Pass [{value!r}]."
                )
            if not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))
        self._resolve_enums()
        check_tbox_iri(self.tbox_iri)
        if self.binning_mode is not None:
            check_bins(self.bins, self.bin_labels)

    def _resolve_enums(self) -> None:
        """Parse each enum field, given as a member or its value, into its member."""
        for name, enum_cls in self._ENUM_FIELDS.items():
            value = getattr(self, name)
            if isinstance(value, tuple):
                resolved: Any = tuple(enum_cls.parse(v, key=name) for v in value)
            else:
                resolved = enum_cls.parse(value, key=name)
            object.__setattr__(self, name, resolved)

    def replace(self, **changes: Any) -> OntoBuildSettings:
        """Return a copy with the given fields changed.

        Raises:
            ConfigError: If a changed field is invalid, as on construction.
        """
        return dataclasses.replace(self, **changes)

    @property
    def has_inverses(self) -> bool:
        """Whether any inverse representation was selected."""
        return bool(self.inverse_representations)

    @property
    def axiomatic_inverses(self) -> bool:
        """Whether inverses are related by an axiom."""
        return InverseRepresentation.AXIOMATIC in self.inverse_representations

    @property
    def implicit_inverses(self) -> bool:
        """Whether inverse pairs are asserted directly."""
        return InverseRepresentation.IMPLICIT in self.inverse_representations

    @property
    def emits_ef(self) -> bool:
        """Whether eventually-follows is emitted in either form."""
        return self.emits_ef_axiomatic or self.emits_ef_explicit

    @property
    def emits_ef_axiomatic(self) -> bool:
        """Whether eventually-follows is left to a reasoner."""
        return EventOrderRepresentation.EF in self.order_representations

    @property
    def emits_ef_explicit(self) -> bool:
        """Whether every eventually-follows pair is asserted."""
        return EventOrderRepresentation.EF_EXPLICIT in self.order_representations

    @property
    def emits_df(self) -> bool:
        """Whether directly-follows properties and their pairs are emitted.

        True also for axiomatic EF, whose pairs a reasoner derives over
        directly-follows.
        """
        return (
            EventOrderRepresentation.DF in self.order_representations
            or self.emits_ef_axiomatic
        )

    @property
    def emits_declare(self) -> bool:
        """Whether traces are typed by the DECLARE constraints they satisfy."""
        return EventOrderRepresentation.DECLARE in self.order_representations

    @property
    def emits_traces(self) -> bool:
        """Whether the log has to be projected onto its case concepts."""
        return self.emits_df or self.emits_ef

    @property
    def emits_attribute_history(self) -> bool:
        """Whether object attribute changes are emitted."""
        return OCELExtension.ATTRIBUTE_HISTORY in self.ocel_extensions

    @property
    def emits_value_individuals(self) -> bool:
        """Whether each attribute value becomes an individual of an attribute class."""
        return AttributeRepresentation.INDIVIDUALS in self.attribute_representations

    @property
    def emits_entity_literals(self) -> bool:
        """Whether an event, object or change record carries its literals itself."""
        return AttributeRepresentation.LITERALS in self.attribute_representations

    @property
    def binning_mode(self) -> AttributeRepresentation | None:
        """The selected binning mode, or None if numeric values are not binned."""
        for mode in (
            AttributeRepresentation.EQUAL_BINS,
            AttributeRepresentation.DIST_BINS,
        ):
            if mode in self.attribute_representations:
                return mode
        return None

    def validate(self) -> None:
        """Check that the fields do not contradict each other.

        Raises:
            ConfigError: If any setting is unusable. The message lists every
                problem.
        """
        errors: list[str] = []
        is_el = self.profile == OWLProfile.OWL2EL

        if EventOrderRepresentation.EF in self.order_representations and is_el:
            errors.append(
                "ef is not available under OWL2EL. Use ef_explicit, or set "
                "profile to OWL2DL."
            )

        if (
            self.emits_ef_axiomatic
            and EventOrderRepresentation.DF not in self.order_representations
        ):
            errors.append(
                "ef needs df in order_representations. Add df, or use ef_explicit."
            )

        if self.axiomatic_inverses and is_el:
            errors.append(
                "axiomatic inverses (InverseObjectProperties axioms) are not "
                "available under OWL2EL. Remove axiomatic from "
                "inverse_representations, or set profile to OWL2DL."
            )

        if not self.attribute_representations:
            errors.append(
                "attribute_representations is empty. Choose individuals, "
                "literals or both."
            )

        if (
            AttributeRepresentation.EQUAL_BINS in self.attribute_representations
            and AttributeRepresentation.DIST_BINS in self.attribute_representations
        ):
            errors.append(
                "equal_bins and dist_bins cannot both be set. Keep at most one "
                "in attribute_representations."
            )

        if self.binning_mode is not None and not self.emits_value_individuals:
            errors.append(
                f"{self.binning_mode.value} needs individuals in "
                "attribute_representations, because a bin is a class of value "
                "individuals."
            )

        if errors:
            raise ConfigError(
                "Invalid OntoBuildSettings:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    def to_dict(self) -> dict:
        """Return every field as a JSON-compatible value, keyed by field name."""
        return {
            field.name: _plain(getattr(self, field.name))
            for field in dataclasses.fields(self)
        }


def check_tbox_iri(iri: str) -> None:
    """Raise `ConfigError` unless `iri` can hold the schema names.

    It must be non-empty, hold no whitespace or `>`, and end in `#` or `/`.
    """
    if not iri or any(c.isspace() for c in iri) or ">" in iri:
        raise ConfigError(
            f"tbox_iri {iri!r} cannot be written into an ontology. Use a "
            "non-empty IRI without whitespace or '>'."
        )
    if not iri.endswith(("#", "/")):
        raise ConfigError(
            f"tbox_iri must end in '#' or '/', got {iri!r}. Append one of them."
        )


def _plain(value: Any) -> Any:
    """Return a setting as a JSON-compatible value."""
    if isinstance(value, BuildEnum):
        return value.value
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value
