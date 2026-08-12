"""What a build emits, as one checked value.

`OntoBuildSettings` is the single configuration object of a build. It is
frozen, it validates itself, and it resolves the files it points at when it is
constructed, so a builder never has to check whether an input is loaded yet.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from semanticlog.core.enums import (
    DataRepresentation as DataRepresentation,
)
from semanticlog.core.enums import (
    EventOrderRepresentation as EventOrderRepresentation,
)
from semanticlog.core.enums import (
    InverseRepresentation as InverseRepresentation,
)
from semanticlog.core.enums import (
    OCELExtension as OCELExtension,
)
from semanticlog.core.enums import (
    OWLProfile as OWLProfile,
)
from semanticlog.core.iri import DEFAULT_IRI as DEFAULT_IRI
from semanticlog.data.binning import DEFAULT_BIN_LABELS as DEFAULT_BIN_LABELS
from semanticlog.data.binning import DEFAULT_BINS as DEFAULT_BINS
from semanticlog.enrich.declare import DeclareResult

DEFAULT_MAX_EF_PAIRS = 50_000_000


@dataclass(frozen=True)
class OntoBuildSettings:
    """Everything a build needs to know beyond the data itself.

    Sequence fields accept any iterable and are stored as tuples. The two
    file-backed fields are read once, when the settings are constructed, and
    the loaded content is then available as `annotation_data` and
    `declare_result`.

    Attributes:
        profile: OWL 2 profile the output stays inside.
        order_representations: How event order is expressed. Empty for none.
        ocel_extensions: Optional OCEL 2.0 features to emit.
        case_concepts: Object types whose instances act as cases. Required by
            every order representation.
        declare_file: Path to a mined DECLARE constraints CSV.
        inverse_representations: How inverse properties are expressed. Empty
            for no inverses.
        data_representations: How numeric attribute values are expressed.
        bins: Number of bins for the binning modes.
        bin_labels: Label per bin. Must have exactly `bins` entries.
        annotation_document: Path to a JSON document of entity name to label
            and comment.
        derive_annotations: Emit labels and comments derived from the original
            names in the log.
        max_ef_pairs: Ceiling on materialized eventually-follows pairs, which
            grow quadratically with trace length. Zero for no limit.
        annotation_data: Loaded content of `annotation_document`, or annotations
            passed directly.
        declare_result: Loaded content of `declare_file`, or constraints passed
            directly.

    Raises:
        ValueError: If `bin_labels` does not have exactly `bins` entries.
    """

    profile: OWLProfile = OWLProfile.OWL2DL
    order_representations: tuple[EventOrderRepresentation, ...] = ()
    ocel_extensions: tuple[OCELExtension, ...] = ()
    case_concepts: tuple[str, ...] = ()
    declare_file: str | None = None
    inverse_representations: tuple[InverseRepresentation, ...] = ()
    data_representations: tuple[DataRepresentation, ...] = (DataRepresentation.VALUES,)
    bins: int = DEFAULT_BINS
    bin_labels: tuple[str, ...] = tuple(DEFAULT_BIN_LABELS)
    annotation_document: str | None = None
    derive_annotations: bool = False
    max_ef_pairs: int = DEFAULT_MAX_EF_PAIRS
    annotation_data: dict[str, dict[str, str]] | None = None
    declare_result: DeclareResult | None = None

    _SEQUENCES = (
        "order_representations",
        "ocel_extensions",
        "case_concepts",
        "inverse_representations",
        "data_representations",
        "bin_labels",
    )

    def __post_init__(self) -> None:
        for name in self._SEQUENCES:
            value = getattr(self, name)
            if not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))
        self._resolve_inputs()
        problem = self._bin_label_problem()
        if problem is not None:
            raise ValueError(problem)

    def _resolve_inputs(self) -> None:
        """Read the file-backed fields, and copy content passed directly."""
        if self.annotation_data is None and self.annotation_document is not None:
            with open(self.annotation_document, encoding="utf-8") as f:
                object.__setattr__(self, "annotation_data", json.load(f))
        elif self.annotation_data is not None:
            object.__setattr__(
                self, "annotation_data", copy.deepcopy(self.annotation_data)
            )
        if self.declare_result is None and self.declare_file is not None:
            object.__setattr__(
                self, "declare_result", DeclareResult.read(self.declare_file)
            )

    def _bin_label_problem(self) -> str | None:
        if len(self.bin_labels) != self.bins:
            return (
                f"bin_labels must have exactly bins={self.bins} "
                f"elements, got {len(self.bin_labels)}."
            )
        return None

    def replace(self, **changes: Any) -> OntoBuildSettings:
        """Return a copy with some fields changed.

        Changing a path field drops the content loaded from the old path, so
        the new one is read instead of the stale content being kept.

        Args:
            **changes: Field values to change.

        Returns:
            A new settings object.
        """
        for path_field, content_field in (
            ("declare_file", "declare_result"),
            ("annotation_document", "annotation_data"),
        ):
            if path_field in changes and content_field not in changes:
                changes[content_field] = None
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
    def emits_declare(self) -> bool:
        """Whether traces are typed by the DECLARE constraints they satisfy."""
        return EventOrderRepresentation.DECLARE in self.order_representations

    @property
    def emits_traces(self) -> bool:
        """Whether the log has to be projected onto its case concepts."""
        return bool(
            {EventOrderRepresentation.DF, EventOrderRepresentation.EF}
            & set(self.order_representations)
            or self.emits_ef
        )

    @property
    def emits_attribute_history(self) -> bool:
        """Whether object attribute changes are emitted."""
        return OCELExtension.ATTRIBUTE_HISTORY in self.ocel_extensions

    @property
    def binning_mode(self) -> DataRepresentation | None:
        """The selected binning mode, or None if numeric values are not binned."""
        for mode in (DataRepresentation.EQUAL_BINS, DataRepresentation.DIST_BINS):
            if mode in self.data_representations:
                return mode
        return None

    def validate(self, require_inputs: bool = True) -> None:
        """Check that the settings describe a build that can run.

        Every problem is collected before raising, so one call reports all of
        them.

        Args:
            require_inputs: Also check the settings that depend on data the
                build has not seen yet, such as case concepts and DECLARE
                constraints. The CLI resolves its configuration before those
                are known, so it passes False there and the builder checks
                again with True.

        Raises:
            ValueError: If any setting is unusable, listing every problem.
        """
        errors: list[str] = []
        is_el = self.profile == OWLProfile.OWL2EL

        bin_problem = self._bin_label_problem()
        if bin_problem is not None:
            errors.append(bin_problem)

        if EventOrderRepresentation.EF in self.order_representations and is_el:
            errors.append(
                "EventOrderRepresentation.EF uses TBox transitivity axioms which are not "
                "supported in OWL2EL. Use EventOrderRepresentation.EF_EXPLICIT instead, or "
                "switch to OWLProfile.OWL2DL."
            )

        if self.axiomatic_inverses and is_el:
            errors.append(
                "InverseRepresentation.AXIOMATIC (InverseObjectProperty TBox axioms) is not "
                "supported in OWL2EL. Remove AXIOMATIC from inverse_representations or switch "
                "to OWLProfile.OWL2DL."
            )

        if (
            DataRepresentation.EQUAL_BINS in self.data_representations
            and DataRepresentation.DIST_BINS in self.data_representations
        ):
            errors.append(
                "DataRepresentation.EQUAL_BINS and DIST_BINS are mutually exclusive; "
                "include at most one in data_representations."
            )

        if require_inputs:
            if self.emits_traces and not self.case_concepts:
                errors.append(
                    "order_representations includes DF/EF/EF_EXPLICIT but case_concepts is "
                    "empty. Set settings.case_concepts before calling build_ontology()."
                )

            if self.emits_declare and self.declare_result is None:
                errors.append(
                    "order_representations includes DECLARE but no DECLARE constraints are "
                    "available. Set settings.declare_file or assign settings.declare_result."
                )

        if errors:
            raise ValueError(
                "Invalid OntoBuildSettings:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    def to_dict(self) -> dict:
        """Return the settings as plain JSON-compatible values.

        This is what a build records as its provenance, so every field that
        changes the output appears here.

        Returns:
            A dict of primitives, lists and dicts.
        """
        return {
            "profile": self.profile.value,
            "order_representations": [e.value for e in self.order_representations],
            "ocel_extensions": [e.value for e in self.ocel_extensions],
            "case_concepts": list(self.case_concepts),
            "declare_file": self.declare_file,
            "declare_result": (
                self.declare_result.to_dict()
                if self.declare_result is not None
                else None
            ),
            "inverse_representations": [e.value for e in self.inverse_representations],
            "data_representations": [e.value for e in self.data_representations],
            "bins": self.bins,
            "bin_labels": list(self.bin_labels),
            "annotation_document": self.annotation_document,
            "annotation_data": self.annotation_data,
            "derive_annotations": self.derive_annotations,
            "max_ef_pairs": self.max_ef_pairs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> OntoBuildSettings:
        """Rebuild settings from the output of `to_dict`.

        Args:
            d: A dict as written by `to_dict`. Only `profile` is required, and
                every other key falls back to its default.

        Returns:
            The settings the dict describes.
        """
        declare_result = d.get("declare_result")
        return cls(
            profile=OWLProfile(d["profile"]),
            order_representations=_members(
                EventOrderRepresentation, d.get("order_representations", ())
            ),
            ocel_extensions=_members(OCELExtension, d.get("ocel_extensions", ())),
            case_concepts=tuple(d.get("case_concepts", ())),
            declare_file=d.get("declare_file"),
            inverse_representations=_members(
                InverseRepresentation, d.get("inverse_representations", ())
            ),
            data_representations=_members(
                DataRepresentation,
                d.get("data_representations", (DataRepresentation.VALUES,)),
            ),
            bins=d.get("bins", DEFAULT_BINS),
            bin_labels=tuple(d.get("bin_labels", DEFAULT_BIN_LABELS)),
            annotation_document=d.get("annotation_document"),
            derive_annotations=d.get("derive_annotations", False),
            max_ef_pairs=d.get("max_ef_pairs", DEFAULT_MAX_EF_PAIRS),
            annotation_data=d.get("annotation_data"),
            declare_result=(
                DeclareResult.from_dict(declare_result)
                if declare_result is not None
                else None
            ),
        )


def _members[E: Enum](enum_cls: type[E], values: Iterable) -> tuple[E, ...]:
    """Convert stored values back into enum members."""
    return tuple(enum_cls(v) for v in values)
