from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semanticlog.enrich.declare import DeclareResult

# Re-exported: callers have always imported DEFAULT_IRI from here, but the
# definition lives outside the onto package so the data layer can use it too.
from semanticlog.iri import DEFAULT_IRI as DEFAULT_IRI

# Classes that define configuration settings for onto instantiation


class OwlProfile(str, Enum):
    """
    Enum specifying the OWL 2 profile to use for ontology instantiation.

    - OWL2DL: Supports full OWL 2 DL semantics
    - OWL2EL: Lightweight profile optimised for performance
    """

    OWL2DL = "OWL2DL"
    OWL2EL = "OWL2EL"


class DataRepresentation(str, Enum):
    """
    Enum specifying how numeric attribute columns are represented in the ontology.

    - NONE: Numeric attributes are omitted entirely — no class, property, or individual.
    - VALUES: One named individual per distinct value; raw number attached via has_data_value.
    - EQUAL_BINS: Equal-frequency binning; one bin subclass per bin label (e.g.
      price_low), no literal. The attribute individual is typed as that bin subclass.
    - DIST_BINS: Distribution-fitted binning; falls back to EQUAL_BINS when no
      distribution fits. One bin subclass per label, no literal.
    """

    NONE = "none"
    VALUES = "values"
    EQUAL_BINS = "equal_bins"
    DIST_BINS = "dist_bins"


class OcelExtension(str, Enum):
    """
    Enum specifying optional OCEL 2.0 features to preserve in the pipeline.

    - ATTRIBUTE_HISTORY: Load and represent object attribute change history from
      pm4py's ``log.object_changes``. Adds ``AttributeChange`` individuals linked
      to their object, attribute value, and timestamp in the ontology.
    """

    ATTRIBUTE_HISTORY = "attribute_history"


class EventOrderRepresentation(str, Enum):
    """
    Enum specifying how event order should be represented in the ontology.

    - DF: Directly-follows — adjacent event pairs within each case trace are asserted
      under a per-case-concept sub-property of ``df``.
    - EF: Eventually-follows via TBox axioms — transitivity, sub-property, and local
      reflexivity axioms are added so a reasoner derives all eventually-follows pairs
      without ABox materialisation. OWL2DL only; raises ValueError for OWL2EL.
    - EF_explicit: Eventually-follows with explicit ABox materialisation — all
      eventually-follows pairs are enumerated and asserted directly. Compatible with
      both OWL2DL and OWL2EL.
    - DECLARE: DECLARE constraint mining — constraint formulae are added as named
      TBox subclasses of ``DeclareConstraint``; case instances whose traces satisfy
      a constraint are typed as members of the corresponding class.
    """

    DF = "df"
    EF = "ef"
    EF_explicit = "ef_explicit"
    DECLARE = "declare"


class InverseRepresentation(str, Enum):
    """
    Enum specifying how inverse relations are represented in the ontology.

    - AXIOMATIC: Declares each inverse via an ``InverseObjectProperty`` TBox axiom.
      A reasoner can then derive inverse assertions without ABox materialisation.
      OWL2DL only; not supported in OWL2EL.
    - IMPLICIT: Materialises the inverse of each relation assertion directly in the
      ABox. Enables inverse traversal via SPARQL without a reasoner. Supported in
      both OWL2DL and OWL2EL.
    """

    AXIOMATIC = "axiomatic"
    IMPLICIT = "implicit"


_DEFAULT_BINS = 5
_DEFAULT_BIN_LABELS = ["very_low", "low", "average", "high", "very_high"]

#: Ceiling on materialised eventually-follows pairs, per case concept.
#:
#: EF_explicit is quadratic in trace length, not in event count, so a log that
#: looks small can still ask for hundreds of millions of axioms: one 5,000-event
#: trace is 12.5M pairs on its own. 50M is high enough not to interrupt real work
#: and low enough to catch the runaway before it exhausts memory.
_DEFAULT_MAX_EF_PAIRS = 50_000_000


@dataclass
class OntoBuildSettings:
    """
    Configuration settings for ontology instantiation.

    Attributes:
        profile: The OWL 2 profile to use (OWL2DL or OWL2EL).
        order_representation: Event order strategies to include. Multiple strategies
            may be combined. Empty list omits event order entirely.
        ocel_extensions: Optional OCEL 2.0 features to enable (e.g.
            ATTRIBUTE_HISTORY). Empty list by default (no extensions active).
        case_concepts: Object types to use as case concepts; each defines an
            independent trace perspective. Must be set before calling build_ontology.
        declare_file: Path to a pre-computed DECLARE results CSV. Read on first access
            to ``declare_result``.
        inverse_representation: Inverse relation strategies to include. AXIOMATIC
            adds InverseObjectProperty TBox axioms (OWL2DL only); IMPLICIT
            materialises inverse assertions in the ABox (both profiles). Empty by
            default, meaning no inverse relations are emitted.
        data_representation: How numeric attribute columns are represented. A list of
            modes; empty list means omit numeric attributes entirely. VALUES and one
            binning mode may coexist; EQUAL_BINS and DIST_BINS are mutually exclusive.
        bins: Number of bins for EQUAL_BINS and DIST_BINS modes.
        bin_labels: Labels for each bin; must have exactly bins elements.
        annotation_document: Path to a JSON file mapping entity names to annotation
            dicts. Each entry may contain "label" and/or "comment" keys. Read on first
            access to ``annotation_data``.
        derive_annotations: If True, emit rdfs:label on every event, Time, and object
            individual using a human-readable label constructed from the source data.

    The ``annotation_data`` and ``declare_result`` properties resolve
    ``annotation_document`` and ``declare_file`` lazily on first read, and can also
    be assigned directly when the data did not come from a file.
    """

    profile: OwlProfile = OwlProfile.OWL2DL
    order_representation: list[EventOrderRepresentation] = field(default_factory=list)
    ocel_extensions: list[OcelExtension] = field(default_factory=list)
    case_concepts: list[str] = field(default_factory=list)
    declare_file: str | None = None
    inverse_representation: list[InverseRepresentation] = field(default_factory=list)
    data_representation: list[DataRepresentation] = field(
        default_factory=lambda: [DataRepresentation.VALUES]
    )
    bins: int = _DEFAULT_BINS
    bin_labels: list[str] = field(default_factory=lambda: list(_DEFAULT_BIN_LABELS))
    annotation_document: str | None = None
    derive_annotations: bool = False
    max_ef_pairs: int = _DEFAULT_MAX_EF_PAIRS

    def __post_init__(self) -> None:
        # Each cache remembers the path it came from, so re-pointing the path field
        # after a read invalidates it rather than returning the previous file.
        self._annotation_data: dict[str, dict[str, str]] | None = None
        self._annotation_source: str | None = None
        self._declare_result: DeclareResult | None = None
        self._declare_source: str | None = None
        problem = self._bin_label_problem()
        if problem is not None:
            raise ValueError(problem)

    def _bin_label_problem(self) -> str | None:
        if len(self.bin_labels) != self.bins:
            return (
                f"bin_labels must have exactly bins={self.bins} "
                f"elements, got {len(self.bin_labels)}."
            )
        return None

    # ------------------------------------------------------------------
    # Lazily resolved inputs
    # ------------------------------------------------------------------

    @property
    def annotation_data(self) -> dict[str, dict[str, str]] | None:
        """Annotation document contents, read from ``annotation_document`` on first access.

        Assigning directly takes precedence over the path for as long as
        ``annotation_document`` is not changed afterwards.
        """
        if self.annotation_document is not None and (
            self._annotation_data is None
            or self._annotation_source != self.annotation_document
        ):
            with open(self.annotation_document, encoding="utf-8") as f:
                self._annotation_data = json.load(f)
            self._annotation_source = self.annotation_document
        return self._annotation_data

    @annotation_data.setter
    def annotation_data(self, value: dict[str, dict[str, str]] | None) -> None:
        self._annotation_data = value
        self._annotation_source = self.annotation_document

    @property
    def declare_result(self) -> DeclareResult | None:
        """Mined DECLARE constraints, read from ``declare_file`` on first access.

        Assigning directly takes precedence over the path for as long as
        ``declare_file`` is not changed afterwards.
        """
        if self.declare_file is not None and (
            self._declare_result is None or self._declare_source != self.declare_file
        ):
            from semanticlog.enrich.declare import DeclareResult

            self._declare_result = DeclareResult.read(self.declare_file)
            self._declare_source = self.declare_file
        return self._declare_result

    @declare_result.setter
    def declare_result(self, value: DeclareResult | None) -> None:
        self._declare_result = value
        self._declare_source = self.declare_file

    # ------------------------------------------------------------------
    # Derived flags
    # ------------------------------------------------------------------

    @property
    def has_inverses(self) -> bool:
        """True if any inverse representation strategy is active."""
        return bool(self.inverse_representation)

    @property
    def axiomatic_inverses(self) -> bool:
        """True if inverses are declared via InverseObjectProperty TBox axioms."""
        return InverseRepresentation.AXIOMATIC in self.inverse_representation

    @property
    def implicit_inverses(self) -> bool:
        """True if inverse assertions are materialised in the ABox."""
        return InverseRepresentation.IMPLICIT in self.inverse_representation

    @property
    def instantiate_df(self) -> bool:
        """True if directly-follows relations should be asserted."""
        return EventOrderRepresentation.DF in self.order_representation

    @property
    def instantiate_ef(self) -> bool:
        """True if an eventually-follows property is needed, axiomatically or explicitly."""
        return self.instantiate_ef_axiomatic or self.instantiate_ef_explicit

    @property
    def instantiate_ef_axiomatic(self) -> bool:
        """True if eventually-follows should be left for a reasoner to derive.

        Distinct from :attr:`instantiate_ef_explicit`: the two are alternative
        strategies for the same relation, so selecting materialisation must not
        also emit the TBox axioms that materialisation exists to avoid.
        """
        return EventOrderRepresentation.EF in self.order_representation

    @property
    def instantiate_ef_explicit(self) -> bool:
        """True if every eventually-follows pair should be materialised in the ABox."""
        return EventOrderRepresentation.EF_explicit in self.order_representation

    @property
    def instantiate_declare(self) -> bool:
        """True if DECLARE constraint membership should be asserted."""
        return EventOrderRepresentation.DECLARE in self.order_representation

    @property
    def instantiate_traces(self) -> bool:
        """True if any order strategy requires case traces, and therefore case concepts."""
        return self.instantiate_df or self.instantiate_ef

    @property
    def instantiate_attribute_history(self) -> bool:
        """True if object attribute change history should be represented."""
        return OcelExtension.ATTRIBUTE_HISTORY in self.ocel_extensions

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, require_inputs: bool = True) -> None:
        """Raise ValueError if the configuration is internally inconsistent.

        Checks profile compatibility for EF and AXIOMATIC inverses and mutual
        exclusivity of EQUAL_BINS and DIST_BINS.  All problems are collected and
        reported together.

        Args:
            require_inputs: Also check that the data each active strategy needs is
                present (``case_concepts``, ``declare_result``).  Pass ``False`` to
                validate a flag combination before that data has been gathered.

        Raises:
            ValueError: If any check fails.
        """
        errors: list[str] = []
        is_el = self.profile == OwlProfile.OWL2EL

        # Re-checked here as well as in __post_init__: bins and bin_labels are
        # commonly assigned field-by-field after construction (e.g. by the CLI).
        bin_problem = self._bin_label_problem()
        if bin_problem is not None:
            errors.append(bin_problem)

        if EventOrderRepresentation.EF in self.order_representation and is_el:
            errors.append(
                "EventOrderRepresentation.EF uses TBox transitivity axioms which are not "
                "supported in OWL2EL. Use EventOrderRepresentation.EF_explicit instead, or "
                "switch to OwlProfile.OWL2DL."
            )

        if self.axiomatic_inverses and is_el:
            errors.append(
                "InverseRepresentation.AXIOMATIC (InverseObjectProperty TBox axioms) is not "
                "supported in OWL2EL. Remove AXIOMATIC from inverse_representation or switch "
                "to OwlProfile.OWL2DL."
            )

        if (
            DataRepresentation.EQUAL_BINS in self.data_representation
            and DataRepresentation.DIST_BINS in self.data_representation
        ):
            errors.append(
                "DataRepresentation.EQUAL_BINS and DIST_BINS are mutually exclusive; "
                "include at most one in data_representation."
            )

        if require_inputs:
            if self.instantiate_traces and not self.case_concepts:
                errors.append(
                    "order_representation includes DF/EF/EF_explicit but case_concepts is "
                    "empty. Set settings.case_concepts before calling build_ontology()."
                )

            if self.instantiate_declare and self.declare_result is None:
                errors.append(
                    "order_representation includes DECLARE but no DECLARE constraints are "
                    "available. Set settings.declare_file or assign settings.declare_result."
                )

        if errors:
            raise ValueError(
                "Invalid OntoBuildSettings:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def resolve(self) -> None:
        """Read every lazily-loaded input now.

        After this, nothing else touches the filesystem, so serialising or
        inspecting the settings has no side effects and cannot fail on a missing
        file halfway through a build.
        """
        _ = self.annotation_data
        _ = self.declare_result

    def snapshot(self) -> OntoBuildSettings:
        """Return an independent copy with the same resolved inputs.

        A build takes one of these so that mutating the caller's settings
        afterwards cannot change what the build does, and so the settings stored
        on the finished :class:`~semanticlog.onto.semlog.SemanticLog` are provably
        the ones that produced it. The list fields are copied because those are
        what callers mutate in place; the resolved caches are carried over so the
        copy does not re-read from disk.
        """
        clone = copy.copy(self)
        clone.order_representation = list(self.order_representation)
        clone.ocel_extensions = list(self.ocel_extensions)
        clone.case_concepts = list(self.case_concepts)
        clone.inverse_representation = list(self.inverse_representation)
        clone.data_representation = list(self.data_representation)
        clone.bin_labels = list(self.bin_labels)
        return clone

    def to_dict(self) -> dict:
        """Serialise settings to a JSON-compatible dict.

        Reports the inputs already resolved rather than reading them: a
        serialisation call that silently opened files would make the result
        depend on when it was called. Call :meth:`resolve` first to include them.
        """
        declare_result = self._declare_result
        return {
            "profile": self.profile.value,
            "order_representation": [e.value for e in self.order_representation],
            "ocel_extensions": [e.value for e in self.ocel_extensions],
            "case_concepts": self.case_concepts,
            "declare_file": self.declare_file,
            "declare_result": (
                declare_result.to_dict() if declare_result is not None else None
            ),
            "inverse_representation": [e.value for e in self.inverse_representation],
            "data_representation": [e.value for e in self.data_representation],
            "bins": self.bins,
            "bin_labels": self.bin_labels,
            "annotation_document": self.annotation_document,
            "annotation_data": self._annotation_data,
            "derive_annotations": self.derive_annotations,
            "max_ef_pairs": self.max_ef_pairs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> OntoBuildSettings:
        """Restore settings from a dict produced by to_dict()."""
        from semanticlog.enrich.declare import DeclareResult

        settings = cls(
            profile=OwlProfile(d["profile"]),
            order_representation=[
                EventOrderRepresentation(v) for v in d.get("order_representation", [])
            ],
            ocel_extensions=[OcelExtension(v) for v in d.get("ocel_extensions", [])],
            case_concepts=d.get("case_concepts", []),
            declare_file=d.get("declare_file"),
            inverse_representation=[
                InverseRepresentation(v) for v in d.get("inverse_representation", [])
            ],
            data_representation=[
                DataRepresentation(v)
                for v in d.get("data_representation", [DataRepresentation.VALUES])
            ],
            bins=d.get("bins", _DEFAULT_BINS),
            bin_labels=d.get("bin_labels", list(_DEFAULT_BIN_LABELS)),
            annotation_document=d.get("annotation_document"),
            derive_annotations=d.get("derive_annotations", False),
            max_ef_pairs=d.get("max_ef_pairs", _DEFAULT_MAX_EF_PAIRS),
        )
        if d.get("declare_result") is not None:
            settings.declare_result = DeclareResult.from_dict(d["declare_result"])
        if d.get("annotation_data") is not None:
            settings.annotation_data = d["annotation_data"]
        return settings
