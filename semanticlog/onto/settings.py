from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semanticlog.enrich.declare import DeclareResult

DEFAULT_IRI: str = "https://oced#"

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
        declare_file: Path to a pre-computed DECLARE results CSV. Used as a reference
            path alongside declare_result (which holds the loaded data).
        declare_result: Loaded DECLARE constraint results. Required when DECLARE is in
            order_representation. Populate via load_declare_file() or set directly.
        inverse_representation: Inverse relation strategies to include. AXIOMATIC
            adds InverseObjectProperty TBox axioms (OWL2DL only); IMPLICIT
            materialises inverse assertions in the ABox (both profiles). Both are
            enabled by default; use an empty list to omit inverses entirely.
        data_representation: How numeric attribute columns are represented. A list of
            modes; empty list means omit numeric attributes entirely. VALUES and one
            binning mode may coexist; EQUAL_BINS and DIST_BINS are mutually exclusive.
        bins: Number of bins for EQUAL_BINS and DIST_BINS modes.
        bin_labels: Labels for each bin; must have exactly bins elements.
        annotation_document: Path to a JSON file mapping entity names to annotation
            dicts. Each entry may contain "label" and/or "comment" keys. Used as a
            reference path alongside annotation_data (which holds the loaded content).
        annotation_data: Loaded contents of annotation_document. Populated via
            load_annotation_document() or set directly. Used by the builder.
        derive_annotations: If True, emit rdfs:label on every event, Time, and object
            individual using a human-readable label constructed from the source data.
    """

    profile: OwlProfile = OwlProfile.OWL2DL
    order_representation: list[EventOrderRepresentation] = field(default_factory=list)
    ocel_extensions: list[OcelExtension] = field(default_factory=list)
    case_concepts: list[str] = field(default_factory=list)
    declare_file: str | None = None
    declare_result: DeclareResult | None = None
    inverse_representation: list[InverseRepresentation] = field(
        default_factory=lambda: [
            InverseRepresentation.AXIOMATIC,
            InverseRepresentation.IMPLICIT,
        ]
    )
    data_representation: list[DataRepresentation] = field(
        default_factory=lambda: [DataRepresentation.VALUES]
    )
    bins: int = 5
    bin_labels: list[str] = field(
        default_factory=lambda: ["very_low", "low", "average", "high", "very_high"]
    )
    annotation_document: str | None = None
    annotation_data: dict[str, dict[str, str]] | None = None
    derive_annotations: bool = False

    def __post_init__(self) -> None:
        if len(self.bin_labels) != self.bins:
            raise ValueError(
                f"bin_labels must have exactly bins={self.bins} "
                f"elements, got {len(self.bin_labels)}."
            )

    def load_annotation_document(self) -> None:
        """Load annotation_document from disk into annotation_data.

        Raises:
            ValueError: If annotation_document is not set.
        """
        if self.annotation_document is None:
            raise ValueError(
                "annotation_document must be set before calling load_annotation_document()."
            )
        import json

        with open(self.annotation_document) as f:
            self.annotation_data = json.load(f)

    def load_declare_file(self) -> None:
        """Load declare_file from disk into declare_result.

        Raises:
            ValueError: If declare_file is not set.
        """
        if self.declare_file is None:
            raise ValueError(
                "declare_file must be set before calling load_declare_file()."
            )
        from semanticlog.enrich.declare import DeclareResult

        self.declare_result = DeclareResult.read(self.declare_file)

    def to_dict(self) -> dict:
        """Serialise settings to a JSON-compatible dict."""
        return {
            "profile": self.profile.value,
            "order_representation": [e.value for e in self.order_representation],
            "ocel_extensions": [e.value for e in self.ocel_extensions],
            "case_concepts": self.case_concepts,
            "declare_file": self.declare_file,
            "declare_result": (
                self.declare_result.to_dict()
                if self.declare_result is not None
                else None
            ),
            "inverse_representation": [e.value for e in self.inverse_representation],
            "data_representation": [e.value for e in self.data_representation],
            "bins": self.bins,
            "bin_labels": self.bin_labels,
            "annotation_document": self.annotation_document,
            "annotation_data": self.annotation_data,
            "derive_annotations": self.derive_annotations,
        }

    @classmethod
    def from_dict(cls, d: dict) -> OntoBuildSettings:
        """Restore settings from a dict produced by to_dict()."""
        from semanticlog.enrich.declare import DeclareResult

        declare_result = (
            DeclareResult.from_dict(d["declare_result"])
            if d.get("declare_result") is not None
            else None
        )
        return cls(
            profile=OwlProfile(d["profile"]),
            order_representation=[
                EventOrderRepresentation(v) for v in d.get("order_representation", [])
            ],
            ocel_extensions=[OcelExtension(v) for v in d.get("ocel_extensions", [])],
            case_concepts=d.get("case_concepts", []),
            declare_file=d.get("declare_file"),
            declare_result=declare_result,
            inverse_representation=[
                InverseRepresentation(v)
                for v in d.get(
                    "inverse_representation",
                    [InverseRepresentation.AXIOMATIC, InverseRepresentation.IMPLICIT],
                )
            ],
            data_representation=[
                DataRepresentation(v)
                for v in d.get("data_representation", [DataRepresentation.VALUES])
            ],
            bins=d.get("bins", 5),
            bin_labels=d.get(
                "bin_labels", ["very_low", "low", "average", "high", "very_high"]
            ),
            annotation_document=d.get("annotation_document"),
            annotation_data=d.get("annotation_data"),
            derive_annotations=d.get("derive_annotations", False),
        )


def validate_settings(settings: OntoBuildSettings) -> None:
    """Validate OntoBuildSettings and raise ValueError with an informative message on conflict.

    Checks profile compatibility for EF and AXIOMATIC inverses, and mutual exclusivity
    of EQUAL_BINS and DIST_BINS.
    """
    errors: list[str] = []

    if (
        EventOrderRepresentation.EF in settings.order_representation
        and settings.profile == OwlProfile.OWL2EL
    ):
        errors.append(
            "EventOrderRepresentation.EF uses TBox transitivity axioms which are not "
            "supported in OWL2EL. Use EventOrderRepresentation.EF_explicit instead, or "
            "switch to OwlProfile.OWL2DL."
        )

    if (
        InverseRepresentation.AXIOMATIC in settings.inverse_representation
        and settings.profile == OwlProfile.OWL2EL
    ):
        errors.append(
            "InverseRepresentation.AXIOMATIC (InverseObjectProperty TBox axioms) is not "
            "supported in OWL2EL. Remove AXIOMATIC from inverse_representation or switch "
            "to OwlProfile.OWL2DL."
        )

    if (
        DataRepresentation.EQUAL_BINS in settings.data_representation
        and DataRepresentation.DIST_BINS in settings.data_representation
    ):
        errors.append(
            "DataRepresentation.EQUAL_BINS and DIST_BINS are mutually exclusive; "
            "include at most one in data_representation."
        )

    order_needs_cases = {
        EventOrderRepresentation.DF,
        EventOrderRepresentation.EF,
        EventOrderRepresentation.EF_explicit,
    }
    if (
        any(o in settings.order_representation for o in order_needs_cases)
        and not settings.case_concepts
    ):
        errors.append(
            "order_representation includes DF/EF/EF_explicit but case_concepts is empty. "
            "Set settings.case_concepts before calling build_ontology()."
        )

    if (
        EventOrderRepresentation.DECLARE in settings.order_representation
        and settings.declare_result is None
    ):
        errors.append(
            "order_representation includes DECLARE but settings.declare_result is None. "
            "Call settings.load_declare_file() or assign a DeclareResult directly."
        )

    if errors:
        raise ValueError(
            "Invalid OntoBuildSettings:\n" + "\n".join(f"  - {e}" for e in errors)
        )
