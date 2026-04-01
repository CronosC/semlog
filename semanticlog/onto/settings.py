from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

DEFAULT_IRI: str = "https://oced#"

# Classes that define configuration settings for onto instantiation


class OwlProfile(str, Enum):
    """
    Enum specifying the OWL 2 profile to use for ontology instantiation.

    - OWL2DL: Supports full OWL 2 DL semantics
    - OWL2EL: Lightweight profile optimized for performance
    """

    OWL2DL = "OWL2DL"
    OWL2EL = "OWL2EL"


class DataRepresentation(str, Enum):
    """
    Enum specifying how numeric attribute columns are represented in the ontology.

    - NONE: Numeric attributes are omitted entirely — no class, property, or individual.
    - VALUES: One named individual per distinct value; raw number attached via has_data_value.
    - QUANTIZE_SIMPLE: Equal-frequency binning; one individual per bin label, no literal.
    - QUANTIZE_DIST: Distribution-fitted binning; falls back to QUANTIZE_SIMPLE when no
      distribution fits. One individual per bin label, no literal.
    """

    NONE = "none"
    VALUES = "values"
    QUANTIZE_SIMPLE = "quantize_simple"
    QUANTIZE_DIST = "quantize_dist"


class OcelExtension(str, Enum):
    """
    Enum specifying optional OCEL 2.0 features to preserve in the pipeline.

    - ATTRIBUTE_HISTORY: Load and represent object attribute change history from
      pm4py's ``log.object_changes``. Adds ``AttributeChange`` individuals linked
      to their object, attribute value, and timestamp in the ontology.
    - RELATION_TIMESTAMPS: Retain ``ocel:timestamp`` on event-to-object relations
      and represent each timed observation as a reified ``RelationObservation``
      individual linked to the event, object, and timestamp.
    """

    ATTRIBUTE_HISTORY = "attribute_history"
    RELATION_TIMESTAMPS = "relation_timestamps"


class EventOrderRepresentation(str, Enum):
    """
    Enum specifying how event order should be represented in the ontology.

    - DF: Uses a directly-follows (df) object property to model event sequence.
    - EF: DF + eventually-follows (ef) as transitive, reflexive closure of DF.
    - EF_explicit: Makes EF explicit (materialises all ef assertions in the ABox).
    - DECLARE: Uses additional constraint classes or axioms to declare ordering explicitly.
    """

    DF = "df"
    EF = "ef"
    EF_explicit = "ef_explicit"
    DECLARE = "declare"


@dataclass
class OntoBuildSettings:
    """
    Configuration settings for ontology instantiation.

    Attributes:
        profile: The OWL 2 profile to use (e.g. DL or EL).
        order_representation: The strategy for representing event order.
        ocel_extensions: Optional OCEL 2.0 features to enable (e.g. ATTRIBUTE_HISTORY,
            RELATION_TIMESTAMPS). Empty list by default (no extensions active).
        data_representation: How numeric attribute columns are represented.
        quantize_bins: Number of bins for QUANTIZE_SIMPLE and QUANTIZE_DIST modes.
        quantize_labels: Labels for each bin; must have exactly quantize_bins elements.
    """

    profile: OwlProfile = OwlProfile.OWL2DL
    order_representation: list[EventOrderRepresentation] = field(default_factory=list)
    ocel_extensions: list[OcelExtension] = field(default_factory=list)
    case_concepts: list[str] = field(default_factory=list)
    declare_file: Optional[str] = None
    inverse_properties: bool = True
    make_inverses_explicit: bool = True
    data_representation: DataRepresentation = DataRepresentation.VALUES
    quantize_bins: int = 5
    quantize_labels: list[str] = field(
        default_factory=lambda: ["very_low", "low", "average", "high", "very_high"]
    )

    def __post_init__(self) -> None:
        if len(self.quantize_labels) != self.quantize_bins:
            raise ValueError(
                f"quantize_labels must have exactly quantize_bins={self.quantize_bins} "
                f"elements, got {len(self.quantize_labels)}."
            )
