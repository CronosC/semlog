from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum


# Classes that define configuration settings for onto instantiation

class OwlProfile(str, Enum):
    """
    Enum specifying the OWL 2 profile to use for ontology instantiation.

    - OWL2DL: Supports full OWL 2 DL semantics, enabling expressivity with decidability.
    - OWL2EL: Lightweight profile optimized for performance, typically used in large-scale ontologies.
    """
    OWL2DL = "OWL2DL"
    OWL2EL = "OWL2EL"

class EventOrderRepresentation(str, Enum):
    """
    Enum specifying how event order should be represented in the ontology.

    - IMPLICIT: No explicit ordering; relies on timestamps or implicit sequence.
    - DF: Uses a directly-follows (df) object property to model event sequence.
    - DECLARE: Uses additional constraint classes or axioms to declare ordering explicitly.
    """
    IMPLICIT = "implicit"
    DF = "df"
    DECLARE = "declare"

@dataclass
class OntoBuildSettings:
    """
    Configuration settings for ontology instantiation.

    Attributes:
        profile: The OWL 2 profile to use (e.g. DL or EL).
        order_representation: The strategy for representing event order.
    """
    profile: OwlProfile = OwlProfile.OWL2DL
    order_representation: EventOrderRepresentation = EventOrderRepresentation.IMPLICIT
    case_concepts: list[str] = field(default_factory=list)
    declare_file: Optional[str] = None
