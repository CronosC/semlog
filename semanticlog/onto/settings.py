from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum


# Classes that define configuration settings for onto instantiation

class OwlProfile(str, Enum):
    """
    Enum specifying the OWL 2 profile to use for ontology instantiation.

    - OWL2DL: Supports full OWL 2 DL semantics
    - OWL2EL: Lightweight profile optimized for performance
    """
    OWL2DL = "OWL2DL"
    OWL2EL = "OWL2EL"

class EventOrderRepresentation(str, Enum):
    """
    Enum specifying how event order should be represented in the ontology.

    - DF: Uses a directly-follows (df) object property to model event sequence.
    - EF: DF + eventually-follows (ef) as transitive, reflexive closure of DF.
    - EF: Makes EF explicit.
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
    """
    profile: OwlProfile = OwlProfile.OWL2DL
    order_representation: list[EventOrderRepresentation] = field(default_factory=list)
    case_concepts: list[str] = field(default_factory=list)
    declare_file: Optional[str] = None
    inverse_properties: bool = True
    make_inverses_explicit: bool = True
