"""The build configurations the matrix covers, and the named builds tests ask for.

`MATRIX` holds the configurations every whole-build test runs over. Adding an entry adds a row to the shape table in `tests/output/test_build_shape.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd

from semanticlog.core.enums import (
    DataRepresentation,
    EventOrderRepresentation,
    InverseRepresentation,
    OCELExtension,
    OWLProfile,
)
from semanticlog.onto.inputs import BuildInputs
from semanticlog.onto.settings import OntoBuildSettings


def declare_constraints() -> pd.DataFrame:
    """Constraints over one case concept, for the DECLARE order representation."""
    return pd.DataFrame(
        {
            "trace_id": ["o1", "o2"],
            "case_concept": ["order", "order"],
            "Response__place_order__ship__": [True, False],
        }
    )


def ragged_declare_constraints() -> pd.DataFrame:
    """Constraints over two case concepts, where neither holds of the other's traces."""
    return pd.DataFrame(
        {
            "trace_id": ["o1", "o2", "i1", "i2"],
            "case_concept": ["order", "order", "item", "item"],
            "Response__place_order__ship__": [True, False, np.nan, np.nan],
            "Existence__ship__": [np.nan, np.nan, True, False],
        }
    )


@dataclass(frozen=True)
class Configuration:
    """One build of the matrix: its settings and the inputs they need."""

    settings: OntoBuildSettings
    inputs: BuildInputs = field(default_factory=BuildInputs)


def configured(spec: Configuration | OntoBuildSettings) -> Configuration:
    """Return a configuration, wrapping bare settings in the default inputs."""
    if isinstance(spec, Configuration):
        return spec
    return Configuration(spec)


_INPUT_FIELDS = {f.name for f in fields(BuildInputs)}


def configuration(**named) -> Configuration:
    """Return a configuration from settings and input fields, told apart by name."""
    inputs = {k: v for k, v in named.items() if k in _INPUT_FIELDS}
    rest = {k: v for k, v in named.items() if k not in _INPUT_FIELDS}
    return Configuration(OntoBuildSettings(**rest), BuildInputs(**inputs))


def settings(**overrides) -> Configuration:
    """Return one configuration for the matrix store, with fields changed."""
    base: dict = {
        "profile": OWLProfile.OWL2DL,
        "order_representations": [EventOrderRepresentation.DF],
        "case_concepts": ["order"],
    }
    base.update(overrides)
    return configuration(**base)


# The matrix: every value of every setting appears in at least one entry, and
# the entries pair values that meet in one emitter.
MATRIX: dict[str, Callable[[], Configuration]] = {
    "dl_df": lambda: settings(),
    "dl_inverses": lambda: settings(
        inverse_representations=[
            InverseRepresentation.AXIOMATIC,
            InverseRepresentation.IMPLICIT,
        ],
        derive_annotations=True,
    ),
    "el_ef_explicit_bins": lambda: settings(
        profile=OWLProfile.OWL2EL,
        order_representations=[EventOrderRepresentation.EF_EXPLICIT],
        inverse_representations=[InverseRepresentation.IMPLICIT],
        data_representations=[DataRepresentation.VALUES, DataRepresentation.EQUAL_BINS],
        bins=2,
        bin_labels=["low", "high"],
        derive_annotations=True,
    ),
    "dl_ef_dist_bins_history": lambda: settings(
        order_representations=[
            EventOrderRepresentation.DF,
            EventOrderRepresentation.EF,
            EventOrderRepresentation.EF_EXPLICIT,
        ],
        data_representations=[DataRepresentation.VALUES, DataRepresentation.DIST_BINS],
        ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY],
    ),
    "el_history_bins_only": lambda: settings(
        profile=OWLProfile.OWL2EL,
        ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY],
        data_representations=[DataRepresentation.EQUAL_BINS],
        bins=2,
        bin_labels=["low", "high"],
        derive_annotations=True,
    ),
    "dl_declare_multi_cc": lambda: settings(
        case_concepts=["order", "item"],
        order_representations=[
            EventOrderRepresentation.DF,
            EventOrderRepresentation.DECLARE,
        ],
        declare_constraints=ragged_declare_constraints(),
        inverse_representations=[InverseRepresentation.IMPLICIT],
        data_representations=[],
        derive_annotations=True,
    ),
    "dl_no_case_concept": lambda: settings(
        case_concepts=[],
        order_representations=[],
        inverse_representations=[InverseRepresentation.AXIOMATIC],
        ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY],
    ),
}

# Every named build: the matrix and the builds outside it a test asks for by name.
CONFIGURATIONS: dict[str, Callable[[], Configuration]] = {
    **MATRIX,
    "dl_df_ef": lambda: settings(
        order_representations=[EventOrderRepresentation.DF, EventOrderRepresentation.EF]
    ),
    "dl_ef_explicit": lambda: settings(
        order_representations=[
            EventOrderRepresentation.DF,
            EventOrderRepresentation.EF_EXPLICIT,
        ]
    ),
    "dl_history": lambda: settings(
        ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY],
        derive_annotations=True,
    ),
}

NAMES = sorted(MATRIX)

EL_NAMES = sorted(
    name
    for name, factory in MATRIX.items()
    if factory().settings.profile is OWLProfile.OWL2EL
)
