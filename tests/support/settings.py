"""The build configurations the matrix covers.

One entry per combination worth checking. Adding an entry adds a row to the
shape table in `tests/onto/test_build_shape.py` and a case to every test that
runs over the matrix, so a new representation is covered everywhere by naming
it once here.
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
    """One build of the matrix: its settings and the inputs they need.

    A build takes both, and which of the two a value belongs to is not always
    obvious from the name, so the matrix names them together.
    """

    settings: OntoBuildSettings
    inputs: BuildInputs = field(default_factory=BuildInputs)


def configured(spec: Configuration | OntoBuildSettings) -> Configuration:
    """Return a configuration, wrapping bare settings in the default inputs."""
    if isinstance(spec, Configuration):
        return spec
    return Configuration(spec)


_INPUT_FIELDS = {f.name for f in fields(BuildInputs)}


def configuration(**named) -> Configuration:
    """Return a configuration from fields of either half, told apart by name.

    A test names what it wants to change without having to know whether the
    build reads it from the settings or from the inputs.
    """
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


CONFIGURATIONS: dict[str, Callable[[], Configuration]] = {
    "dl_df": lambda: settings(),
    "dl_df_ef": lambda: settings(
        order_representations=[EventOrderRepresentation.DF, EventOrderRepresentation.EF]
    ),
    "dl_ef_explicit": lambda: settings(
        order_representations=[
            EventOrderRepresentation.DF,
            EventOrderRepresentation.EF_EXPLICIT,
        ]
    ),
    "el_df": lambda: settings(profile=OWLProfile.OWL2EL),
    "el_ef_explicit": lambda: settings(
        profile=OWLProfile.OWL2EL,
        order_representations=[EventOrderRepresentation.EF_EXPLICIT],
    ),
    "dl_inverses": lambda: settings(
        inverse_representations=[
            InverseRepresentation.AXIOMATIC,
            InverseRepresentation.IMPLICIT,
        ],
        derive_annotations=True,
    ),
    "dl_bins": lambda: settings(
        data_representations=[DataRepresentation.VALUES, DataRepresentation.EQUAL_BINS],
        bins=2,
        bin_labels=["low", "high"],
    ),
    "dl_bins_only": lambda: settings(
        data_representations=[DataRepresentation.EQUAL_BINS],
        bins=2,
        bin_labels=["low", "high"],
    ),
    "dl_dist_bins": lambda: settings(
        data_representations=[DataRepresentation.VALUES, DataRepresentation.DIST_BINS],
    ),
    "el_inverses": lambda: settings(
        profile=OWLProfile.OWL2EL,
        inverse_representations=[InverseRepresentation.IMPLICIT],
    ),
    "dl_no_case_concept": lambda: settings(case_concepts=[], order_representations=[]),
    "dl_no_numeric": lambda: settings(data_representations=[]),
    "dl_history": lambda: settings(
        ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY],
        derive_annotations=True,
    ),
    # Binning and history together: a changed number has to reach the same
    # bin as the same number on the object itself.
    "dl_history_bins": lambda: settings(
        ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY],
        data_representations=[DataRepresentation.EQUAL_BINS],
        bins=2,
        bin_labels=["low", "high"],
    ),
    "dl_declare": lambda: settings(
        order_representations=[
            EventOrderRepresentation.DF,
            EventOrderRepresentation.DECLARE,
        ],
        declare_constraints=declare_constraints(),
    ),
    "dl_declare_multi_cc": lambda: settings(
        case_concepts=["order", "item"],
        order_representations=[
            EventOrderRepresentation.DF,
            EventOrderRepresentation.DECLARE,
        ],
        declare_constraints=ragged_declare_constraints(),
    ),
}

NAMES = sorted(CONFIGURATIONS)

EL_NAMES = sorted(
    name
    for name, factory in CONFIGURATIONS.items()
    if factory().settings.profile is OWLProfile.OWL2EL
)
