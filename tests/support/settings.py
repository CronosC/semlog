"""The build configurations the matrix covers, and the named builds tests ask for.

`MATRIX` holds the configurations every whole-build test runs over. Adding an entry adds a row to the shape table in `tests/output/test_build_shape.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields

from semanticlog.core.enums import (
    DataRepresentation,
    EventOrderRepresentation,
    InverseRepresentation,
    OCELExtension,
    OWLProfile,
)
from semanticlog.data.declare import ConceptConstraints, DeclareResult, MiningSettings
from semanticlog.onto.inputs import BuildInputs
from semanticlog.onto.settings import OntoBuildSettings

DISCOVERY = MiningSettings("discovery", ("Response",), 0.1, 0.9)


def declare_constraints(
    constraint: str = "Response__place_order__ship__",
) -> DeclareResult:
    """One constraint over the orders o1 and o2, which o1 satisfies."""
    return DeclareResult(
        concepts=(
            ConceptConstraints("order", ("o1", "o2"), {constraint: frozenset({"o1"})}),
        ),
        mining=DISCOVERY,
    )


def ragged_declare_constraints() -> DeclareResult:
    """Constraints over two case concepts, each mined for one of them only."""
    return DeclareResult(
        concepts=(
            ConceptConstraints(
                "order",
                ("o1", "o2"),
                {"Response__place_order__ship__": frozenset({"o1"})},
            ),
            ConceptConstraints(
                "item", ("i1", "i2"), {"Existence__ship__": frozenset({"i1"})}
            ),
        ),
        mining=DISCOVERY,
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
        declare=ragged_declare_constraints(),
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
