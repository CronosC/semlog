"""The inputs a build reads from the log or from a file.

`BuildInputs` holds the object types that act as cases, the DECLARE
constraints each trace satisfies, and the labels for the emitted names. The
caller resolves them and passes them to `build_ontology`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from semanticlog.core.errors import ConfigError
from semanticlog.data.tables import CONSTRAINT_INDEX_COLUMNS
from semanticlog.onto.settings import OntoBuildSettings


@dataclass(frozen=True)
class BuildInputs:
    """The inputs of one build, resolved against one log.

    Attributes:
        case_concepts: Object types whose instances act as cases. Every order
            representation but DECLARE needs at least one.
        declare_constraints: One row per trace, with `case_concept` and
            `trace_id` saying which trace and every other column naming one
            constraint. None when the build emits no DECLARE membership.
        annotations: Entity name to its label and comment. Applied after
            every name has been declared.
    """

    case_concepts: tuple[str, ...] = ()
    declare_constraints: pd.DataFrame | None = None
    annotations: dict[str, dict[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.case_concepts, str):
            raise ConfigError(
                "case_concepts must be a list, got the string "
                f"{self.case_concepts!r}. Pass [{self.case_concepts!r}]."
            )
        if not isinstance(self.case_concepts, tuple):
            object.__setattr__(self, "case_concepts", tuple(self.case_concepts))
        if self.declare_constraints is not None:
            # A copy, so a caller holding the original cannot change frozen inputs.
            object.__setattr__(
                self, "declare_constraints", self.declare_constraints.copy()
            )

    def validate(self, settings: OntoBuildSettings) -> None:
        """Check that these inputs hold what the settings need.

        Raises:
            ConfigError: If a selected representation has no input here. The
                message lists every problem.
        """
        errors: list[str] = []

        if settings.emits_traces and not self.case_concepts:
            errors.append(
                "order_representations includes df, ef or ef_explicit but no "
                "case concepts were given. Name at least one object type in "
                "case_concepts."
            )

        if settings.emits_declare and self.declare_constraints is None:
            errors.append(
                "order_representations includes declare but no constraints were "
                "given. Mine them with 'semlog mine-declare' and set "
                "declare_file to the table it writes."
            )

        if errors:
            raise ConfigError(
                "Invalid BuildInputs:\n" + "\n".join(f"  - {e}" for e in errors)
            )


def constraint_names(constraints: pd.DataFrame) -> list[str]:
    """Return the constraint columns of a table, without the ones that index it."""
    return [c for c in constraints.columns if c not in CONSTRAINT_INDEX_COLUMNS]
