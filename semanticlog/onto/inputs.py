"""What a build needs beyond its settings and the log itself.

These are the three things a build cannot be configured with in advance,
because each is read off the log or out of a file: which object types act as
cases, which traces satisfy which mined DECLARE constraint, and what to label
the emitted names. A caller resolves them and hands them to `build_ontology`.

The DECLARE constraints arrive as the table itself rather than as a
`DeclareResult`, so nothing here depends on how they were mined.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from semanticlog.core.errors import ConfigError
from semanticlog.onto.settings import OntoBuildSettings

# The columns of a constraints table that say which trace a row is about. Every
# other column is one constraint.
CONSTRAINT_INDEX_COLUMNS: tuple[str, ...] = ("case_concept", "trace_id")


@dataclass(frozen=True)
class BuildInputs:
    """The inputs of one build, resolved against one log.

    Attributes:
        case_concepts: Object types whose instances act as cases. Every order
            representation but DECLARE needs at least one.
        declare_constraints: One row per trace, with `case_concept` and
            `trace_id` saying which trace and every other column naming one
            constraint. None when the build emits no DECLARE membership.
        annotations: Entity name to its label and comment. Applied last, once
            every name it could refer to exists.
    """

    case_concepts: tuple[str, ...] = ()
    declare_constraints: pd.DataFrame | None = None
    annotations: dict[str, dict[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.case_concepts, tuple):
            object.__setattr__(self, "case_concepts", tuple(self.case_concepts))

    def validate(self, settings: OntoBuildSettings) -> None:
        """Check that these inputs carry what the settings ask for.

        Every problem is collected before raising, so one call reports all of
        them.

        Args:
            settings: Settings of the build these inputs serve.

        Raises:
            ConfigError: If a representation the settings select has nothing
                here to work from, listing every such problem.
        """
        errors: list[str] = []

        if settings.emits_traces and not self.case_concepts:
            errors.append(
                "order_representations includes DF/EF/EF_EXPLICIT but no case "
                "concepts were given, and a trace is the projection of the log "
                "onto one object type. Name at least one in BuildInputs."
            )

        if settings.emits_declare and self.declare_constraints is None:
            errors.append(
                "order_representations includes DECLARE but no constraints were "
                "given. Mine them with 'semlog mine-declare' and pass the table "
                "as BuildInputs.declare_constraints."
            )

        if errors:
            raise ConfigError(
                "Invalid BuildInputs:\n" + "\n".join(f"  - {e}" for e in errors)
            )


def constraint_names(constraints: pd.DataFrame) -> list[str]:
    """Return the constraint columns of a table, without the ones that index it."""
    return [c for c in constraints.columns if c not in CONSTRAINT_INDEX_COLUMNS]
