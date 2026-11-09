"""The inputs a build reads from the log or from a file.

`BuildInputs` holds the object types that act as cases, the DECLARE
constraints each trace satisfies, and the labels for the emitted names. The
caller resolves them and passes them to `build_ontology`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from semanticlog.core.errors import ConfigError
from semanticlog.data.declare import DeclareResult
from semanticlog.onto.settings import OntoBuildSettings


@dataclass(frozen=True)
class BuildInputs:
    """The inputs of one build, resolved against one log.

    Attributes:
        case_concepts: Object types whose instances act as cases. Every order
            representation but DECLARE needs at least one.
        declare: The mined DECLARE constraints. None when the build emits no
            DECLARE membership.
        annotations: Entity name to its label and comment. Applied after
            every name has been declared.
    """

    case_concepts: tuple[str, ...] = ()
    declare: DeclareResult | None = None
    annotations: dict[str, dict[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.case_concepts, str):
            raise ConfigError(
                "The setting case_concepts must be a list, got the string "
                f"{self.case_concepts!r}. Pass [{self.case_concepts!r}]."
            )
        if not isinstance(self.case_concepts, tuple):
            object.__setattr__(self, "case_concepts", tuple(self.case_concepts))

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

        if settings.emits_declare and self.declare is None:
            errors.append(
                "order_representations includes declare but no constraints were "
                "given. Mine them with 'semlog mine-declare' and set "
                "declare_file to the file it writes."
            )

        if errors:
            raise ConfigError("The build inputs are incomplete: " + " ".join(errors))
