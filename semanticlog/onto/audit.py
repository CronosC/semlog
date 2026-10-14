"""The checks a build runs before it emits anything.

`audit_build` rejects a case concept the log cannot be projected onto, DECLARE
constraints whose traces are not in this log, and a relation row naming an
entity the store does not hold. A failed check raises before any file is
written.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from semanticlog.core.checks import CheckReport, Fixed, raise_reports
from semanticlog.data.cleaning import object_type_renames
from semanticlog.data.flatten import case_concept_candidates
from semanticlog.data.integrity import check_integrity
from semanticlog.data.readable import ReadableStore
from semanticlog.data.tables import CONSTRAINT_INDEX_COLUMNS, EntityKind
from semanticlog.onto.inputs import BuildInputs
from semanticlog.onto.settings import OntoBuildSettings

logger = logging.getLogger(__name__)


def audit_build(
    data: ReadableStore, settings: OntoBuildSettings, inputs: BuildInputs
) -> None:
    """Run every check that applies to the settings.

    Raises:
        ConfigError: If only the case concepts are wrong. The message lists
            every finding.
        InputError: If anything is wrong with the log or the constraints. The
            message lists every finding.
    """
    raise_reports(
        [
            check_case_concepts(data, settings, inputs.case_concepts),
            check_integrity(data),
            check_declare_input(data, settings, inputs.declare_constraints),
        ]
    )


def check_case_concepts(
    data: ReadableStore, settings: OntoBuildSettings, case_concepts: Sequence[str]
) -> CheckReport | None:
    """Check that every case concept is an object type with an event-to-object relation.

    Names are compared as cleaning left them.

    Args:
        data: Store being built from.
        settings: Settings. The check applies only when traces are emitted.
        case_concepts: Object types the build will project onto.

    Returns:
        Every case concept the log cannot be projected onto, or None when the
        check passes or does not apply.
    """
    if not settings.emits_traces:
        return None

    object_types = set(data.types(EntityKind.OBJECT))
    candidates = case_concept_candidates(data)
    reachable = set(candidates)
    unknown = [c for c in case_concepts if c not in object_types]
    unreachable = [c for c in case_concepts if c in object_types and c not in reachable]
    if not unknown and not unreachable:
        return None

    renames = object_type_renames(data)
    problems = []
    for case_concept in unknown:
        moved = renames.get(case_concept)
        if moved is None:
            problems.append(f"{case_concept!r} is not an object type of this log")
        else:
            problems.append(
                f"{case_concept!r} was renamed to {moved!r} by cleaning. Use that name"
            )
    for case_concept in unreachable:
        problems.append(
            f"{case_concept!r} is an object type of this log, but no "
            "event-to-object relation reaches it"
        )

    return CheckReport(
        header=(
            "The case concepts do not name object types this log can be projected onto:"
        ),
        problems=tuple(problems),
        advice=(
            f"This log can be projected onto: {', '.join(sorted(reachable))}. "
            "Set the case concepts to types it holds. 'semlog inspect' lists "
            "them along with what each one was called in the source."
        ),
        about=Fixed.CONFIG,
    )


def check_declare_input(
    data: ReadableStore, settings: OntoBuildSettings, constraints: pd.DataFrame | None
) -> CheckReport | None:
    """Check that every trace id in the DECLARE constraints is an object of its case concept.

    Args:
        data: Store being built from.
        settings: Settings. The check applies only when DECLARE is emitted.
        constraints: The mined constraints, one row per trace.

    Returns:
        Everything about the constraints that does not match this log, or None
        when the check passes or does not apply.
    """
    if not settings.emits_declare or constraints is None:
        return None

    absent = [c for c in CONSTRAINT_INDEX_COLUMNS if c not in constraints.columns]
    if absent:
        return CheckReport(
            header=(
                f"DECLARE constraints have no "
                f"{' or '.join(repr(c) for c in absent)} column. Its columns "
                f"are {', '.join(map(str, constraints.columns))}. "
                "'semlog mine-declare' writes both."
            ),
            about=Fixed.INPUT,
        )

    object_types = set(data.types(EntityKind.OBJECT))
    problems: list[str] = []
    unnamed = int(constraints["case_concept"].isna().sum())
    if unnamed:
        problems.append(f"{unnamed} row(s) have no case concept")
    for case_concept, group in constraints.groupby("case_concept"):
        case_concept = str(case_concept)
        if case_concept not in object_types:
            problems.append(
                f"case concept {case_concept!r} is not an object type of "
                f"this log (it has {', '.join(sorted(object_types))})"
            )
            continue
        known = set(map(str, data.ids(EntityKind.OBJECT, case_concept)))
        unknown = sorted(set(group["trace_id"].astype(str)) - known)
        if unknown:
            problems.append(
                f"{len(unknown)} of {group['trace_id'].nunique()} "
                f"{case_concept!r} trace id(s) are not objects of that type, "
                f"starting with {', '.join(unknown[:5])}"
            )

    if not problems:
        return None
    return CheckReport(
        header="The DECLARE constraints do not match this log:",
        problems=tuple(problems),
        advice=(
            "They were most likely mined from a different log, or from this "
            "one before it was filtered. Re-run 'semlog mine-declare' against "
            "the log being built."
        ),
        about=Fixed.INPUT,
    )
