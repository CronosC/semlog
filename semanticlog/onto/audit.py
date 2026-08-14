"""The checks a build runs before it emits anything.

Most refuse a store the emitters could not turn into a sound ontology: a case
concept the log cannot be projected onto, DECLARE constraints whose traces
this log does not hold, a relation row naming an entity that is not there, or
two attributes whose values would become one individual. Running them up front
means a build that cannot succeed fails before it has written half an ontology,
with a message about the input.

`warn_on_foreign_declare` is the exception. It answers the same question as
`check_declare_input`, from the checksum rather than from the store, and only
warns. It takes the digest of the input, so a caller that has one runs it.
"""

from __future__ import annotations

import logging

from semanticlog.core import vocabulary as vocab
from semanticlog.data.cleaning import object_type_renames
from semanticlog.data.flatten import case_concept_candidates
from semanticlog.data.integrity import require_integrity
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind
from semanticlog.onto.emitters.attributes import attribute_owners
from semanticlog.onto.settings import OntoBuildSettings

logger = logging.getLogger(__name__)


def audit_build(data: OCEDStore, settings: OntoBuildSettings) -> None:
    """Run every applicable check, stopping at the first failure.

    The settings decide whether the checks that only apply to some
    representations do anything.

    Raises:
        ValueError: If a check rejects the store.
    """
    check_case_concepts(data, settings)
    require_integrity(data)
    check_declare_input(data, settings)
    check_converging_values(data)


def check_case_concepts(data: OCEDStore, settings: OntoBuildSettings) -> None:
    """Check that every case concept is one this log can be projected onto.

    Flattening looks each case concept up in the store, which holds names
    as cleaning left them, and then needs an event-to-object relation to
    reach it. Either failure would otherwise surface only once the build
    had emitted its event data.

    Args:
        data: Store being built from.
        settings: Settings holding the case concepts.

    Raises:
        ValueError: If a case concept is not an object type of this log, or
            is one no event-to-object relation reaches.
    """
    if not settings.emits_traces:
        return

    object_types = set(data.get_object_types())
    candidates = case_concept_candidates(data)
    reachable = set(candidates)
    unknown = [c for c in settings.case_concepts if c not in object_types]
    unreachable = [
        c for c in settings.case_concepts if c in object_types and c not in reachable
    ]
    if not unknown and not unreachable:
        return

    renames = object_type_renames(data)
    problems = []
    for case_concept in unknown:
        moved = renames.get(case_concept)
        if moved is None:
            problems.append(f"  {case_concept!r} is not an object type of this log")
        else:
            problems.append(
                f"  {case_concept!r} was renamed to {moved!r} by cleaning, so "
                "that is what it is called here"
            )
    for case_concept in unreachable:
        problems.append(
            f"  {case_concept!r} is an object type of this log, but no "
            "event-to-object relation reaches it, so it has no events to order"
        )

    raise ValueError(
        "The case concepts do not name object types this log can be "
        "projected onto:\n"
        + "\n".join(problems)
        + f"\n\nThis log can be projected onto: {', '.join(sorted(reachable))}. "
        "Set settings.case_concepts to types it holds. 'semlog inspect' lists "
        "them along with what each one was called in the source."
    )


def check_declare_input(data: OCEDStore, settings: OntoBuildSettings) -> None:
    """Check that the DECLARE constraints belong to the log being built.

    Constraints are mined in a separate step and are easy to pair with the
    wrong log, in which case the trace ids would name objects this log
    does not hold and the membership assertions would be about nothing.
    This runs before the build so the failure names the mismatch rather
    than surfacing as an odd ontology.

    Args:
        data: Store being built from.
        settings: Settings holding the mined constraints.

    Raises:
        ValueError: If the constraints have no trace column, name a case
            concept this log does not have, or name trace ids that are not
            objects of that type.
    """
    if not settings.emits_declare:
        return
    declare_result = settings.declare_result
    if declare_result is None:
        return

    constraints = declare_result.constraints
    if "trace_id" not in constraints.columns:
        raise ValueError(
            "DECLARE constraints have no 'trace_id' column, so there is "
            "nothing to type as a member of each constraint class."
        )

    object_types = set(data.get_object_types())
    problems: list[str] = []
    for case_concept, group in constraints.groupby("case_concept"):
        case_concept = str(case_concept)
        if case_concept not in object_types:
            problems.append(
                f"  case concept {case_concept!r} is not an object type of "
                f"this log (it has {', '.join(sorted(object_types))})"
            )
            continue
        known = set(map(str, data.get_ids(EntityKind.OBJECT, case_concept)))
        unknown = sorted(set(group["trace_id"].astype(str)) - known)
        if unknown:
            problems.append(
                f"  {len(unknown)} of {group['trace_id'].nunique()} "
                f"{case_concept!r} trace id(s) are not objects of that type, "
                f"starting with {', '.join(unknown[:5])}"
            )

    if problems:
        origin = settings.declare_file or "the assigned DECLARE result"
        raise ValueError(
            f"The DECLARE constraints in {origin} do not match this log:\n"
            + "\n".join(problems)
            + "\n\nThey were most likely mined from a different log, or from "
            "this one before it was filtered. Re-run 'semlog mine-declare' "
            "against the log being built."
        )


def warn_on_foreign_declare(settings: OntoBuildSettings, digest: str) -> None:
    """Warn when the DECLARE constraints were mined from a different file.

    The build continues, because the trace ids still resolve and a filtered
    copy of a log is a reasonable thing to mine from. It is worth saying so.
    """
    result = settings.declare_result
    if result is None or result.source_sha256 is None:
        return
    if result.source_sha256 != digest:
        logger.warning(
            f"DECLARE constraints were mined from {result.source_log!r} "
            f"(sha256 {result.source_sha256[:12]}), which is not the file being "
            f"built (sha256 {digest[:12]}). The trace ids still resolve, so the "
            "build continues."
        )


def check_converging_values(data: OCEDStore) -> None:
    """Refuse attribute names that would produce the same value individual.

    A value individual is named `attribute_value`, and an underscore is
    legal in both halves, so `a` with value `b_c` and `a_b` with value `c`
    would both be `a_b_c`. Only attribute pairs that differ by an
    underscore can collide this way, so only those are compared.

    Args:
        data: Store to check.

    Raises:
        NameCollisionError: If two attributes have a value in common under
            this naming, quoting one example.
    """
    owners = attribute_owners(data)
    candidates = [
        (shorter, longer)
        for shorter in owners
        for longer in owners
        if shorter != longer and longer.startswith(f"{shorter}_")
    ]
    if not candidates:
        return

    values = {
        attribute: _distinct_values(data, attribute)
        for attribute in owners
        if any(attribute in pair for pair in candidates)
    }
    for shorter, longer in candidates:
        shared = {
            vocab.attribute_value(shorter, v) for v in values.get(shorter, ())
        } & {vocab.attribute_value(longer, v) for v in values.get(longer, ())}
        if shared:
            example = sorted(shared)[0]
            raise vocab.NameCollisionError(
                f"{example!r} would be emitted for a value of attribute "
                f"{shorter!r} and for a value of attribute {longer!r}, which "
                "makes one individual the value of both:\n"
                f"  {shorter!r} and {longer!r} differ only by an underscore, "
                "which is also legal inside a value.\n\n"
                "Rename one of the two columns in the input data before building."
            )


def _distinct_values(data: OCEDStore, attribute: str) -> set[object]:
    """Collect the distinct values an attribute takes across every table holding it."""
    found: set[object] = set()
    for view in data.tables():
        if attribute in view.attributes:
            found |= set(view.frame[attribute].dropna().unique())
    return found
