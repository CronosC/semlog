"""The checks a build runs before it emits anything.

Most refuse a store the emitters could not turn into a sound ontology: a case
concept the log cannot be projected onto, DECLARE constraints whose traces
this log does not hold, a relation row naming an entity that is not there, or
two attributes whose values would become one individual. Running them up front
means a build that cannot succeed fails before it has written half an ontology,
with a message about the input.

Every check here reads the store. Whether a set of constraints came from the
file being built is answered from the checksum instead, which is the caller's
to hold, so that check lives with the caller.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from semanticlog.core import vocabulary as vocab
from semanticlog.core.errors import ConfigError, InputError
from semanticlog.data.cleaning import object_type_renames
from semanticlog.data.flatten import case_concept_candidates
from semanticlog.data.integrity import require_integrity
from semanticlog.data.oced import attribute_owners
from semanticlog.data.readable import ReadableStore
from semanticlog.data.tables import EntityKind
from semanticlog.onto.inputs import BuildInputs
from semanticlog.onto.settings import OntoBuildSettings

logger = logging.getLogger(__name__)


def audit_build(
    data: ReadableStore, settings: OntoBuildSettings, inputs: BuildInputs
) -> None:
    """Run every applicable check, stopping at the first failure.

    The settings decide whether the checks that only apply to some
    representations do anything.

    Raises:
        ConfigError: If the case concepts do not name types this log holds.
        InputError: If a check rejects the store.
    """
    check_case_concepts(data, settings, inputs.case_concepts)
    require_integrity(data)
    check_declare_input(data, settings, inputs.declare_constraints)
    check_converging_values(data)


def check_case_concepts(
    data: ReadableStore, settings: OntoBuildSettings, case_concepts: Sequence[str]
) -> None:
    """Check that every case concept is one this log can be projected onto.

    Flattening looks each case concept up in the store, which holds names
    as cleaning left them, and then needs an event-to-object relation to
    reach it. Either failure would otherwise surface only once the build
    had emitted its event data.

    Args:
        data: Store being built from.
        settings: Settings, for whether any of this applies.
        case_concepts: Object types the build will project onto.

    Raises:
        ConfigError: If a case concept is not an object type of this log, or
            is one no event-to-object relation reaches.
    """
    if not settings.emits_traces:
        return

    object_types = set(data.get_object_types())
    candidates = case_concept_candidates(data)
    reachable = set(candidates)
    unknown = [c for c in case_concepts if c not in object_types]
    unreachable = [c for c in case_concepts if c in object_types and c not in reachable]
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

    raise ConfigError(
        "The case concepts do not name object types this log can be "
        "projected onto:\n"
        + "\n".join(problems)
        + f"\n\nThis log can be projected onto: {', '.join(sorted(reachable))}. "
        "Set the case concepts to types it holds. 'semlog inspect' lists "
        "them along with what each one was called in the source."
    )


def check_declare_input(
    data: ReadableStore, settings: OntoBuildSettings, constraints: pd.DataFrame | None
) -> None:
    """Check that the DECLARE constraints belong to the log being built.

    Constraints are mined in a separate step and are easy to pair with the
    wrong log, in which case the trace ids would name objects this log
    does not hold and the membership assertions would be about nothing.
    This runs before the build so the failure names the mismatch rather
    than surfacing as an odd ontology.

    Args:
        data: Store being built from.
        settings: Settings, for whether any of this applies.
        constraints: The mined constraints, one row per trace.

    Raises:
        InputError: If the constraints have no trace column, name a case
            concept this log does not have, or name trace ids that are not
            objects of that type.
    """
    if not settings.emits_declare or constraints is None:
        return

    if "trace_id" not in constraints.columns:
        raise InputError(
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
        raise InputError(
            "The DECLARE constraints do not match this log:\n"
            + "\n".join(problems)
            + "\n\nThey were most likely mined from a different log, or from "
            "this one before it was filtered. Re-run 'semlog mine-declare' "
            "against the log being built."
        )


def check_converging_values(data: ReadableStore) -> None:
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


def _distinct_values(data: ReadableStore, attribute: str) -> set[object]:
    """Collect the distinct values an attribute takes across every table holding it."""
    found: set[object] = set()
    for view in data.tables():
        if attribute in view.attributes:
            found |= set(view.frame[attribute].dropna().unique())
    return found
