"""Projection of object-centric data onto one case concept.

`get_log` and `iter_logs` turn each object of a chosen type into a case
holding every event related to it. `choose_case_concepts` picks the types.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Sequence

import pandas as pd

from semanticlog.core.checks import first_five
from semanticlog.core.errors import InputError
from semanticlog.core.summary import count_phrase
from semanticlog.data.cleaning import object_type_renames
from semanticlog.data.readable import ReadableStore
from semanticlog.data.tables import EntityKind, RelationKind

logger = logging.getLogger(__name__)

_POSITION = "semlog:position"


def get_log(data: ReadableStore, case_concept: str) -> pd.DataFrame:
    """Return the flat log for one case concept.

    Each object of the case concept becomes a case. A repeated event-object
    pair is kept once. An event with no timestamp is left out and logged.
    Events of one case with the same timestamp keep the order of their
    positions in the log. Events of an activity without positions follow
    those with one, in the order the store holds them.

    Returns:
        A frame indexed by event id, with the PM4Py columns
        `case:concept:name`, `concept:name` and `time:timestamp`, sorted by
        case, time and position in the log.

    Raises:
        InputError: If the type is unknown, no relation reaches it, the store
            holds no events, or its activities mix zoned and unzoned
            timestamps.
    """
    return next(frame for _, frame in iter_logs(data, [case_concept]))


def iter_logs(
    data: ReadableStore, case_concepts: Iterable[str]
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield each case concept with its flat log.

    Yields:
        Pairs of case concept and its flat log, each as `get_log` returns it.

    Raises:
        InputError: If a type is unknown, no relation reaches it, the store
            holds no events, or its activities mix zoned and unzoned
            timestamps.
    """
    events_data: pd.DataFrame | None = None
    for case_concept in case_concepts:
        rel_data = _case_relations(data, case_concept)
        if events_data is None:
            events_data = _event_frame(data)
        log = _flatten(case_concept, rel_data, events_data)
        measured = count_phrase(
            [
                (len(log), "event", "events"),
                (log["case:concept:name"].nunique(), "object", "objects"),
            ],
            keep_zero=True,
        )
        logger.info("  Extracted flattened log (%s) with %s", case_concept, measured)
        yield case_concept, log


def case_concept_candidates(data: ReadableStore) -> list[str]:
    """Return the object types this store can be projected onto.

    Returns:
        The object types named by at least one event-to-object relation, in
        the order the store holds them.
    """
    reached = {
        t
        for relation in data.relations_of(RelationKind.E2O).values()
        for t in relation.to_type
    }
    return [t for t in data.types(EntityKind.OBJECT) if t in reached]


def projection_advice(data: ReadableStore) -> str:
    """Return the sentence that tells a user which object types `case_concepts` can name."""
    candidates = case_concept_candidates(data)
    if not candidates:
        return (
            "No event-to-object relation of this log names an object type, so "
            "case_concepts cannot name one."
        )
    return f"Set case_concepts to one of: {', '.join(sorted(candidates))}."


def _rank_case_concepts(data: ReadableStore, n: int) -> list[str]:
    """Return the n `case_concept_candidates` with the most instances."""
    reachable = set(case_concept_candidates(data))
    ranked = data.most_common_objects(n=len(data.types(EntityKind.OBJECT)))
    return [obj_type for obj_type in ranked if obj_type in reachable][:n]


def _follow_renames(data: ReadableStore, case_concepts: Sequence[str]) -> list[str]:
    """Return the case concepts under the names the store holds them.

    A name that matches no object type and no rename is returned unchanged.
    """
    object_types = set(data.types(EntityKind.OBJECT))
    renames = object_type_renames(data)
    chosen = []
    for case_concept in case_concepts:
        moved = renames.get(case_concept)
        if case_concept in object_types or moved is None:
            chosen.append(case_concept)
            continue
        logger.info(
            "  case concept %s is called %s after cleaning", case_concept, moved
        )
        chosen.append(moved)
    return chosen


def choose_case_concepts(
    data: ReadableStore, case_concepts: Sequence[str] | None, auto: int = 3
) -> tuple[list[str], str]:
    """Take the given case concepts, or pick the most common object types.

    Args:
        case_concepts: Object types as the source log spells them, or None to
            choose them.
        auto: How many to choose when `case_concepts` is None.

    Returns:
        The chosen types and the source of the choice, `config` or
        `auto(n=...)`, for the build log.
    """
    if case_concepts is not None:
        return _follow_renames(data, case_concepts), "config"
    chosen = _rank_case_concepts(data, auto)
    if not chosen and data.types(EntityKind.OBJECT):
        logger.warning(
            "No event-to-object relation names an object type of this log. "
            "No case concept was chosen."
        )
    return chosen, f"auto(n={auto})"


def _case_relations(data: ReadableStore, case_concept: str) -> pd.DataFrame:
    """Return the distinct event-object pairs of relations reaching a case concept."""
    if case_concept not in data.types(EntityKind.OBJECT):
        raise InputError(
            f"case_concepts names {case_concept!r}, which is not an object type "
            f"of this log. {projection_advice(data)}"
        )

    relations = data.relations_of(RelationKind.E2O)
    data_list = [
        relation.data.rename(
            columns={relation.to_id: case_concept, relation.from_id: "event_id"}
        )
        for relation in relations.values()
        if case_concept in relation.to_type
    ]
    if not data_list:
        raise InputError(
            f"case_concepts names {case_concept!r}, but no event-to-object "
            f"relation reaches it. {projection_advice(data)}"
        )

    rel_data = pd.concat(data_list, ignore_index=True)
    pairs = rel_data.drop_duplicates(["event_id", case_concept])
    if len(pairs) < len(rel_data):
        logger.debug(
            "  %s relation row(s) reaching %s are duplicate event-object pairs",
            len(rel_data) - len(pairs),
            case_concept,
        )
    rel_data = pairs
    valid_case_ids = set(data.ids(EntityKind.OBJECT, case_concept))
    dropped_ids = set(rel_data[case_concept]) - valid_case_ids
    if dropped_ids:
        _warn_unknown_case_ids(data, case_concept, dropped_ids)
    return rel_data[rel_data[case_concept].isin(valid_case_ids)]


def _event_frame(data: ReadableStore) -> pd.DataFrame:
    """Return every event as one frame of id, activity, timestamp and position.

    The position numbers the events by their position column, and the
    events of activities without one after them in store order.

    Raises:
        InputError: If the log has no events, or its activities mix zoned and
            unzoned timestamps.
    """
    events_data_list = []
    zoned: dict[bool, list[str]] = {True: [], False: []}
    for event_type in data.types(EntityKind.EVENT):
        event_table = data.table(EntityKind.EVENT, event_type)
        event_data = event_table.data[[event_table.id, event_table.time]].copy()
        event_data["concept:name"] = event_type
        event_data = event_data.rename(
            columns={event_table.id: "event_id", event_table.time: "time:timestamp"}
        )
        event_data[_POSITION] = (
            float("nan")
            if event_table.position is None
            else event_table.data[event_table.position].to_numpy()
        )
        for is_zoned in _zone_kinds(event_data["time:timestamp"]):
            zoned[is_zoned].append(event_type)
        events_data_list.append(event_data)

    if not events_data_list:
        raise InputError(
            "The log has no events. Check that the input is the log to build "
            "and that it holds events."
        )
    if zoned[True] and zoned[False]:
        raise InputError(
            "The activities of this log mix zoned and unzoned timestamps. "
            f"With a zone: {first_five(sorted(zoned[True]))}. "
            f"Without one: {first_five(sorted(zoned[False]))}. "
            "Give every event table the same kind of timestamp before flattening."
        )
    events = pd.concat(events_data_list, ignore_index=True)
    events = events.sort_values(_POSITION, kind="mergesort", na_position="last")
    events[_POSITION] = range(len(events))
    return events


def _zone_kinds(stamps: pd.Series) -> set[bool]:
    """Return the set of whether each timestamp of one activity carries a zone."""
    if pd.api.types.is_datetime64_any_dtype(stamps):
        return {stamps.dt.tz is not None}
    return {getattr(value, "tzinfo", None) is not None for value in stamps.dropna()}


def _flatten(
    case_concept: str, rel_data: pd.DataFrame, events_data: pd.DataFrame
) -> pd.DataFrame:
    """Join events to their cases, sorted by case, timestamp and position in the log."""
    log = pd.merge(events_data, rel_data, on="event_id", how="inner")
    log = log.rename(columns={case_concept: "case:concept:name"})
    log = _drop_undated(log, case_concept)

    log = log.sort_values(by=["case:concept:name", "time:timestamp", _POSITION])
    log = log.set_index("event_id")
    return log[["case:concept:name", "concept:name", "time:timestamp"]]


def _drop_undated(log: pd.DataFrame, case_concept: str) -> pd.DataFrame:
    """Drop the events with no timestamp, and log how many there were."""
    dated = log[log["time:timestamp"].notna()]
    undated = len(log) - len(dated)
    if undated:
        logger.warning(
            "%s event(s) have no timestamp and are left out of the event order of %s.",
            undated,
            case_concept,
        )
    return dated


def _warn_unknown_case_ids(
    data: ReadableStore, case_concept: str, dropped_ids: set[object]
) -> None:
    """Warn about referenced case ids that are in no object table.

    An id of another object type is not reported.
    """
    all_object_ids: set[object] = set()
    for table in data.tables_of(EntityKind.OBJECT).values():
        all_object_ids.update(table.data[table.id].unique())

    genuinely_missing = dropped_ids - all_object_ids
    if genuinely_missing:
        logger.warning(
            "%s referenced object id(s) are in no object "
            "table. The %s relation rows naming them are dropped "
            "(sample: %s).",
            len(genuinely_missing),
            case_concept,
            sorted(str(x) for x in genuinely_missing)[:5],
        )
