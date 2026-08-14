"""Projection of object-centric data onto one case notion.

Event order and DECLARE mining both need a classical log, where every event
belongs to exactly one case. Flattening produces that view for a chosen object
type: each object of that type becomes a case, and every event related to it
becomes a row of that case. An event related to several objects appears in
several cases, which is the usual consequence of flattening.

Two sorts of row do not survive the projection: an event related to one object
more than once, which the case holds only once, and an event with no timestamp,
which has no position in an order. Both are reported.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator

import pandas as pd

from semanticlog.core.summary import count_phrase
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind

logger = logging.getLogger(__name__)


def get_log(data: OCEDStore, case_concept: str) -> pd.DataFrame:
    """Return the flat log for one case concept.

    Args:
        data: Store to project.
        case_concept: Object type whose instances become cases.

    Returns:
        A frame indexed by event id, with the pm4py columns
        `case:concept:name`, `concept:name` and `time:timestamp`, sorted by
        case and time.

    Raises:
        ValueError: If the type is unknown, no relation reaches it, or the
            store holds no events.
    """
    return next(frame for _, frame in iter_logs(data, [case_concept]))


def iter_logs(
    data: OCEDStore, case_concepts: Iterable[str]
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield each case concept with its flat log.

    The event table is assembled once and reused across case concepts, so
    several projections of one store cost less than the same number of
    `get_log` calls.

    Args:
        data: Store to project.
        case_concepts: Object types whose instances become cases.

    Yields:
        Pairs of case concept and its flat log, each sorted by case and then by
        time.

    Raises:
        ValueError: If a type is unknown, no relation reaches it, or the store
            holds no events.
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
        logger.info(f"  Extracted flattened log ({case_concept}) with {measured}")
        yield case_concept, log


def case_concept_candidates(data: OCEDStore) -> list[str]:
    """Return the object types this store can be projected onto.

    A case concept needs events, and events reach objects through the
    event-to-object relations, so a type no such relation names has no trace
    to order.

    Args:
        data: Store to inspect.

    Returns:
        The object types named by at least one event-to-object relation, in
        the order the store holds them.
    """
    reached = {t for relation in data.get_e2o().values() for t in relation.to_type}
    return [t for t in data.get_object_types() if t in reached]


def _case_relations(data: OCEDStore, case_concept: str) -> pd.DataFrame:
    """Collect the event-to-object rows that reach this case concept.

    Every relation naming the type contributes, so an event linked to one
    object under two qualifiers arrives twice. A case holds an event once
    however many relations connect them, so the pairs are reduced here.
    """
    if case_concept not in data.get_object_types():
        raise ValueError(f"Unknown object type: {case_concept}")

    relations = data.get_e2o()
    data_list = [
        relation.data.rename(
            columns={relation.to_id: case_concept, relation.from_id: "event_id"}
        )
        for relation in relations.values()
        if case_concept in relation.to_type
    ]
    if not data_list:
        raise ValueError(
            f"No event-to-object relations reference object type '{case_concept}'."
        )

    rel_data = pd.concat(data_list, ignore_index=True)
    pairs = rel_data.drop_duplicates(["event_id", case_concept])
    if len(pairs) < len(rel_data):
        logger.debug(
            f"  get_log[{case_concept}]: {len(rel_data) - len(pairs)} relation row(s) "
            "repeat a pair the projection already holds"
        )
    rel_data = pairs
    valid_case_ids = set(data.get_ids(EntityKind.OBJECT, case_concept))
    dropped_ids = set(rel_data[case_concept]) - valid_case_ids
    if dropped_ids:
        _warn_unknown_case_ids(data, case_concept, dropped_ids)
    return rel_data[rel_data[case_concept].isin(valid_case_ids)]


def _event_frame(data: OCEDStore) -> pd.DataFrame:
    """Stack every activity into one frame of id, activity and timestamp."""
    events_data_list = []
    for event_type in data.get_event_types():
        event_table = data.get_event_table(event_type)
        event_data = event_table.data[[event_table.id, event_table.time]].copy()
        event_data["concept:name"] = event_type
        event_data = event_data.rename(
            columns={event_table.id: "event_id", event_table.time: "time:timestamp"}
        )
        events_data_list.append(event_data)

    if not events_data_list:
        raise ValueError("No event data available.")
    return pd.concat(events_data_list, ignore_index=True)


def _flatten(
    case_concept: str, rel_data: pd.DataFrame, events_data: pd.DataFrame
) -> pd.DataFrame:
    """Join events to their cases and put the result in pm4py column form."""
    log = pd.merge(events_data, rel_data, on="event_id", how="inner")
    log = log.rename(columns={case_concept: "case:concept:name"})
    log = _drop_undated(log, case_concept)

    log = log.set_index("event_id")
    return log[["case:concept:name", "concept:name", "time:timestamp"]].sort_values(
        by=["case:concept:name", "time:timestamp"], ascending=True
    )


def _drop_undated(log: pd.DataFrame, case_concept: str) -> pd.DataFrame:
    """Drop the events with no timestamp, saying how many there were.

    Order is what a projection exists to express, and an event with no moment
    has no place in it. Those events keep their class and their relations in
    the ontology, so the loss is only of their position.
    """
    dated = log[log["time:timestamp"].notna()]
    undated = len(log) - len(dated)
    if undated:
        logger.warning(
            f"get_log[{case_concept}]: {undated} event(s) have no timestamp and take "
            "no place in the event order. They are still emitted as events, with "
            "their attributes and relations."
        )
    return dated


def _warn_unknown_case_ids(
    data: OCEDStore, case_concept: str, dropped_ids: set
) -> None:
    """Warn about referenced case ids that are in no object table.

    An id belonging to another object type is left alone. Projecting onto one
    case notion excludes those by definition.
    """
    all_object_ids: set = set()
    for table in data.get_object_tables().values():
        all_object_ids.update(table.data[table.id].unique())

    genuinely_missing = dropped_ids - all_object_ids
    if genuinely_missing:
        logger.warning(
            f"get_log[{case_concept}]: {len(genuinely_missing)} referenced object "
            "id(s) are in no object table at all, so the relation rows naming them "
            f"are dropped (sample: {sorted(str(x) for x in genuinely_missing)[:5]})"
        )
