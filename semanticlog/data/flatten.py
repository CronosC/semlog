"""Projection of object-centric data onto one case notion.

Event order and DECLARE mining both need a classical log, where every event
belongs to exactly one case. Flattening produces that view for a chosen object
type: each object of that type becomes a case, and every event related to it
becomes a row of that case. An event related to several objects appears in
several cases, which is the usual consequence of flattening.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator

import pandas as pd
import pm4py as pm

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


def get_logs(data: OCEDStore, case_concepts: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Return one flat log per case concept, keyed by case concept."""
    return dict(iter_logs(data, case_concepts))


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
        Pairs of case concept and its flat log.

    Raises:
        ValueError: If a type is unknown, no relation reaches it, or the store
            holds no events.
    """
    events_data: pd.DataFrame | None = None
    for case_concept in case_concepts:
        rel_data = _case_relations(data, case_concept)
        if events_data is None:
            events_data = _event_frame(data)
        yield case_concept, _flatten(case_concept, rel_data, events_data)


def _case_relations(data: OCEDStore, case_concept: str) -> pd.DataFrame:
    """Collect the event-to-object rows that reach this case concept."""
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

    log = pm.format_dataframe(
        log,
        case_id=case_concept,
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )

    log = log.set_index("event_id")
    return log[["case:concept:name", "concept:name", "time:timestamp"]].sort_values(
        by=["case:concept:name", "time:timestamp"], ascending=True
    )


def _warn_unknown_case_ids(
    data: OCEDStore, case_concept: str, dropped_ids: set
) -> None:
    """Report dropped case ids, separating other types from unknown ids."""
    all_object_ids: set = set()
    for table in data.get_object_tables().values():
        all_object_ids.update(table.data[table.id].unique())

    genuinely_missing = dropped_ids - all_object_ids
    other_types = dropped_ids - genuinely_missing
    if other_types:
        logger.info(
            f"get_log[{case_concept}]: {len(other_types)} referenced id(s) belong "
            "to another object type and are outside this projection."
        )
    if genuinely_missing:
        logger.warning(
            f"get_log[{case_concept}]: {len(genuinely_missing)} referenced object "
            "id(s) are in no object table at all, so the relation rows naming them "
            f"are dropped (sample: {sorted(str(x) for x in genuinely_missing)[:5]})"
        )
