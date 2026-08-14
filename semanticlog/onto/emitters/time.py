"""Time, as one individual per distinct point in time.

Timestamps become individuals rather than plain literals on their events, so
that several events at the same moment share one time individual and a query
can reach the events of a moment from the moment itself. The literal value is
kept on the individual as well.
"""

from __future__ import annotations

import pandas as pd

from semanticlog.core import vocabulary as vocab
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.annotations import timestamp_labels
from semanticlog.onto.emitters.base import Emitter


class TimeEmitter(Emitter):
    """Emits time individuals and links events to them.

    It remembers which moments it has already written, so that the same
    timestamp reached from several activities, or later from the attribute
    history, is emitted once.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._written: set[str] = set()

    def emit_individuals(self, ids: pd.Series, stamps: pd.Series) -> None:
        """Emit the time individuals that have not been emitted yet.

        An entity whose timestamp is missing has no moment to name, so those
        rows are dropped here and the entity is left without a time.

        Args:
            ids: Individual names of the moments.
            stamps: The timestamps themselves, aligned with the ids.
        """
        frame = pd.DataFrame({"time_str": ids.to_numpy(), "time": stamps.to_numpy()})
        frame = frame[frame["time_str"].notna()]
        fresh = frame[~frame["time_str"].isin(self._written)].drop_duplicates(
            "time_str"
        )
        if fresh.empty:
            return
        self._written.update(fresh["time_str"])
        self.sink.assert_individuals(fresh["time_str"].to_list(), vocab.TIME)
        self.annotations.time_individuals(
            fresh["time_str"], timestamp_labels(fresh["time"])
        )
        self.sink.assert_relations(
            fresh,
            prop=vocab.HAS_DATA_VALUE,
            from_key="time_str",
            to_key="time",
            data_prop=True,
        )

    def emit(self, data: OCEDStore) -> None:
        """Emit the moments of every event, and link each event to its own."""
        self.annotations.bind(data.backmap)

        for activity in data.get_event_types():
            table = data.get_event_table(activity)
            df_time = table.data[[table.id, table.time]].rename(
                columns={table.id: "event", table.time: "time"}
            )
            df_time = df_time.assign(time_str=vocab.timestamp_ids(df_time["time"]))

            self.emit_individuals(df_time["time_str"], df_time["time"])
            self.sink.assert_relations(
                df_time,
                prop=vocab.OBSERVED_AT,
                from_key="event",
                to_key="time_str",
                inverse_name=vocab.OBSERVES,
            )
