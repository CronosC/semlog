"""Time, as one individual per distinct timestamp.

Events with the same timestamp share one time individual. The individual
carries the literal value.
"""

from __future__ import annotations

import pandas as pd

from semanticlog.core import vocabulary as vocab
from semanticlog.data.readable import ReadableStore
from semanticlog.data.tables import EntityKind
from semanticlog.onto.annotations import timestamp_labels
from semanticlog.onto.emitters.base import Emitter


class TimeEmitter(Emitter):
    """Emits time individuals and links events to them.

    `emitted.timestamps` records which are written, so the attribute history
    does not write a timestamp twice.
    """

    def emit_individuals(self, ids: pd.Series, stamps: pd.Series) -> None:
        """Emit the time individuals not yet emitted.

        Rows with a missing timestamp are dropped.

        Args:
            ids: Individual name per timestamp.
            stamps: Timestamps, aligned with `ids`.
        """
        written = self.emitted.timestamps
        frame = pd.DataFrame({"time_str": ids.to_numpy(), "time": stamps.to_numpy()})
        frame = frame[frame["time_str"].notna()]
        fresh = frame[~frame["time_str"].isin(written)].drop_duplicates("time_str")
        if fresh.empty:
            return
        written.update(fresh["time_str"])
        self.sink.assert_individuals(vocab.TIME, fresh["time_str"])
        self.annotations.time_individuals(
            fresh["time_str"], timestamp_labels(fresh["time"])
        )
        self.assert_data(vocab.HAS_DATA_VALUE, fresh, "time_str", "time")

    def emit(self, data: ReadableStore) -> None:
        """Emit the timestamp of every event and link each event to it."""

        for activity in data.types(EntityKind.EVENT):
            table = data.table(EntityKind.EVENT, activity)
            df_time = table.data[[table.id, table.time]].rename(
                columns={table.id: "event", table.time: "time"}
            )
            df_time = df_time.assign(time_str=vocab.timestamp_ids(df_time["time"]))

            self.emit_individuals(df_time["time_str"], df_time["time"])
            self.assert_relations(
                vocab.OBSERVED_AT,
                df_time,
                "event",
                "time_str",
                inverse_name=vocab.OBSERVES,
            )
