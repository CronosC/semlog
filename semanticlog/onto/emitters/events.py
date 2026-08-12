"""Events, as one class per activity and one individual per event."""

from __future__ import annotations

from semanticlog.core import vocabulary as vocab
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.emitters.base import _EntityEmitter


class EventEmitter(_EntityEmitter):
    """Emits the activity classes and the events themselves."""

    def emit(self, data: OCEDStore) -> None:
        """Emit every activity and its events.

        All classes are declared before any individual is asserted, so that a
        name used for both is caught while the sink can still say which two
        sources claimed it. The timestamp column is left to the time emitter,
        and every other column is an attribute.
        """
        self.annotations.bind(data.backmap)
        activities = data.get_event_types()

        for activity in activities:
            self.sink.add_class(activity, vocab.EVENT, source=f"activity {activity!r}")
        for activity in activities:
            self.annotations.activity_class(activity)
        for activity in activities:
            table = data.get_event_table(activity)
            self.sink.assert_individuals(list(table.data[table.id]), activity)
            self.annotations.entity_individuals(activity, table.data[table.id])
            self.attributes.emit_columns(
                table.data.drop(columns=[table.time]),
                table.id,
                vocab.EVENT_ATTRIBUTE,
            )
