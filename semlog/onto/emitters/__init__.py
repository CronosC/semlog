"""The build steps that turn stored data into axioms.

Each emitter writes one part of the ontology through the sink. `EntityEmitter`
and `HistoryEmitter` call `AttributeEmitter` for attribute columns, and
`HistoryEmitter` calls `TimeEmitter`.
"""

from semlog.onto.emitters.attributes import AttributeEmitter
from semlog.onto.emitters.base import Emitter
from semlog.onto.emitters.build_record import BuildRecordEmitter
from semlog.onto.emitters.entities import EntityEmitter
from semlog.onto.emitters.history import HistoryEmitter
from semlog.onto.emitters.order import OrderEmitter
from semlog.onto.emitters.relations import RelationEmitter
from semlog.onto.emitters.tbox import TBoxEmitter
from semlog.onto.emitters.time import TimeEmitter

__all__ = [
    "AttributeEmitter",
    "BuildRecordEmitter",
    "Emitter",
    "EntityEmitter",
    "HistoryEmitter",
    "OrderEmitter",
    "RelationEmitter",
    "TBoxEmitter",
    "TimeEmitter",
]
