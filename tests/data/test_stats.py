"""Measurements of a loaded store."""

from semanticlog.data.stats import rows_by_kind
from semanticlog.data.tables import EntityKind


def test_each_entity_type_is_counted_under_its_kind(tiny_store):
    rows = rows_by_kind(tiny_store)

    assert rows == {EntityKind.EVENT: [("step", 3)], EntityKind.OBJECT: [("order", 1)]}
