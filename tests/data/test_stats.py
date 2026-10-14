import hashlib

from semanticlog.data.stats import file_digest, rows_by_kind, store_stats
from semanticlog.data.tables import EntityKind


def test_a_file_digest_is_the_sha256_of_its_bytes(tmp_path):
    path = tmp_path / "log.jsonocel"
    path.write_bytes(b"x" * 3_000_000)

    assert file_digest(path) == hashlib.sha256(b"x" * 3_000_000).hexdigest()


def test_each_entity_type_is_counted_under_its_kind(tiny_store):
    rows = rows_by_kind(tiny_store)

    assert rows == {EntityKind.EVENT: [("step", 3)], EntityKind.OBJECT: [("order", 1)]}


def test_the_store_counts_cover_types_entities_and_both_relation_families(
    tiny_store,
):
    assert store_stats(tiny_store) == {
        "activities": 1,
        "events": 3,
        "object_types": 1,
        "objects": 1,
        "event_to_object_relations": 3,
        "object_to_object_relations": 0,
    }
