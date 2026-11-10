"""The record of the source name behind each cleaned name."""

from semanticlog.data.cleaning.renames import RenameKind, RenameLog


def test_successive_renames_map_the_last_name_to_the_source_name():
    log = RenameLog()
    log.record(RenameKind.ACTIVITY, "Has Price", "Has_Price")
    log.record(RenameKind.ACTIVITY, "Has_Price", "Has_Price_")
    log.record(RenameKind.OBJECT_TYPE, "item", "item")

    assert log.backmap() == {"activity": {"Has_Price_": "Has Price"}}


def test_renames_are_listed_by_kind_with_one_name_kept_apart_per_kind():
    log = RenameLog()
    log.record(RenameKind.ID, "o 1", "o_1")
    log.record(RenameKind.ACTIVITY, "o.1", "o_1")
    log.record(RenameKind.OBJECT_TYPE, "item", "item")
    log.record_value("color", "off_white", "off white")

    assert log.by_kind() == {
        "activity": ["o.1 -> o_1"],
        "id": ["o 1 -> o_1"],
        "value": ["off white -> color.off_white"],
    }


def test_a_value_is_named_under_its_column_and_follows_the_column_when_renamed():
    log = RenameLog()
    log.record_value("event", "by_hand", "by hand")

    log.rename_value_column("event", "attribute_event")

    assert log.backmap() == {"value": {"attribute_event.by_hand": "by hand"}}
