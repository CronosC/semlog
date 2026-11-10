"""The ledger of what each emitted name is declared as and which schema statements were written."""

from __future__ import annotations

import pytest

from semlog.core.enums import SinkEntity
from semlog.core.vocabulary import (
    NameCollisionError,
    UnsafeLocalNameError,
    is_safe_local_name,
)
from semlog.onto.sinks.ledger import NameLedger

CLASS = SinkEntity.CLASS
OBJECT = SinkEntity.OBJECT_PROPERTY
DATA = SinkEntity.DATA_PROPERTY


@pytest.fixture
def ledger() -> NameLedger:
    return NameLedger()


@pytest.mark.parametrize("name", ["bad name", "a+b", "has_a~b"])
def test_a_name_that_cannot_be_an_iri_local_name_is_rejected(ledger, name):
    """RDF/XML writes a name as an XML element, which takes no space, `+` or `~`."""
    with pytest.raises(UnsafeLocalNameError, match="cannot be an IRI local name"):
        ledger.declare(name, CLASS, "a test")


def test_an_unsafe_name_is_reported_in_the_terms_a_user_types(ledger):
    with pytest.raises(
        UnsafeLocalNameError, match="Rename it in the input log or the declare_file"
    ) as failure:
        ledger.declare("bad name", CLASS, "a test")

    assert "clean_oced_data" not in str(failure.value)
    assert "CSV" not in str(failure.value)


def test_a_property_name_starting_with_a_digit_is_reported_without_python_names(
    ledger,
):
    with pytest.raises(UnsafeLocalNameError, match="Rename it") as failure:
        ledger.declare("2modifies", OBJECT, "a test")

    assert "clean_oced_data" not in str(failure.value)


def test_a_hyphen_or_a_dot_is_accepted_in_a_name(ledger):
    ledger.declare("web-workers", CLASS, "a test")
    ledger.declare("has_unit.price", OBJECT, "a test")

    assert ledger.count == 2


@pytest.mark.parametrize("kind", [OBJECT, DATA])
def test_a_property_name_starting_with_a_digit_is_rejected(ledger, kind):
    """RDF/XML writes a property as an element name, which cannot start with a digit."""
    with pytest.raises(UnsafeLocalNameError, match="cannot be a property name"):
        ledger.declare("1st_step", kind, "a test")


def test_a_class_name_may_start_with_a_digit(ledger):
    ledger.declare("1st_step", CLASS, "a test")

    assert ledger.is_declared("1st_step")


def test_a_name_declared_again_as_the_same_kind_from_its_source_is_accepted(ledger):
    ledger.declare("order", CLASS, "an object type")
    ledger.declare("order", CLASS, "an object type")

    assert ledger.count == 1


@pytest.mark.parametrize(
    ("kind", "source"),
    [(CLASS, "an activity"), (OBJECT, "a relation type")],
)
def test_a_name_declared_again_as_another_entity_names_both_sources(
    ledger, kind, source
):
    """An activity and an object type of one name are two classes."""
    ledger.declare("order", CLASS, "an object type")

    with pytest.raises(NameCollisionError, match="as class from an object type") as e:
        ledger.declare("order", kind, source)

    assert f"as {kind.value} from {source}" in str(e.value)


def test_an_individual_named_like_a_class_is_accepted(ledger):
    ledger.declare("order", CLASS, "an object type")

    assert ledger.check_individuals(["o1", "order"], "a test") == ["o1", "order"]
    assert is_safe_local_name("order") is True


def test_an_unsafe_individual_name_is_rejected(ledger):
    with pytest.raises(UnsafeLocalNameError, match="not a name"):
        ledger.check_individuals(["e1", "not a name"], "a test")


def test_individuals_are_deduplicated_by_the_name_they_are_emitted_under(ledger):
    assert ledger.check_individuals([1, "1", 2, None], "a test") == ["1", "2"]


def test_two_properties_cannot_share_one_inverse_name(ledger):
    ledger.claim_inverse("hasChild", "hasParent", "a test")

    with pytest.raises(NameCollisionError, match="is the inverse of 'hasChild'"):
        ledger.claim_inverse("hasMaleChild", "hasParent", "a test")

    assert ledger.inverse_name("hasParent") == "hasChild"


def test_declared_names_are_counted_by_kind(ledger):
    ledger.declare("order", CLASS, "a test")
    ledger.declare("concerns", OBJECT, "a test")
    ledger.declare("has_data_value", DATA, "a test")

    assert ledger.by_kind() == {
        "classes": 1,
        "object_properties": 1,
        "data_properties": 1,
        "annotation_properties": 0,
    }


def test_names_declared_after_a_mark_are_grouped_by_kind_in_order(ledger):
    ledger.declare("order", CLASS, "a test")
    mark = ledger.count
    ledger.declare("item", CLASS, "a test")
    ledger.declare("shipment", CLASS, "a test")
    ledger.declare("concerns", OBJECT, "a test")

    assert ledger.since(mark) == {"class": ["item", "shipment"], "object": ["concerns"]}
