"""The name space every claimant is allocated out of."""

from itertools import islice

import pytest

from semanticlog.core.names import NameRegistry, candidates


def _has(name: str) -> set[str]:
    return {f"has_{name}"}


def test_a_claimant_is_offered_its_bare_name_then_its_prefixed_one_then_a_counter():
    offered = list(islice(candidates("ship", "event_"), 4))

    assert offered == ["ship", "event_ship", "event_ship_2", "event_ship_3"]


def test_the_first_claimant_keeps_a_name_and_the_second_takes_its_prefix():
    registry = NameRegistry()

    first = registry.allocate("order", prefix="object_")
    second = registry.allocate("order", prefix="event_")

    assert (first, second) == ("order", "event_order")


def test_a_name_whose_prefixed_form_is_taken_is_numbered_past():
    registry = NameRegistry()
    registry.allocate("Event", prefix="event_")
    registry.allocate("object_Event", prefix="relation_")

    assert registry.allocate("Event", prefix="object_") == "object_Event_2"


def test_a_claim_takes_the_names_derived_from_it():
    registry = NameRegistry()

    registry.allocate("price", prefix="attribute_", derives=_has)

    assert registry.allocate("has_price", prefix="relation_") == "relation_has_price"


def test_a_candidate_whose_derived_name_is_taken_is_passed_over():
    registry = NameRegistry()
    registry.allocate("has_price", prefix="relation_")

    free = registry.allocate("price", prefix="attribute_", derives=_has)

    assert free == "attribute_price"


@pytest.mark.parametrize(
    ("name", "pending", "expected"),
    [
        ("Event", {"Event", "object_Event"}, "object_Event_2"),
        ("order", {"order", "item"}, "order"),
    ],
    ids=["held_for_another", "held_for_itself"],
)
def test_a_name_pending_for_a_later_claimant_is_not_taken_from_it(
    name, pending, expected
):
    registry = NameRegistry()
    registry.allocate("Event", prefix="event_")

    assert registry.allocate(name, prefix="object_", pending=pending) == expected


def test_a_candidate_the_claimant_cannot_be_emitted_under_is_passed_over():
    registry = NameRegistry()

    free = registry.allocate(
        "2fast", prefix="relation_", accepts=lambda n: n[0].isalpha()
    )

    assert free == "relation_2fast"


def test_a_prefix_no_candidate_passes_is_rejected():
    registry = NameRegistry()

    with pytest.raises(ValueError, match="Prefix '2_' gives '2_2fast'"):
        registry.allocate("2fast", prefix="2_", accepts=lambda n: n[0].isalpha())
